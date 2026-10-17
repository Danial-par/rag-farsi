from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Protocol

from aiobotocore.session import get_session  # type: ignore[import-untyped]
from botocore.exceptions import BotoCoreError, ClientError  # type: ignore[import-untyped]


class ObjectStoreError(RuntimeError):
    """An immutable source object could not be loaded safely."""


class ObjectStore(Protocol):
    async def get(self, object_key: str) -> bytes: ...


class LocalObjectStore:
    """Read fixture objects below one configured root without permitting traversal."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    async def get(self, object_key: str) -> bytes:
        candidate = (self._root / object_key).resolve()
        if not candidate.is_relative_to(self._root) or not candidate.is_file():
            raise ObjectStoreError("Local source object is unavailable")
        try:
            return await asyncio.to_thread(candidate.read_bytes)
        except OSError as exc:
            raise ObjectStoreError("Local source object could not be read") from exc


class S3ObjectStore:
    """Load private immutable objects from an S3-compatible bucket asynchronously."""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None,
        region: str | None,
        access_key_id: str | None,
        secret_access_key: str | None,
    ) -> None:
        if not bucket.strip():
            raise ValueError("S3_BUCKET is required for the S3 object store")
        self._bucket = bucket
        self._endpoint_url = endpoint_url
        self._region = region
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session: Any = get_session()

    async def get(self, object_key: str) -> bytes:
        if not object_key.strip():
            raise ObjectStoreError("Source object key is empty")
        try:
            async with self._session.create_client(
                "s3",
                endpoint_url=self._endpoint_url,
                region_name=self._region,
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
            ) as client:
                response = await client.get_object(Bucket=self._bucket, Key=object_key)
                async with response["Body"] as body:
                    payload = await body.read()
        except (BotoCoreError, ClientError, OSError) as exc:
            raise ObjectStoreError("S3 source object could not be loaded") from exc
        if not isinstance(payload, bytes) or not payload:
            raise ObjectStoreError("S3 source object is empty")
        return payload
