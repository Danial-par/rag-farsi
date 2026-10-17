from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from aiobotocore.session import get_session  # type: ignore[import-untyped]
from botocore.exceptions import BotoCoreError, ClientError  # type: ignore[import-untyped]


class ObjectStoreError(RuntimeError):
    """An immutable source object could not be loaded safely."""


class ObjectReader(Protocol):
    async def get(self, object_key: str) -> bytes: ...


class ObjectWriter(Protocol):
    async def put(self, object_key: str, payload: bytes, *, content_type: str) -> None: ...

    async def delete(self, object_key: str) -> None: ...

    async def list(
        self, prefix: str, *, limit: int = 500, continuation_token: str | None = None
    ) -> StoredObjectPage: ...


class ObjectStore(ObjectReader, ObjectWriter, Protocol):
    """The combined source object capability used by the API and worker."""


@dataclass(frozen=True, slots=True)
class StoredObject:
    key: str
    last_modified: datetime


@dataclass(frozen=True, slots=True)
class StoredObjectPage:
    objects: list[StoredObject]
    continuation_token: str | None


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

    async def put(self, object_key: str, payload: bytes, *, content_type: str) -> None:
        del content_type
        candidate = self._path(object_key)
        try:
            await asyncio.to_thread(self._write_immutable, candidate, payload)
        except FileExistsError as exc:
            raise ObjectStoreError("Immutable source object key already exists") from exc
        except OSError as exc:
            raise ObjectStoreError("Local source object could not be written") from exc

    async def delete(self, object_key: str) -> None:
        candidate = self._path(object_key)
        try:
            await asyncio.to_thread(candidate.unlink, missing_ok=True)
        except OSError as exc:
            raise ObjectStoreError("Local source object could not be deleted") from exc

    async def list(
        self, prefix: str, *, limit: int = 500, continuation_token: str | None = None
    ) -> StoredObjectPage:
        if limit < 1:
            raise ValueError("Object listing limit must be positive")
        directory = self._path(prefix)
        if not directory.exists():
            return StoredObjectPage(objects=[], continuation_token=None)
        try:
            paths = await asyncio.to_thread(lambda: list(directory.rglob("*.pdf")))
            result = []
            for path in paths:
                if path.is_file():
                    modified = await asyncio.to_thread(path.stat)
                    result.append(
                        StoredObject(
                            key=path.relative_to(self._root).as_posix(),
                            last_modified=datetime.fromtimestamp(modified.st_mtime, UTC),
                        )
                    )
            ordered = sorted(result, key=lambda item: item.key)
            if continuation_token is not None:
                ordered = [item for item in ordered if item.key > continuation_token]
            page = ordered[:limit]
            next_token = page[-1].key if len(ordered) > limit else None
            return StoredObjectPage(objects=page, continuation_token=next_token)
        except OSError as exc:
            raise ObjectStoreError("Local source objects could not be listed") from exc

    def _path(self, object_key: str) -> Path:
        candidate = (self._root / object_key).resolve()
        if not candidate.is_relative_to(self._root):
            raise ObjectStoreError("Object key escapes the configured local storage root")
        return candidate

    @staticmethod
    def _write_immutable(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)


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

    async def put(self, object_key: str, payload: bytes, *, content_type: str) -> None:
        try:
            async with self._session.create_client(
                "s3",
                endpoint_url=self._endpoint_url,
                region_name=self._region,
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
            ) as client:
                await client.put_object(
                    Bucket=self._bucket,
                    Key=object_key,
                    Body=payload,
                    ContentType=content_type,
                    IfNoneMatch="*",
                )
        except (BotoCoreError, ClientError, OSError) as exc:
            raise ObjectStoreError("S3 source object could not be written") from exc

    async def delete(self, object_key: str) -> None:
        try:
            async with self._session.create_client(
                "s3",
                endpoint_url=self._endpoint_url,
                region_name=self._region,
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
            ) as client:
                await client.delete_object(Bucket=self._bucket, Key=object_key)
        except (BotoCoreError, ClientError, OSError) as exc:
            raise ObjectStoreError("S3 source object could not be deleted") from exc

    async def list(
        self, prefix: str, *, limit: int = 500, continuation_token: str | None = None
    ) -> StoredObjectPage:
        if limit < 1:
            raise ValueError("Object listing limit must be positive")
        try:
            async with self._session.create_client(
                "s3",
                endpoint_url=self._endpoint_url,
                region_name=self._region,
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
            ) as client:
                request: dict[str, Any] = {
                    "Bucket": self._bucket,
                    "Prefix": prefix,
                    "MaxKeys": min(limit, 1000),
                }
                if continuation_token is not None:
                    request["ContinuationToken"] = continuation_token
                page = await client.list_objects_v2(**request)
                objects = []
                for item in page.get("Contents", []):
                    modified = item.get("LastModified")
                    key = item.get("Key")
                    if isinstance(key, str) and isinstance(modified, datetime):
                        objects.append(
                            StoredObject(key=key, last_modified=modified.astimezone(UTC))
                        )
                next_token = page.get("NextContinuationToken")
                return StoredObjectPage(
                    objects=objects,
                    continuation_token=next_token if isinstance(next_token, str) else None,
                )
        except (BotoCoreError, ClientError, OSError) as exc:
            raise ObjectStoreError("S3 source objects could not be listed") from exc
