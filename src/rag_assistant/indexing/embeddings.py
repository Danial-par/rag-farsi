from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol

import httpx


class EmbeddingError(RuntimeError):
    """Base class for safe embedding failures."""


class RetryableEmbeddingError(EmbeddingError):
    """A transient embedding failure that may be retried."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class EmbeddingTimeoutError(RetryableEmbeddingError):
    """The embedding provider did not answer within the configured timeout."""


class PermanentEmbeddingError(EmbeddingError):
    """A non-retryable provider request or response failure."""


class EmbeddingDimensionError(PermanentEmbeddingError):
    """An embedding response does not match the indexed vector configuration."""


class EmbeddingClient(Protocol):
    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int,
    ) -> list[list[float]]: ...


class OpenRouterEmbeddingClient:
    """Call OpenRouter's OpenAI-compatible embeddings endpoint without hidden retries."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for embedding generation")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(
                connect=connect_timeout_seconds,
                read=read_timeout_seconds,
                write=read_timeout_seconds,
                pool=connect_timeout_seconds,
            ),
        )

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int,
    ) -> list[list[float]]:
        if not texts or any(not text.strip() for text in texts):
            raise PermanentEmbeddingError("Embedding inputs must be non-empty")
        try:
            response = await self._client.post(
                "/embeddings",
                json={"model": model, "input": list(texts), "dimensions": dimensions},
            )
        except httpx.TimeoutException as exc:
            raise EmbeddingTimeoutError("Embedding request timed out") from exc
        except httpx.TransportError as exc:
            raise RetryableEmbeddingError("Embedding transport failed") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise RetryableEmbeddingError(
                "Embedding provider is temporarily unavailable",
                retry_after_seconds=_retry_after(response.headers.get("Retry-After")),
            )
        if response.status_code >= 400:
            raise PermanentEmbeddingError(
                f"Embedding provider rejected the request with HTTP {response.status_code}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise PermanentEmbeddingError("Embedding provider returned invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise PermanentEmbeddingError("Embedding response has an invalid shape")

        indexed_vectors: list[tuple[int, list[float]]] = []
        for position, item in enumerate(payload["data"]):
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise PermanentEmbeddingError("Embedding response item has an invalid shape")
            index = item.get("index", position)
            if not isinstance(index, int):
                raise PermanentEmbeddingError("Embedding response index is invalid")
            vector_values = item["embedding"]
            if any(not isinstance(value, int | float) for value in vector_values):
                raise PermanentEmbeddingError("Embedding vector contains a non-number")
            indexed_vectors.append((index, [float(value) for value in vector_values]))
        indexed_vectors.sort(key=lambda value: value[0])
        if [index for index, _ in indexed_vectors] != list(range(len(indexed_vectors))):
            raise PermanentEmbeddingError("Embedding response indices are incomplete")
        return [vector for _, vector in indexed_vectors]

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


@dataclass(frozen=True, slots=True)
class EmbeddingBatchPolicy:
    batch_size: int = 64
    concurrency: int = 2
    max_retries: int = 3
    retry_base_seconds: float = 1.0
    maximum_retry_after_seconds: float = 60.0


async def embed_in_batches(
    client: EmbeddingClient,
    texts: Sequence[str],
    *,
    model: str,
    dimensions: int,
    policy: EmbeddingBatchPolicy,
    on_batch_completed: Callable[[], Awaitable[None]] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    random_value: Callable[[], float] = random.random,
) -> list[list[float]]:
    """Embed ordered texts concurrently while retaining response order and strict dimensions."""
    if not texts or any(not text.strip() for text in texts):
        raise PermanentEmbeddingError("At least one non-empty embedding input is required")
    if policy.batch_size <= 0 or policy.concurrency <= 0 or policy.max_retries < 0:
        raise ValueError("Embedding batch policy is invalid")

    batches = [
        (offset, texts[offset : offset + policy.batch_size])
        for offset in range(0, len(texts), policy.batch_size)
    ]
    semaphore = asyncio.Semaphore(policy.concurrency)

    async def run_batch(offset: int, batch: Sequence[str]) -> tuple[int, list[list[float]]]:
        async with semaphore:
            vectors = await _embed_with_retries(
                client,
                batch,
                model=model,
                dimensions=dimensions,
                policy=policy,
                sleep=sleep,
                random_value=random_value,
            )
            if len(vectors) != len(batch):
                raise PermanentEmbeddingError("Embedding response count does not match the request")
            for vector in vectors:
                if len(vector) != dimensions:
                    raise EmbeddingDimensionError(
                        f"Embedding vector has {len(vector)} dimensions; expected {dimensions}"
                    )
                if any(not math.isfinite(value) for value in vector):
                    raise PermanentEmbeddingError("Embedding vector contains a non-finite value")
            if on_batch_completed is not None:
                await on_batch_completed()
            return offset, vectors

    completed = await asyncio.gather(*(run_batch(offset, batch) for offset, batch in batches))
    completed.sort(key=lambda value: value[0])
    return [vector for _, vectors in completed for vector in vectors]


async def _embed_with_retries(
    client: EmbeddingClient,
    texts: Sequence[str],
    *,
    model: str,
    dimensions: int,
    policy: EmbeddingBatchPolicy,
    sleep: Callable[[float], Awaitable[None]],
    random_value: Callable[[], float],
) -> list[list[float]]:
    for attempt in range(policy.max_retries + 1):
        try:
            return await client.embed(texts, model=model, dimensions=dimensions)
        except RetryableEmbeddingError as exc:
            if attempt >= policy.max_retries:
                raise
            retry_after = exc.retry_after_seconds
            if retry_after is None:
                retry_after = policy.retry_base_seconds * (2**attempt) * random_value()
            await sleep(min(retry_after, policy.maximum_retry_after_seconds))
    raise AssertionError("Embedding retry loop did not return or raise")


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = (retry_at - datetime.now(UTC)).total_seconds()
    return max(0.0, seconds)
