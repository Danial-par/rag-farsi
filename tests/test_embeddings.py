from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx
import pytest

from rag_assistant.indexing.embeddings import (
    EmbeddingBatchPolicy,
    EmbeddingDimensionError,
    EmbeddingTimeoutError,
    OpenRouterEmbeddingClient,
    RetryableEmbeddingError,
    embed_in_batches,
)


class FakeEmbeddingClient:
    def __init__(self, *, failures: int = 0, dimensions: int = 3) -> None:
        self.failures = failures
        self.dimensions = dimensions
        self.calls: list[list[str]] = []
        self.active = 0
        self.maximum_active = 0

    async def embed(
        self, texts: Sequence[str], *, model: str, dimensions: int
    ) -> list[list[float]]:
        del model, dimensions
        self.calls.append(list(texts))
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        if self.failures:
            self.failures -= 1
            raise RetryableEmbeddingError("temporary", retry_after_seconds=0)
        return [[float(index)] * self.dimensions for index, _ in enumerate(texts)]


async def test_embedding_batches_preserve_order_and_bound_concurrency() -> None:
    client = FakeEmbeddingClient()

    vectors = await embed_in_batches(
        client,
        [f"text-{index}" for index in range(7)],
        model="openai/text-embedding-3-large",
        dimensions=3,
        policy=EmbeddingBatchPolicy(batch_size=2, concurrency=2),
    )

    assert len(vectors) == 7
    assert [len(call) for call in client.calls] == [2, 2, 2, 1]
    assert client.maximum_active == 2


async def test_retryable_embedding_failure_uses_bounded_retry_policy() -> None:
    client = FakeEmbeddingClient(failures=2)
    sleeps: list[float] = []

    vectors = await embed_in_batches(
        client,
        ["متن"],
        model="model",
        dimensions=3,
        policy=EmbeddingBatchPolicy(batch_size=1, concurrency=1, max_retries=2),
        sleep=lambda delay: _record_sleep(sleeps, delay),
    )

    assert len(vectors) == 1
    assert len(client.calls) == 3
    assert sleeps == [0, 0]


async def _record_sleep(values: list[float], delay: float) -> None:
    values.append(delay)


async def test_embedding_dimension_mismatch_is_rejected() -> None:
    client = FakeEmbeddingClient(dimensions=2)

    with pytest.raises(EmbeddingDimensionError):
        await embed_in_batches(
            client,
            ["متن"],
            model="model",
            dimensions=3,
            policy=EmbeddingBatchPolicy(batch_size=1, concurrency=1),
        )


async def test_openrouter_client_preserves_index_order_without_external_calls() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/embeddings"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    async with httpx.AsyncClient(
        base_url="https://openrouter.invalid/api/v1",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = OpenRouterEmbeddingClient(
            base_url="https://openrouter.invalid/api/v1",
            api_key="test-only",
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
            client=http_client,
        )
        vectors = await client.embed(["یک", "دو"], model="model", dimensions=2)

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


async def test_openrouter_rate_limit_is_retryable() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "2"})

    async with httpx.AsyncClient(
        base_url="https://openrouter.invalid/api/v1",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = OpenRouterEmbeddingClient(
            base_url="https://openrouter.invalid/api/v1",
            api_key="test-only",
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
            client=http_client,
        )
        with pytest.raises(RetryableEmbeddingError) as captured:
            await client.embed(["متن"], model="model", dimensions=2)

    assert captured.value.retry_after_seconds == 2


async def test_openrouter_timeout_is_classified_for_bounded_retry() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated", request=request)

    async with httpx.AsyncClient(
        base_url="https://openrouter.invalid/api/v1",
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = OpenRouterEmbeddingClient(
            base_url="https://openrouter.invalid/api/v1",
            api_key="test-only",
            connect_timeout_seconds=1,
            read_timeout_seconds=1,
            client=http_client,
        )
        with pytest.raises(EmbeddingTimeoutError):
            await client.embed(["متن"], model="model", dimensions=2)
