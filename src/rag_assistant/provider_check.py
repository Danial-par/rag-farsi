"""Safe live compatibility check for the configured OpenRouter clients."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict

from rag_assistant.assistant.generation import create_chat_model
from rag_assistant.core.config import Settings
from rag_assistant.indexing.embeddings import (
    EmbeddingClient,
    EmbeddingError,
    OpenRouterEmbeddingClient,
)


class ProviderCheckReply(BaseModel):
    """Tiny response used to validate structured chat output compatibility."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["ok"]


@dataclass(frozen=True, slots=True)
class ProviderCheckResult:
    chat_model: str
    embedding_model: str
    embedding_dimensions: int


class ProviderCheckError(RuntimeError):
    """A failure with a safe category suitable for command-line display."""

    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


async def run_provider_check(
    settings: Settings,
    *,
    embedding_client: EmbeddingClient | None = None,
    chat_model_factory: Callable[[Settings], Any] = create_chat_model,
) -> ProviderCheckResult:
    """Make one embedding and one structured chat request using application clients."""
    if settings.openrouter_api_key is None or not settings.openrouter_api_key.get_secret_value():
        raise ProviderCheckError("configuration")

    client = embedding_client or OpenRouterEmbeddingClient(
        base_url=str(settings.openrouter_base_url),
        api_key=settings.openrouter_api_key.get_secret_value(),
        connect_timeout_seconds=settings.embedding_connect_timeout_seconds,
        read_timeout_seconds=settings.embedding_read_timeout_seconds,
    )
    owns_client = embedding_client is None
    try:
        try:
            vectors = await client.embed(
                ["provider compatibility check"],
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
            )
        except Exception as exc:
            raise ProviderCheckError(_embedding_failure_category(exc)) from None
        if (
            len(vectors) != 1
            or len(vectors[0]) != settings.embedding_dimensions
            or any(
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in vectors[0]
            )
        ):
            raise ProviderCheckError("embedding_invalid_response")

        try:
            structured = chat_model_factory(settings).with_structured_output(
                ProviderCheckReply, include_raw=True
            )
            response = await structured.ainvoke(
                [HumanMessage(content='Return the structured value {"status":"ok"}.')]
            )
        except Exception as exc:
            raise ProviderCheckError(_chat_failure_category(exc)) from None
        parsed = response.get("parsed") if isinstance(response, dict) else None
        if not isinstance(parsed, ProviderCheckReply) or parsed.status != "ok":
            raise ProviderCheckError("chat_invalid_response")
        return ProviderCheckResult(
            chat_model=settings.chat_model,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
        )
    finally:
        if owns_client:
            await client.aclose()  # type: ignore[attr-defined]


def _embedding_failure_category(error: Exception) -> str:
    if isinstance(error, EmbeddingError):
        if "timed out" in str(error):
            return "embedding_timeout"
        if "dimension" in str(error).lower() or "shape" in str(error).lower():
            return "embedding_invalid_response"
        return "embedding_provider"
    if isinstance(error, httpx.TimeoutException):
        return "embedding_timeout"
    if isinstance(error, (ValueError, TypeError)):
        return "embedding_invalid_response"
    return "embedding_request_failed"


def _chat_failure_category(error: Exception) -> str:
    if isinstance(error, (httpx.TimeoutException, TimeoutError)):
        return "chat_timeout"
    if isinstance(error, (ValueError, TypeError)):
        return "chat_invalid_response"
    if type(error).__name__ in {"APIConnectionError", "ConnectError", "ConnectTimeout"}:
        return "chat_connection"
    if type(error).__name__ in {"RateLimitError", "APIStatusError"}:
        return "chat_provider"
    return "chat_request_failed"
