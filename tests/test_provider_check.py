from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from rag_assistant.assistant.generation import create_chat_model
from rag_assistant.core.config import Settings
from rag_assistant.provider_check import (
    ProviderCheckError,
    ProviderCheckReply,
    run_provider_check,
)


def _settings(**overrides: Any) -> Settings:
    return Settings(
        app_env="test",
        database_url="postgresql+psycopg://user:password@localhost/database",
        jwt_secret="test-only-jwt-secret-with-at-least-32-bytes",
        openrouter_api_key=SecretStr("test-provider-key"),
        **overrides,
    )


def test_chat_reasoning_effort_defaults_to_none_and_reaches_chat_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("rag_assistant.assistant.generation.ChatOpenAI", FakeChatOpenAI)
    settings = _settings()

    create_chat_model(settings)

    assert settings.chat_reasoning_effort == "none"
    assert captured["reasoning_effort"] == "none"
    assert captured["model"] == "openai/gpt-6-luna"


async def test_provider_check_uses_one_embedding_and_structured_chat_request() -> None:
    class FakeEmbeddings:
        def __init__(self) -> None:
            self.calls: list[tuple[Sequence[str], str, int]] = []

        async def embed(
            self, texts: Sequence[str], *, model: str, dimensions: int
        ) -> list[list[float]]:
            self.calls.append((texts, model, dimensions))
            return [[0.0] * dimensions]

    class FakeStructured:
        def __init__(self) -> None:
            self.messages: list[HumanMessage] = []

        async def ainvoke(self, messages: list[HumanMessage]) -> dict[str, Any]:
            self.messages = messages
            return {"parsed": ProviderCheckReply(status="ok"), "raw": object()}

    class FakeChat:
        def __init__(self) -> None:
            self.structured = FakeStructured()

        def with_structured_output(self, schema: type[Any], *, include_raw: bool) -> FakeStructured:
            assert schema is ProviderCheckReply
            assert include_raw is True
            return self.structured

    embeddings = FakeEmbeddings()
    chat = FakeChat()
    result = await run_provider_check(
        _settings(),
        embedding_client=embeddings,
        chat_model_factory=lambda _: chat,
    )

    assert len(embeddings.calls) == 1
    assert embeddings.calls[0][1:] == ("openai/text-embedding-3-large", 1024)
    assert len(chat.structured.messages) == 1
    assert result.embedding_dimensions == 1024
    assert result.chat_model == "openai/gpt-6-luna"


async def test_provider_failure_exposes_only_a_sanitized_category() -> None:
    class FailingEmbeddings:
        async def embed(
            self, texts: Sequence[str], *, model: str, dimensions: int
        ) -> list[list[float]]:
            del texts, model, dimensions
            raise RuntimeError("https://private.invalid/key?token=secret")

    with pytest.raises(ProviderCheckError) as raised:
        await run_provider_check(_settings(), embedding_client=FailingEmbeddings())

    assert raised.value.category == "embedding_request_failed"
    assert "private.invalid" not in str(raised.value)
    assert "secret" not in str(raised.value)


async def test_provider_check_rejects_wrong_embedding_dimension() -> None:
    class WrongDimensions:
        async def embed(
            self, texts: Sequence[str], *, model: str, dimensions: int
        ) -> list[list[float]]:
            del texts, model, dimensions
            return [[1.0, 2.0]]

    with pytest.raises(ProviderCheckError) as raised:
        await run_provider_check(_settings(), embedding_client=WrongDimensions())

    assert raised.value.category == "embedding_invalid_response"
