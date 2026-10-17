from __future__ import annotations

from typing import Any

from rag_assistant.core.config import Settings
from rag_assistant.core.observability import configure_langfuse, shutdown_langfuse


def test_configure_langfuse_passes_pydantic_settings_explicitly(
    monkeypatch: Any,
) -> None:
    settings = Settings(
        database_url="postgresql+psycopg://user:password@localhost/database",
        jwt_secret="test-only-jwt-secret-with-at-least-32-bytes",
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_base_url="https://langfuse.example.test",
    )
    calls: list[dict[str, str]] = []

    def capture_client(**kwargs: str) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("rag_assistant.core.observability.Langfuse", capture_client)

    assert configure_langfuse(settings) is True
    assert calls == [
        {
            "public_key": "pk-test",
            "secret_key": "sk-test",
            "base_url": "https://langfuse.example.test",
            "additional_headers": {"x-langfuse-ingestion-version": "4"},
        }
    ]


def test_configure_langfuse_without_credentials_does_not_create_disabled_client(
    monkeypatch: Any,
) -> None:
    settings = Settings(
        database_url="postgresql+psycopg://user:password@localhost/database",
        jwt_secret="test-only-jwt-secret-with-at-least-32-bytes",
    )

    def fail_if_called(**kwargs: str) -> None:
        raise AssertionError(f"Unexpected Langfuse initialization: {kwargs}")

    monkeypatch.setattr("rag_assistant.core.observability.Langfuse", fail_if_called)

    assert configure_langfuse(settings) is False


def test_shutdown_langfuse_uses_explicit_project_key(monkeypatch: Any) -> None:
    settings = Settings(
        database_url="postgresql+psycopg://user:password@localhost/database",
        jwt_secret="test-only-jwt-secret-with-at-least-32-bytes",
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )
    calls: list[str] = []

    class FakeClient:
        def flush(self) -> None:
            calls.append("flush")

        def shutdown(self) -> None:
            calls.append("shutdown")

    def client_for_project(*, public_key: str) -> FakeClient:
        calls.append(public_key)
        return FakeClient()

    monkeypatch.setattr(
        "rag_assistant.core.observability.get_client",
        client_for_project,
    )

    shutdown_langfuse(settings)

    assert calls == ["pk-test", "flush", "shutdown"]
