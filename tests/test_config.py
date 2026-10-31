from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from rag_assistant.core.config import Settings
from rag_assistant.main import create_app

JWT_SECRET = "test-only-jwt-secret-with-at-least-32-bytes"


def test_database_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValidationError):
        Settings(jwt_secret=JWT_SECRET, _env_file=None)


def test_database_url_requires_psycopg() -> None:
    with pytest.raises(ValidationError, match="postgresql\\+psycopg"):
        Settings(
            database_url="postgresql+asyncpg://user:password@localhost/database",
            jwt_secret=JWT_SECRET,
        )


def test_log_level_is_case_insensitive() -> None:
    settings = Settings(
        database_url="postgresql+psycopg://user:password@localhost/database",
        jwt_secret=JWT_SECRET,
        log_level="debug",
    )

    assert settings.log_level == "DEBUG"


def test_jwt_secret_requires_at_least_32_bytes() -> None:
    with pytest.raises(ValidationError, match="at least 32 bytes"):
        Settings(
            database_url="postgresql+psycopg://user:password@localhost/database",
            jwt_secret="too-short",
        )


def test_secure_cookies_are_environment_aware() -> None:
    local = Settings(
        app_env="local",
        database_url="postgresql+psycopg://user:password@localhost/database",
        jwt_secret=JWT_SECRET,
    )
    production = Settings(
        app_env="production",
        database_url="postgresql+psycopg://user:password@localhost/database",
        jwt_secret=JWT_SECRET,
        openrouter_api_key="test-openrouter-key",
        object_store_backend="s3",
        s3_bucket="private-test-bucket",
        allowed_origins=["https://app.example.test"],
        reranker_base_url="https://reranker.example.test",
    )

    assert local.secure_cookies is False
    assert production.secure_cookies is True


def test_production_requires_private_s3_storage() -> None:
    with pytest.raises(ValidationError, match="OBJECT_STORE_BACKEND=s3"):
        Settings(
            app_env="production",
            database_url="postgresql+psycopg://user:password@localhost/database",
            jwt_secret=JWT_SECRET,
        )


def test_chunk_target_cannot_exceed_hard_maximum() -> None:
    with pytest.raises(ValidationError, match="CHUNK_TARGET_TOKENS"):
        Settings(
            database_url="postgresql+psycopg://user:password@localhost/database",
            jwt_secret=JWT_SECRET,
            chunk_target_tokens=601,
            chunk_max_tokens=600,
        )


def test_langfuse_credentials_must_be_configured_together() -> None:
    with pytest.raises(ValidationError, match="LANGFUSE_PUBLIC_KEY"):
        Settings(
            database_url="postgresql+psycopg://user:password@localhost/database",
            jwt_secret=JWT_SECRET,
            langfuse_public_key="pk-test",
        )


def test_production_requires_model_credentials_and_allowed_origins() -> None:
    with pytest.raises(ValidationError, match="OPENROUTER_API_KEY"):
        Settings(
            app_env="production",
            database_url="postgresql+psycopg://user:password@localhost/database",
            jwt_secret=JWT_SECRET,
            object_store_backend="s3",
            s3_bucket="private-test-bucket",
            allowed_origins=["https://app.example.test"],
        )
    with pytest.raises(ValidationError, match="ALLOWED_ORIGINS"):
        Settings(
            app_env="production",
            database_url="postgresql+psycopg://user:password@localhost/database",
            jwt_secret=JWT_SECRET,
            object_store_backend="s3",
            s3_bucket="private-test-bucket",
            openrouter_api_key="test-openrouter-key",
        )
    with pytest.raises(ValidationError, match="Production ALLOWED_ORIGINS must use HTTPS"):
        Settings(
            app_env="production",
            database_url="postgresql+psycopg://user:password@localhost/database",
            jwt_secret=JWT_SECRET,
            object_store_backend="s3",
            s3_bucket="private-test-bucket",
            openrouter_api_key="test-openrouter-key",
            allowed_origins=["http://app.example.test"],
            reranker_base_url="https://reranker.example.test",
        )


def test_production_requires_a_non_loopback_reranker_endpoint() -> None:
    common: dict[str, Any] = {
        "app_env": "production",
        "database_url": "postgresql+psycopg://user:password@localhost/database",
        "jwt_secret": JWT_SECRET,
        "object_store_backend": "s3",
        "s3_bucket": "private-test-bucket",
        "openrouter_api_key": "test-openrouter-key",
        "allowed_origins": ["https://app.example.test"],
    }

    with pytest.raises(ValidationError, match="RERANKER_BASE_URL"):
        Settings(**common)

    settings = Settings(**common, reranker_base_url="https://reranker.example.test")
    assert str(settings.reranker_base_url).startswith("https://reranker.example.test")


def test_s3_static_credentials_must_be_configured_as_a_pair() -> None:
    with pytest.raises(ValidationError, match="S3_ACCESS_KEY_ID"):
        Settings(
            database_url="postgresql+psycopg://user:password@localhost/database",
            jwt_secret=JWT_SECRET,
            s3_access_key_id="access-key",
        )


def test_origin_and_proxy_allowlists_are_validated() -> None:
    with pytest.raises(ValidationError, match=r"exact HTTP\(S\) origins"):
        Settings(
            database_url="postgresql+psycopg://user:password@localhost/database",
            jwt_secret=JWT_SECRET,
            allowed_origins=["*"],
        )
    with pytest.raises(ValidationError, match="IP addresses or CIDRs"):
        Settings(
            database_url="postgresql+psycopg://user:password@localhost/database",
            jwt_secret=JWT_SECRET,
            trusted_proxy_hosts=["proxy.internal"],
        )


async def test_cors_uses_only_the_configured_browser_origins() -> None:
    settings = Settings(
        app_env="test",
        database_url="postgresql+psycopg://user:password@localhost/database",
        jwt_secret=JWT_SECRET,
        allowed_origins=["https://app.example.test"],
    )
    application = create_app(settings)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=application), base_url="http://test"
        ) as client:
            allowed = await client.options(
                "/api/v1/auth/register",
                headers={
                    "Origin": "https://app.example.test",
                    "Access-Control-Request-Method": "POST",
                },
            )
            denied = await client.options(
                "/api/v1/auth/register",
                headers={
                    "Origin": "https://unexpected.example.test",
                    "Access-Control-Request-Method": "POST",
                },
            )
    finally:
        await application.state.database_engine.dispose()

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://app.example.test"
    assert allowed.headers["access-control-allow-credentials"] == "true"
    assert "access-control-allow-origin" not in denied.headers
