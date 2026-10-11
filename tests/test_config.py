from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_assistant.core.config import Settings

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
    )

    assert local.secure_cookies is False
    assert production.secure_cookies is True
