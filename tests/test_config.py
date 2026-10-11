from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_assistant.core.config import Settings


def test_database_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_database_url_requires_psycopg() -> None:
    with pytest.raises(ValidationError, match="postgresql\\+psycopg"):
        Settings(database_url="postgresql+asyncpg://user:password@localhost/database")


def test_log_level_is_case_insensitive() -> None:
    settings = Settings(
        database_url="postgresql+psycopg://user:password@localhost/database",
        log_level="debug",
    )

    assert settings.log_level == "DEBUG"
