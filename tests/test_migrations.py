from __future__ import annotations

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


def test_migration_history_has_one_head() -> None:
    scripts = ScriptDirectory.from_config(Config("alembic.ini"))

    assert scripts.get_heads() == ["20261011_0002"]


def test_migration_matches_model_metadata(migrated_database_url: str) -> None:
    del migrated_database_url
    command.check(Config("alembic.ini"))


async def test_migrations_upgrade_empty_postgres_to_head(migrated_database_url: str) -> None:
    engine = create_async_engine(migrated_database_url)
    try:
        async with engine.connect() as connection:
            revision = await connection.scalar(text("SELECT version_num FROM alembic_version"))
            vector_enabled = await connection.scalar(
                text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            )
            tables = set(
                await connection.scalars(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public'"
                    )
                )
            )
    finally:
        await engine.dispose()

    assert revision == "20261011_0002"
    assert vector_enabled is True
    assert {
        "users",
        "auth_sessions",
        "documents",
        "document_versions",
        "source_paragraphs",
        "chunks",
        "chunk_source_paragraphs",
        "conversations",
        "conversation_documents",
        "messages",
    } <= tables
