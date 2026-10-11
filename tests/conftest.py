from __future__ import annotations

import os
from asyncio import AbstractEventLoop
from collections.abc import AsyncIterator, Callable, Iterator

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from testcontainers.community.postgres import PostgresContainer

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://rag_assistant:test@127.0.0.1:5432/rag_assistant_test",
)
os.environ.setdefault("APP_ENV", "test")

from rag_assistant.core.config import (
    Settings,  # noqa: E402
    get_settings,  # noqa: E402
)
from rag_assistant.core.event_loop import compatible_event_loop_factory  # noqa: E402
from rag_assistant.main import create_app  # noqa: E402


def pytest_asyncio_loop_factories(
    config: pytest.Config, item: pytest.Item
) -> dict[str, Callable[[], AbstractEventLoop]]:
    """Run async tests on the event loop supported by psycopg on Windows."""
    del config, item
    return {"compatible": compatible_event_loop_factory}


@pytest.fixture
def settings() -> Settings:
    return Settings(
        app_env="test",
        database_url="postgresql+psycopg://rag_assistant:test@127.0.0.1:5432/rag_assistant_test",
    )


@pytest.fixture
def application(settings: Settings) -> FastAPI:
    return create_app(settings)


@pytest_asyncio.fixture
async def started_application(application: FastAPI) -> AsyncIterator[FastAPI]:
    async with LifespanManager(application):
        yield application


@pytest_asyncio.fixture
async def client(started_application: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=started_application)
    async with AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client


@pytest.fixture(scope="session")
def disposable_postgres_url() -> Iterator[str]:
    """Provide a fresh PostgreSQL/pgvector database for this test session."""
    with PostgresContainer(
        image="pgvector/pgvector:0.8.7-pg17-bookworm",
        username="rag_assistant",
        password="test-only",
        dbname="rag_assistant_test",
        driver="psycopg",
    ) as postgres:
        yield postgres.get_connection_url()


@pytest.fixture(scope="session")
def migrated_database_url(disposable_postgres_url: str) -> Iterator[str]:
    previous_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = disposable_postgres_url
    get_settings.cache_clear()
    command.upgrade(Config("alembic.ini"), "head")
    try:
        yield disposable_postgres_url
    finally:
        if previous_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_url
        get_settings.cache_clear()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def database_engine(migrated_database_url: str) -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(migrated_database_url, pool_pre_ping=True)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(database_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Roll back each integration test while preserving the migrated schema."""
    async with database_engine.connect() as connection:
        transaction = await connection.begin()
        session = AsyncSession(bind=connection, expire_on_commit=False)
        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()
