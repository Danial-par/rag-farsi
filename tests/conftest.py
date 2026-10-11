from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://rag_assistant:test@127.0.0.1:5432/rag_assistant_test",
)
os.environ.setdefault("APP_ENV", "test")

from rag_assistant.core.config import Settings  # noqa: E402
from rag_assistant.main import create_app  # noqa: E402


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
