from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag_assistant.core.config import Settings

AsyncSessionFactory = async_sessionmaker[AsyncSession]


def create_database_resources(settings: Settings) -> tuple[AsyncEngine, AsyncSessionFactory]:
    """Create the process-wide engine and request session factory."""
    engine = create_async_engine(
        settings.sqlalchemy_database_url,
        pool_pre_ping=True,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
    )
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def get_session_factory(request: Request) -> AsyncSessionFactory:
    """Return the application-owned session factory for dependency injection."""
    return cast(AsyncSessionFactory, request.app.state.session_factory)


async def get_db_session(
    session_factory: Annotated[AsyncSessionFactory, Depends(get_session_factory)],
) -> AsyncIterator[AsyncSession]:
    """Yield a request-scoped database session and always close it afterward."""
    async with session_factory() as session:
        yield session
