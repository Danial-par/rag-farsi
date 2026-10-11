from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from rag_assistant import __version__
from rag_assistant.api.health import router as health_router
from rag_assistant.api.router import api_router
from rag_assistant.core.config import Settings, get_settings
from rag_assistant.core.logging import RequestContextMiddleware, configure_logging
from rag_assistant.db.session import create_database_resources


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an independently configurable FastAPI application."""
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    logger = structlog.stdlib.get_logger(__name__)
    engine, session_factory = create_database_resources(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await logger.ainfo(
            "application_started",
            environment=resolved_settings.app_env,
            version=__version__,
        )
        try:
            yield
        finally:
            await engine.dispose()
            await logger.ainfo("application_stopped")

    application = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    application.state.settings = resolved_settings
    application.state.database_engine = engine
    application.state.session_factory = session_factory
    application.add_middleware(RequestContextMiddleware)
    application.include_router(health_router)
    application.include_router(api_router)
    return application


app = create_app()
