from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag_assistant import __version__
from rag_assistant.api.health import router as health_router
from rag_assistant.api.router import api_router
from rag_assistant.assistant import (
    EvidenceAnswerValidator,
    LangfuseNodeObserver,
    OpenRouterAnswerGenerator,
    RAGDependencies,
    SqlAlchemyWorkflowPersistence,
    create_rag_workflow,
)
from rag_assistant.core.config import Settings, get_settings
from rag_assistant.core.logging import RequestContextMiddleware, configure_logging
from rag_assistant.core.security import PasswordManager, TokenManager
from rag_assistant.db.session import AsyncSessionFactory, create_database_resources
from rag_assistant.indexing.embeddings import EmbeddingClient, OpenRouterEmbeddingClient
from rag_assistant.retrieval import RetrievalRequest, RetrievalResult, RetrievalService


class _SessionScopedRetriever:
    """Open a short database session for each authorization-scoped retrieval call."""

    def __init__(
        self,
        session_factory: AsyncSessionFactory,
        embedding_client: EmbeddingClient,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._embedding_client = embedding_client
        self._settings = settings

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        async with self._session_factory() as session:
            service = RetrievalService(
                session=session,
                embedding_client=self._embedding_client,
                settings=self._settings,
            )
            return await service.retrieve(request)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an independently configurable FastAPI application."""
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    logger = structlog.stdlib.get_logger(__name__)
    engine, session_factory = create_database_resources(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        embedding_client = None
        if resolved_settings.openrouter_api_key is not None:
            embedding_client = OpenRouterEmbeddingClient(
                base_url=str(resolved_settings.openrouter_base_url),
                api_key=resolved_settings.openrouter_api_key.get_secret_value(),
                connect_timeout_seconds=resolved_settings.embedding_connect_timeout_seconds,
                read_timeout_seconds=resolved_settings.embedding_read_timeout_seconds,
            )
            application.state.rag_workflow = create_rag_workflow(
                RAGDependencies(
                    retriever=_SessionScopedRetriever(
                        session_factory, embedding_client, resolved_settings
                    ),
                    generator=OpenRouterAnswerGenerator.from_settings(resolved_settings),
                    validator=EvidenceAnswerValidator(),
                    persistence=SqlAlchemyWorkflowPersistence(session_factory),
                    observer=LangfuseNodeObserver(),
                )
            )
        else:
            application.state.rag_workflow = None
        await logger.ainfo(
            "application_started",
            environment=resolved_settings.app_env,
            version=__version__,
        )
        try:
            yield
        finally:
            if embedding_client is not None:
                await embedding_client.aclose()
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
    application.state.password_manager = PasswordManager()
    application.state.token_manager = TokenManager(resolved_settings)
    application.state.rag_workflow = None
    application.add_middleware(RequestContextMiddleware)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Request values can contain passwords or other secrets; validation errors do not need them.
        safe_errors = [
            {key: value for key, value in error.items() if key not in {"input", "ctx"}}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"detail": safe_errors},
            headers={"Cache-Control": "no-store"},
        )

    application.include_router(health_router)
    application.include_router(api_router)
    return application


app = create_app()
