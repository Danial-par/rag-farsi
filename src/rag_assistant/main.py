from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

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
from rag_assistant.core.observability import configure_langfuse, shutdown_langfuse
from rag_assistant.core.security import PasswordManager, TokenManager
from rag_assistant.db.base import utc_now
from rag_assistant.db.repositories import ConversationRepository
from rag_assistant.db.session import AsyncSessionFactory, create_database_resources
from rag_assistant.indexing import create_object_store
from rag_assistant.indexing.embeddings import EmbeddingClient, OpenRouterEmbeddingClient
from rag_assistant.retrieval import RetrievalRequest, RetrievalResult, RetrievalService
from rag_assistant.retrieval.planner import (
    RetrievalPlanner,
    retrieval_planner_from_settings,
)
from rag_assistant.web.router import router as web_router


class _SessionScopedRetriever:
    """Open a short database session for each authorization-scoped retrieval call."""

    def __init__(
        self,
        session_factory: AsyncSessionFactory,
        embedding_client: EmbeddingClient,
        settings: Settings,
        planner: RetrievalPlanner | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._embedding_client = embedding_client
        self._settings = settings
        self._planner = planner

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        async with self._session_factory() as session:
            service = RetrievalService(
                session=session,
                embedding_client=self._embedding_client,
                settings=self._settings,
                planner=self._planner,
            )
            return await service.retrieve(request)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an independently configurable FastAPI application."""
    resolved_settings = settings or get_settings()
    configure_langfuse(resolved_settings)
    configure_logging(resolved_settings.log_level)
    logger = structlog.stdlib.get_logger(__name__)
    engine, session_factory = create_database_resources(resolved_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        embedding_client = None
        recovery_task = None
        try:
            if resolved_settings.openrouter_api_key is not None:
                embedding_client = OpenRouterEmbeddingClient(
                    base_url=str(resolved_settings.openrouter_base_url),
                    api_key=resolved_settings.openrouter_api_key.get_secret_value(),
                    connect_timeout_seconds=resolved_settings.embedding_connect_timeout_seconds,
                    read_timeout_seconds=resolved_settings.embedding_read_timeout_seconds,
                )
                retrieval_planner = retrieval_planner_from_settings(resolved_settings)
                application.state.rag_workflow = create_rag_workflow(
                    RAGDependencies(
                        retriever=_SessionScopedRetriever(
                            session_factory,
                            embedding_client,
                            resolved_settings,
                            planner=retrieval_planner,
                        ),
                        generator=OpenRouterAnswerGenerator.from_settings(resolved_settings),
                        validator=EvidenceAnswerValidator(),
                        persistence=SqlAlchemyWorkflowPersistence(session_factory),
                        observer=LangfuseNodeObserver(),
                        execution_heartbeat_seconds=resolved_settings.assistant_execution_recovery_poll_seconds,
                    )
                )
            else:
                application.state.rag_workflow = None
            if resolved_settings.app_env != "test":
                recovery_task = asyncio.create_task(
                    _recover_abandoned_executions(
                        application.state.session_factory, resolved_settings
                    )
                )
            await logger.ainfo(
                "application_started",
                environment=resolved_settings.app_env,
                version=__version__,
            )
            yield
        finally:
            if recovery_task is not None:
                recovery_task.cancel()
                with suppress(asyncio.CancelledError):
                    await recovery_task
            if embedding_client is not None:
                await embedding_client.aclose()
            shutdown_langfuse(resolved_settings)
            await engine.dispose()
            await logger.ainfo("application_stopped")

    application = FastAPI(
        title=resolved_settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    if resolved_settings.allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=resolved_settings.allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "X-Request-ID", "Idempotency-Key"],
            expose_headers=["X-Request-ID"],
        )
    application.state.settings = resolved_settings
    application.state.object_store = create_object_store(resolved_settings)
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
    application.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).parent / "web" / "static")),
        name="static",
    )
    application.include_router(web_router)
    return application


async def _recover_abandoned_executions(
    session_factory: AsyncSessionFactory, settings: Settings
) -> None:
    """Periodically terminalize chat executions whose process stopped renewing them."""
    interval = settings.assistant_execution_recovery_poll_seconds
    timeout = settings.assistant_execution_timeout_seconds
    logger = structlog.stdlib.get_logger("rag_assistant.assistant.recovery")
    while True:
        try:
            async with session_factory.begin() as session:
                recovered = await ConversationRepository(session).fail_abandoned_assistant_messages(
                    before=utc_now() - timedelta(seconds=timeout)
                )
            if recovered:
                await logger.awarning("abandoned_assistant_executions_failed", count=recovered)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await logger.aerror(
                "assistant_execution_recovery_failed", error_type=type(exc).__name__
            )
        await asyncio.sleep(interval)


app = create_app()
