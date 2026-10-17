from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from uuid import UUID

import uvicorn

from rag_assistant.core.config import Settings, get_settings
from rag_assistant.core.event_loop import compatible_event_loop_factory
from rag_assistant.core.logging import configure_logging
from rag_assistant.db.session import create_database_resources
from rag_assistant.indexing import (
    DocumentIndexingService,
    OpenRouterEmbeddingClient,
    create_object_store,
)
from rag_assistant.indexing.worker import DocumentIngestionWorker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Persian RAG Assistant API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> None:
    """Run Uvicorn with an event loop compatible with async psycopg on Windows."""
    args = _parser().parse_args(argv)
    uvicorn.run(
        "rag_assistant.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        loop="rag_assistant.core.event_loop:compatible_event_loop_factory",
        access_log=False,
    )


def _ingestion_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index one pending document version")
    parser.add_argument("version_id", type=UUID, help="Document-version UUID to claim and index")
    return parser


async def _ingest_version(version_id: UUID, settings: Settings) -> None:
    if settings.openrouter_api_key is None:
        raise ValueError("OPENROUTER_API_KEY is required by rag-ingest")
    engine, session_factory = create_database_resources(settings)
    embedding_client = OpenRouterEmbeddingClient(
        base_url=str(settings.openrouter_base_url),
        api_key=settings.openrouter_api_key.get_secret_value(),
        connect_timeout_seconds=settings.embedding_connect_timeout_seconds,
        read_timeout_seconds=settings.embedding_read_timeout_seconds,
    )
    try:
        service = DocumentIndexingService(
            session_factory=session_factory,
            object_store=create_object_store(settings),
            embedding_client=embedding_client,
            settings=settings,
        )
        await service.index_version(version_id)
    finally:
        await embedding_client.aclose()
        await engine.dispose()


def ingest(argv: Sequence[str] | None = None) -> None:
    """Run the one-shot, lease-protected document indexing command."""
    args = _ingestion_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(
        _ingest_version(args.version_id, settings),
        loop_factory=compatible_event_loop_factory,
    )


async def _run_worker(settings: Settings) -> None:
    if settings.openrouter_api_key is None:
        raise ValueError("OPENROUTER_API_KEY is required by rag-worker")
    engine, session_factory = create_database_resources(settings)
    embedding_client = OpenRouterEmbeddingClient(
        base_url=str(settings.openrouter_base_url),
        api_key=settings.openrouter_api_key.get_secret_value(),
        connect_timeout_seconds=settings.embedding_connect_timeout_seconds,
        read_timeout_seconds=settings.embedding_read_timeout_seconds,
    )
    try:
        object_store = create_object_store(settings)
        indexing_service = DocumentIndexingService(
            session_factory=session_factory,
            object_store=object_store,
            embedding_client=embedding_client,
            settings=settings,
        )
        await DocumentIngestionWorker(
            session_factory=session_factory,
            object_store=object_store,
            indexing_service=indexing_service,
            settings=settings,
        ).run_forever()
    finally:
        await embedding_client.aclose()
        await engine.dispose()


def worker() -> None:
    """Run the restart-safe database polling ingestion worker."""
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(_run_worker(settings), loop_factory=compatible_event_loop_factory)
