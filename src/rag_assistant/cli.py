from __future__ import annotations

import argparse
import asyncio
import json
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
from rag_assistant.provider_check import ProviderCheckError, run_provider_check
from rag_assistant.sample_seed import seed_sample_document


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
        proxy_headers=bool(get_settings().trusted_proxy_hosts),
        forwarded_allow_ips=",".join(get_settings().trusted_proxy_hosts) or "",
    )


def _ingestion_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index one pending document version")
    parser.add_argument("version_id", type=UUID, help="Document-version UUID to claim and index")
    return parser


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the document indexing worker")
    parser.add_argument(
        "--once", action="store_true", help="Process at most one pending version and exit"
    )
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
        from langfuse import get_client

        get_client().flush()
        get_client().shutdown()
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


async def _run_worker(settings: Settings, *, once: bool = False) -> bool:
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
        worker_service = DocumentIngestionWorker(
            session_factory=session_factory,
            object_store=object_store,
            indexing_service=indexing_service,
            settings=settings,
        )
        if once:
            return await worker_service.run_once()
        await worker_service.run_forever()
        return False
    finally:
        await embedding_client.aclose()
        from langfuse import get_client

        get_client().flush()
        get_client().shutdown()
        await engine.dispose()


def worker(argv: Sequence[str] | None = None) -> None:
    """Run the database polling worker continuously or for one claimed item."""
    args = _worker_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    asyncio.run(_run_worker(settings, once=args.once), loop_factory=compatible_event_loop_factory)


async def _check_provider(settings: Settings) -> int:
    try:
        result = await run_provider_check(settings)
    except ProviderCheckError as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "chat_model": settings.chat_model,
                    "embedding_model": settings.embedding_model,
                    "embedding_dimensions": settings.embedding_dimensions,
                    "failure_category": exc.category,
                }
            )
        )
        return 1
    except Exception:
        print(
            json.dumps(
                {
                    "success": False,
                    "chat_model": settings.chat_model,
                    "embedding_model": settings.embedding_model,
                    "embedding_dimensions": settings.embedding_dimensions,
                    "failure_category": "configuration",
                }
            )
        )
        return 1
    print(
        json.dumps(
            {
                "success": True,
                "chat_model": result.chat_model,
                "embedding_model": result.embedding_model,
                "embedding_dimensions": result.embedding_dimensions,
            }
        )
    )
    return 0


def provider_check() -> None:
    """Verify the configured chat and embedding endpoints without exposing responses."""
    try:
        settings = get_settings()
    except Exception:
        print(json.dumps({"success": False, "failure_category": "configuration"}))
        raise SystemExit(1) from None
    raise SystemExit(
        asyncio.run(_check_provider(settings), loop_factory=compatible_event_loop_factory)
    )


async def _seed_sample(settings: Settings) -> None:
    if settings.app_env != "local":
        raise ValueError("Sample seeding is available only when APP_ENV=local")
    engine, session_factory = create_database_resources(settings)
    try:
        result = await seed_sample_document(
            settings=settings,
            session_factory=session_factory,
            object_store=create_object_store(settings),
        )
        print(
            json.dumps(
                {
                    "document_id": str(result.document_id),
                    "version_id": str(result.version_id),
                    "status": result.status.value,
                    "already_active": result.already_active,
                }
            )
        )
    finally:
        await engine.dispose()


def seed_sample() -> None:
    """Create or reuse the local shared-default sample PDF version."""
    try:
        settings = get_settings()
        if settings.app_env != "local":
            print(json.dumps({"success": False, "failure_category": "production_refusal"}))
            raise SystemExit(2)
        asyncio.run(_seed_sample(settings), loop_factory=compatible_event_loop_factory)
    except SystemExit:
        raise
    except Exception:
        print(json.dumps({"success": False, "failure_category": "sample_seed_failed"}))
        raise SystemExit(1) from None
