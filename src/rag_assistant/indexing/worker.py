from __future__ import annotations

import asyncio
from datetime import timedelta
from time import perf_counter

import structlog
from langfuse import observe

from rag_assistant.core.config import Settings
from rag_assistant.db.base import utc_now
from rag_assistant.db.repositories.documents import DocumentRepository, IngestionClaim
from rag_assistant.db.session import AsyncSessionFactory
from rag_assistant.indexing.service import DocumentIndexingService
from rag_assistant.indexing.storage import ObjectStoreError, ObjectWriter


class DocumentIngestionWorker:
    """Poll and safely process durable indexing work from PostgreSQL."""

    def __init__(
        self,
        *,
        session_factory: AsyncSessionFactory,
        object_store: ObjectWriter,
        indexing_service: DocumentIndexingService,
        settings: Settings,
    ) -> None:
        self._session_factory = session_factory
        self._object_store = object_store
        self._indexing_service = indexing_service
        self._settings = settings
        self._logger = structlog.stdlib.get_logger("rag_assistant.worker")

    async def run_once(self) -> bool:
        """Claim and process one version, returning whether work was found."""
        async with self._session_factory() as session, session.begin():
            claim = await DocumentRepository(session).claim_next_for_indexing(
                lease_duration=timedelta(seconds=self._settings.ingestion_lease_seconds)
            )
        if claim is None:
            return False
        await self._process_claim(claim)
        return True

    @observe(name="documents.worker_attempt", capture_input=False, capture_output=False)
    async def _process_claim(self, claim: IngestionClaim) -> None:
        log = self._logger.bind(
            document_id=str(claim.document_id),
            document_version_id=str(claim.version_id),
            ingestion_attempt_id=str(claim.attempt_id),
        )
        started = perf_counter()
        await log.ainfo("worker_attempt_started")
        try:
            await self._indexing_service.index_claim(claim)
        except Exception as exc:
            await log.aerror(
                "worker_attempt_failed",
                exception_type=type(exc).__name__,
                duration_ms=round((perf_counter() - started) * 1000, 2),
            )
        else:
            await log.ainfo(
                "worker_attempt_completed",
                duration_ms=round((perf_counter() - started) * 1000, 2),
            )

    async def reconcile_orphan_objects(self) -> int:
        """Delete old uploaded source objects that have no persisted version row."""
        cutoff = utc_now() - timedelta(seconds=self._settings.orphan_object_grace_seconds)
        deleted = 0
        continuation_token = None
        while True:
            page = await self._object_store.list(
                "documents/", limit=500, continuation_token=continuation_token
            )
            candidates = [item for item in page.objects if item.last_modified < cutoff]
            async with self._session_factory() as session:
                keys = await DocumentRepository(session).persisted_object_keys(
                    [item.key for item in candidates]
                )
            for item in candidates:
                if item.key in keys:
                    continue
                try:
                    await self._object_store.delete(item.key)
                except ObjectStoreError:
                    await self._logger.awarning(
                        "orphan_object_cleanup_failed", object_key=item.key
                    )
                else:
                    deleted += 1
                    await self._logger.ainfo("orphan_object_removed", object_key=item.key)
            continuation_token = page.continuation_token
            if continuation_token is None:
                break
        return deleted

    async def run_forever(self) -> None:
        """Poll continuously and run bounded orphan reconciliation between claims."""
        logger = self._logger
        last_reconciliation = 0.0
        while True:
            try:
                started = perf_counter()
                if (
                    started - last_reconciliation
                    >= self._settings.orphan_reconcile_interval_seconds
                ):
                    try:
                        await self.reconcile_orphan_objects()
                    except ObjectStoreError as exc:
                        await logger.awarning(
                            "orphan_object_scan_failed", exception_type=type(exc).__name__
                        )
                    last_reconciliation = started
                if await self.run_once():
                    continue
                await asyncio.sleep(self._settings.ingestion_poll_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await logger.aerror("worker_poll_failed", exception_type=type(exc).__name__)
                await asyncio.sleep(self._settings.ingestion_poll_seconds)
