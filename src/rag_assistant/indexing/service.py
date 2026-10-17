from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from time import perf_counter
from uuid import UUID

import structlog
from sqlalchemy.exc import SQLAlchemyError

from rag_assistant.core.config import Settings
from rag_assistant.db.models import SourceParagraph
from rag_assistant.db.repositories import (
    ChunkParagraphReference,
    DocumentRepository,
    IngestionClaim,
    InvalidPersistenceStateError,
    NewChunk,
)
from rag_assistant.db.session import AsyncSessionFactory
from rag_assistant.extraction import PdfExtractionError, PdfExtractor
from rag_assistant.extraction.domain import ExtractedDocument
from rag_assistant.extraction.service import extracted_paragraphs
from rag_assistant.indexing.chunking import (
    ChunkingError,
    ChunkingParagraph,
    ParagraphChunker,
    TiktokenTokenCounter,
    TokenCounter,
)
from rag_assistant.indexing.embeddings import (
    EmbeddingBatchPolicy,
    EmbeddingClient,
    EmbeddingDimensionError,
    PermanentEmbeddingError,
    RetryableEmbeddingError,
    embed_in_batches,
)
from rag_assistant.indexing.storage import ObjectStore, ObjectStoreError


class SourceIntegrityError(RuntimeError):
    """The immutable object does not match its persisted size or digest."""


@dataclass(frozen=True, slots=True)
class IndexingResult:
    version_id: UUID
    document_id: UUID
    paragraph_count: int
    chunk_count: int


class DocumentIndexingService:
    """Run one safely claimed document version through the complete indexing pipeline."""

    def __init__(
        self,
        *,
        session_factory: AsyncSessionFactory,
        object_store: ObjectStore,
        embedding_client: EmbeddingClient,
        settings: Settings,
        extractor: PdfExtractor | None = None,
        token_counter_factory: Callable[[str], TokenCounter] = TiktokenTokenCounter,
    ) -> None:
        self._session_factory = session_factory
        self._object_store = object_store
        self._embedding_client = embedding_client
        self._settings = settings
        self._extractor = extractor or PdfExtractor()
        self._token_counter_factory = token_counter_factory
        self._lease_duration = timedelta(seconds=settings.ingestion_lease_seconds)
        self._logger = structlog.stdlib.get_logger("rag_assistant.indexing")

    async def index_version(self, version_id: UUID) -> IndexingResult:
        started = perf_counter()
        claim = await self._claim(version_id)
        log = self._logger.bind(
            document_id=str(claim.document_id),
            document_version_id=str(claim.version_id),
            ingestion_attempt_id=str(claim.attempt_id),
        )
        await log.ainfo("indexing_started")
        try:
            source_started = perf_counter()
            payload = await self._object_store.get(claim.object_key)
            self._verify_source(payload, claim)
            await self._renew(claim)
            await log.ainfo(
                "indexing_stage_completed",
                stage="source_load",
                duration_ms=self._elapsed_ms(source_started),
                byte_count=len(payload),
            )

            extraction_started = perf_counter()
            extracted = await asyncio.to_thread(self._extractor.extract, payload)
            await self._renew(claim)
            paragraphs = await self._persist_extraction(claim, extracted)
            await log.ainfo(
                "indexing_stage_completed",
                stage="extraction",
                duration_ms=self._elapsed_ms(extraction_started),
                paragraph_count=len(paragraphs),
                page_count=extracted.metadata.page_count,
            )

            chunking_started = perf_counter()
            chunker = ParagraphChunker(
                self._token_counter_factory(claim.embedding_model),
                target_tokens=self._settings.chunk_target_tokens,
                max_tokens=self._settings.chunk_max_tokens,
                paragraph_overlap=self._settings.chunk_paragraph_overlap,
            )
            drafts = chunker.chunk([self._chunking_paragraph(row) for row in paragraphs])
            await log.ainfo(
                "indexing_stage_completed",
                stage="chunking",
                duration_ms=self._elapsed_ms(chunking_started),
                chunk_count=len(drafts),
            )

            embedding_started = perf_counter()
            vectors = await embed_in_batches(
                self._embedding_client,
                [draft.normalized_text for draft in drafts],
                model=claim.embedding_model,
                dimensions=claim.embedding_dimensions,
                policy=EmbeddingBatchPolicy(
                    batch_size=self._settings.embedding_batch_size,
                    concurrency=self._settings.embedding_concurrency,
                    max_retries=self._settings.embedding_max_retries,
                ),
                on_batch_completed=lambda: self._renew(claim),
            )
            await log.ainfo(
                "indexing_stage_completed",
                stage="embedding",
                duration_ms=self._elapsed_ms(embedding_started),
                vector_count=len(vectors),
            )

            publication_started = perf_counter()
            chunks = [
                NewChunk(
                    ordinal=draft.ordinal,
                    normalized_text=draft.normalized_text,
                    token_count=draft.token_count,
                    embedding=vector,
                    paragraphs=[
                        ChunkParagraphReference(
                            source_paragraph_id=reference.source_paragraph_id,
                            start_offset=reference.start_offset,
                            end_offset=reference.end_offset,
                        )
                        for reference in draft.references
                    ],
                    metadata=draft.metadata,
                )
                for draft, vector in zip(drafts, vectors, strict=True)
            ]
            await self._publish(claim, chunks, extracted.metadata.model_dump(mode="json"))
            await log.ainfo(
                "indexing_completed",
                stage="publication",
                stage_duration_ms=self._elapsed_ms(publication_started),
                duration_ms=self._elapsed_ms(started),
                paragraph_count=len(paragraphs),
                chunk_count=len(chunks),
            )
            return IndexingResult(
                version_id=claim.version_id,
                document_id=claim.document_id,
                paragraph_count=len(paragraphs),
                chunk_count=len(chunks),
            )
        except Exception as exc:
            failure_code, failure_message, metadata = self._classify_failure(exc)
            await self._mark_failed(claim, failure_code, failure_message, metadata)
            await log.aerror(
                "indexing_failed",
                failure_code=failure_code,
                exception_type=type(exc).__name__,
                duration_ms=self._elapsed_ms(started),
            )
            raise

    async def _claim(self, version_id: UUID) -> IngestionClaim:
        async with self._session_factory() as session, session.begin():
            return await DocumentRepository(session).claim_for_indexing(
                version_id,
                lease_duration=self._lease_duration,
            )

    async def _renew(self, claim: IngestionClaim) -> None:
        async with self._session_factory() as session, session.begin():
            await DocumentRepository(session).renew_ingestion_lease(
                claim.version_id,
                attempt_id=claim.attempt_id,
                lease_duration=self._lease_duration,
            )

    async def _persist_extraction(
        self, claim: IngestionClaim, extracted: ExtractedDocument
    ) -> list[SourceParagraph]:
        async with self._session_factory() as session, session.begin():
            repository = DocumentRepository(session)
            paragraphs = await repository.persist_extraction(
                version_id=claim.version_id,
                paragraphs=extracted_paragraphs(extracted),
                extraction_metadata=extracted.metadata.model_dump(mode="json"),
                attempt_id=claim.attempt_id,
            )
            await repository.renew_ingestion_lease(
                claim.version_id,
                attempt_id=claim.attempt_id,
                lease_duration=self._lease_duration,
            )
            return paragraphs

    async def _publish(
        self,
        claim: IngestionClaim,
        chunks: list[NewChunk],
        extraction_metadata: dict[str, object],
    ) -> None:
        async with self._session_factory() as session, session.begin():
            await DocumentRepository(session).publish_index(
                version_id=claim.version_id,
                attempt_id=claim.attempt_id,
                chunks=chunks,
                extraction_metadata=extraction_metadata,
            )

    async def _mark_failed(
        self,
        claim: IngestionClaim,
        failure_code: str,
        failure_message: str,
        extraction_metadata: dict[str, object] | None,
    ) -> None:
        try:
            async with self._session_factory() as session, session.begin():
                await DocumentRepository(session).mark_failed(
                    claim.version_id,
                    attempt_id=claim.attempt_id,
                    failure_code=failure_code,
                    failure_message=failure_message,
                    extraction_metadata=extraction_metadata,
                )
        except InvalidPersistenceStateError:
            # A newer attempt owns the version; this stale worker must not overwrite it.
            return
        except SQLAlchemyError:
            # The lease will expire and make the unfinished version reclaimable.
            await self._logger.aerror(
                "indexing_failure_state_not_persisted",
                document_version_id=str(claim.version_id),
                ingestion_attempt_id=str(claim.attempt_id),
            )

    @staticmethod
    def _verify_source(payload: bytes, claim: IngestionClaim) -> None:
        if claim.media_type != "application/pdf":
            raise SourceIntegrityError("Document version is not an application/pdf source")
        if len(payload) != claim.file_size_bytes:
            raise SourceIntegrityError("Source object size does not match its version record")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != claim.content_hash:
            raise SourceIntegrityError("Source object hash does not match its version record")

    @staticmethod
    def _chunking_paragraph(paragraph: SourceParagraph) -> ChunkingParagraph:
        return ChunkingParagraph(
            source_paragraph_id=paragraph.id,
            page_number=paragraph.page_number,
            paragraph_number=paragraph.paragraph_number,
            kind=paragraph.kind,
            normalized_text=paragraph.normalized_text,
        )

    @staticmethod
    def _classify_failure(
        exc: Exception,
    ) -> tuple[str, str, dict[str, object] | None]:
        if isinstance(exc, PdfExtractionError):
            return exc.code.value, str(exc), exc.metadata
        if isinstance(exc, SourceIntegrityError):
            return "source_integrity_failed", str(exc), None
        if isinstance(exc, ObjectStoreError):
            return "source_unavailable", "Source object could not be loaded", None
        if isinstance(exc, ChunkingError):
            return "chunking_failed", str(exc), None
        if isinstance(exc, EmbeddingDimensionError):
            return "embedding_dimensions_invalid", str(exc), None
        if isinstance(exc, RetryableEmbeddingError):
            return "embedding_retries_exhausted", "Embedding provider remained unavailable", None
        if isinstance(exc, PermanentEmbeddingError):
            return "embedding_request_failed", str(exc), None
        if isinstance(exc, SQLAlchemyError):
            return "index_persistence_failed", "Index publication failed", None
        return "indexing_internal", "Indexing failed unexpectedly", None

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((perf_counter() - started) * 1000, 2)
