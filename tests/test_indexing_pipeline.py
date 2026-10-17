from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, BinaryIO

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker

from rag_assistant.core.config import Settings
from rag_assistant.db.models import (
    Chunk,
    ChunkSourceParagraph,
    Document,
    DocumentVersion,
    IngestionStatus,
    SourceParagraph,
)
from rag_assistant.db.repositories import (
    DocumentRepository,
    NewDocumentVersion,
    UserRepository,
)
from rag_assistant.extraction import ExtractedDocument, PdfExtractor
from rag_assistant.indexing import DocumentIndexingService, RetryableEmbeddingError

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "documents" / "sample_doc.pdf"


class MemoryObjectStore:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    async def get(self, object_key: str) -> bytes:
        del object_key
        return self.payload


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


class DeterministicEmbeddingClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def embed(
        self, texts: Sequence[str], *, model: str, dimensions: int
    ) -> list[list[float]]:
        del model
        self.calls += 1
        if self.fail:
            raise RetryableEmbeddingError("temporary", retry_after_seconds=0)
        vectors: list[list[float]] = []
        for text_value in texts:
            vector = [0.0] * dimensions
            vector[int(hashlib.sha256(text_value.encode()).hexdigest()[:8], 16) % dimensions] = 1.0
            vectors.append(vector)
        return vectors


class FlakyPdfExtractor(PdfExtractor):
    def __init__(self) -> None:
        super().__init__()
        self.fail = True

    def extract(self, source: bytes | BinaryIO) -> ExtractedDocument:
        if self.fail:
            raise RuntimeError("simulated transient extraction failure")
        return super().extract(source)


async def _new_version(session: AsyncSession, payload: bytes) -> tuple[Document, DocumentVersion]:
    user = await UserRepository(session).create(
        email="indexing@example.com",
        password_hash="test-password-hash",
    )
    repository = DocumentRepository(session)
    document = await repository.create_owned(owner_id=user.id, title="سند نمونه")
    version = await repository.create_owned_version(
        owner_id=user.id,
        document_id=document.id,
        values=NewDocumentVersion(
            object_key="tests/fixtures/documents/sample_doc.pdf",
            content_hash=hashlib.sha256(payload).hexdigest(),
            file_size_bytes=len(payload),
            embedding_model="openai/text-embedding-3-large",
            embedding_dimensions=1024,
            ingestion_config={
                "chunk_target_tokens": 400,
                "chunk_max_tokens": 600,
                "chunk_paragraph_overlap": 1,
            },
        ),
    )
    await session.flush()
    return document, version


def _service(
    *,
    connection: AsyncConnection,
    settings: Settings,
    payload: bytes,
    embeddings: DeterministicEmbeddingClient,
    extractor: PdfExtractor | None = None,
) -> DocumentIndexingService:
    session_factory = async_sessionmaker(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    return DocumentIndexingService(
        session_factory=session_factory,
        object_store=MemoryObjectStore(payload),
        embedding_client=embeddings,
        settings=settings,
        extractor=extractor,
        token_counter_factory=lambda _: WordCounter(),
    )


async def test_sample_pdf_completes_pipeline_and_supports_exact_vector_search(
    db_session: AsyncSession,
    database_connection: AsyncConnection,
    settings: Settings,
) -> None:
    payload = SAMPLE_PDF.read_bytes()
    document, version = await _new_version(db_session, payload)
    document_id = document.id
    version_id = version.id
    embedding_model = version.embedding_model
    embedding_dimensions = version.embedding_dimensions
    embeddings = DeterministicEmbeddingClient()
    service = _service(
        connection=database_connection,
        settings=settings,
        payload=payload,
        embeddings=embeddings,
    )

    result = await service.index_version(version_id)
    db_session.expire_all()
    stored_version = await db_session.get(DocumentVersion, version_id)
    stored_document = await db_session.get(Document, document_id)
    paragraphs = list(
        await db_session.scalars(
            select(SourceParagraph)
            .where(SourceParagraph.document_version_id == version_id)
            .order_by(SourceParagraph.page_number, SourceParagraph.paragraph_number)
        )
    )
    chunks = list(
        await db_session.scalars(
            select(Chunk).where(Chunk.document_version_id == version_id).order_by(Chunk.ordinal)
        )
    )

    assert stored_version is not None and stored_version.status is IngestionStatus.SUCCEEDED
    assert stored_version.activated_at is not None
    assert stored_document is not None and stored_document.active_version_id == version_id
    assert result.paragraph_count == len(paragraphs) > 0
    assert result.chunk_count == len(chunks) > 0
    assert [chunk.ordinal for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk.embedding_model == embedding_model for chunk in chunks)
    assert all(chunk.embedding_dimensions == 1024 for chunk in chunks)
    assert any(len(chunk.provenance_snapshot) > 1 for chunk in chunks)

    first_vector = list(chunks[0].embedding)
    await db_session.execute(text("SET LOCAL enable_indexscan = off"))
    nearest_id = await db_session.scalar(
        select(Chunk.id)
        .where(
            Chunk.document_version_id == version_id,
            Chunk.embedding_model == embedding_model,
            Chunk.embedding_dimensions == embedding_dimensions,
        )
        .order_by(Chunk.embedding.cosine_distance(first_vector))
        .limit(1)
    )
    assert nearest_id == chunks[0].id
    link_count = await db_session.scalar(
        select(func.count())
        .select_from(ChunkSourceParagraph)
        .where(ChunkSourceParagraph.document_version_id == version_id)
    )
    assert link_count is not None and link_count >= len(chunks)


async def test_embedding_failure_is_retriable_and_never_activates_partial_index(
    db_session: AsyncSession,
    database_connection: AsyncConnection,
    settings: Settings,
) -> None:
    payload = SAMPLE_PDF.read_bytes()
    document, version = await _new_version(db_session, payload)
    document_id = document.id
    version_id = version.id
    embeddings = DeterministicEmbeddingClient(fail=True)
    service = _service(
        connection=database_connection,
        settings=settings.model_copy(update={"embedding_max_retries": 0}),
        payload=payload,
        embeddings=embeddings,
    )

    with pytest.raises(RetryableEmbeddingError):
        await service.index_version(version_id)
    db_session.expire_all()
    failed = await db_session.get(DocumentVersion, version_id)
    owner = await db_session.get(Document, document_id)
    assert failed is not None and failed.status is IngestionStatus.FAILED
    assert owner is not None and owner.active_version_id is None
    assert (
        await db_session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_version_id == version_id)
        )
        == 0
    )

    embeddings.fail = False
    await service.index_version(version_id)
    db_session.expire_all()
    succeeded = await db_session.get(DocumentVersion, version_id)
    assert succeeded is not None and succeeded.status is IngestionStatus.SUCCEEDED


async def test_transient_extraction_failure_can_retry_same_never_active_version(
    db_session: AsyncSession,
    database_connection: AsyncConnection,
    settings: Settings,
) -> None:
    payload = SAMPLE_PDF.read_bytes()
    document, version = await _new_version(db_session, payload)
    document_id = document.id
    version_id = version.id
    extractor = FlakyPdfExtractor()
    service = _service(
        connection=database_connection,
        settings=settings,
        payload=payload,
        embeddings=DeterministicEmbeddingClient(),
        extractor=extractor,
    )

    with pytest.raises(RuntimeError, match="simulated"):
        await service.index_version(version_id)
    db_session.expire_all()
    failed = await db_session.get(DocumentVersion, version_id)
    owner = await db_session.get(Document, document_id)
    assert failed is not None and failed.status is IngestionStatus.FAILED
    assert owner is not None and owner.active_version_id is None

    extractor.fail = False
    await service.index_version(version_id)
    db_session.expire_all()
    succeeded = await db_session.get(DocumentVersion, version_id)
    assert succeeded is not None and succeeded.status is IngestionStatus.SUCCEEDED


async def test_persistence_failure_rolls_back_publication_and_can_retry(
    db_session: AsyncSession,
    database_connection: AsyncConnection,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = SAMPLE_PDF.read_bytes()
    document, version = await _new_version(db_session, payload)
    document_id = document.id
    version_id = version.id
    service = _service(
        connection=database_connection,
        settings=settings,
        payload=payload,
        embeddings=DeterministicEmbeddingClient(),
    )
    original = DocumentRepository.publish_index
    calls = 0

    async def fail_once(self: DocumentRepository, **kwargs: Any) -> DocumentVersion:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SQLAlchemyError("simulated publication failure")
        return await original(self, **kwargs)

    monkeypatch.setattr(DocumentRepository, "publish_index", fail_once)
    with pytest.raises(SQLAlchemyError, match="simulated"):
        await service.index_version(version_id)
    db_session.expire_all()
    failed = await db_session.get(DocumentVersion, version_id)
    owner = await db_session.get(Document, document_id)
    assert failed is not None and failed.status is IngestionStatus.FAILED
    assert owner is not None and owner.active_version_id is None
    assert (
        await db_session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_version_id == version_id)
        )
        == 0
    )

    await service.index_version(version_id)
    db_session.expire_all()
    succeeded = await db_session.get(DocumentVersion, version_id)
    assert succeeded is not None and succeeded.status is IngestionStatus.SUCCEEDED
