from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import func, select
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
from rag_assistant.indexing.service import DocumentIndexingService
from rag_assistant.indexing.storage import LocalObjectStore
from rag_assistant.sample_seed import seed_sample_document

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "documents" / "sample_doc.pdf"


class DeterministicEmbeddingClient:
    async def embed(
        self, texts: Sequence[str], *, model: str, dimensions: int
    ) -> list[list[float]]:
        del model
        return [
            [float(index + 1) / dimensions for _ in range(dimensions)]
            for index, _ in enumerate(texts)
        ]


async def test_sample_seed_is_idempotent_for_pending_and_active_versions(
    database_connection: AsyncConnection,
    settings: Settings,
    tmp_path: Path,
) -> None:
    session_factory = async_sessionmaker(
        bind=database_connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    local_settings = settings.model_copy(
        update={"app_env": "local", "local_object_store_root": tmp_path}
    )
    store = LocalObjectStore(tmp_path)
    first = await seed_sample_document(
        settings=local_settings,
        session_factory=session_factory,
        object_store=store,
        fixture_path=SAMPLE_PDF,
    )
    repeated = await seed_sample_document(
        settings=local_settings,
        session_factory=session_factory,
        object_store=store,
        fixture_path=SAMPLE_PDF,
    )

    assert first.document_id == repeated.document_id
    assert first.version_id == repeated.version_id
    assert first.status is IngestionStatus.PENDING
    assert repeated.already_active is False

    service = DocumentIndexingService(
        session_factory=session_factory,
        object_store=store,
        embedding_client=DeterministicEmbeddingClient(),
        settings=local_settings,
    )
    result = await service.index_version(first.version_id)
    active_repeat = await seed_sample_document(
        settings=local_settings,
        session_factory=session_factory,
        object_store=store,
        fixture_path=SAMPLE_PDF,
    )

    assert active_repeat.version_id == first.version_id
    assert active_repeat.already_active is True
    assert active_repeat.status is IngestionStatus.SUCCEEDED
    assert result.paragraph_count > 0
    assert result.chunk_count > 0

    updated_settings = local_settings.model_copy(update={"chunk_target_tokens": 350})
    reindexed = await seed_sample_document(
        settings=updated_settings,
        session_factory=session_factory,
        object_store=store,
        fixture_path=SAMPLE_PDF,
    )

    assert reindexed.document_id == first.document_id
    assert reindexed.version_id != first.version_id
    assert reindexed.status is IngestionStatus.PENDING
    assert reindexed.already_active is False
    async with session_factory() as session:
        document = await session.get(Document, first.document_id)
    assert document is not None and document.active_version_id == first.version_id


async def test_sample_seed_refuses_non_local_environment(settings: Settings) -> None:
    with pytest.raises(ValueError, match="APP_ENV=local"):
        await seed_sample_document(
            settings=settings.model_copy(update={"app_env": "production"}),
            session_factory=None,  # type: ignore[arg-type]
            object_store=None,  # type: ignore[arg-type]
            fixture_path=SAMPLE_PDF,
        )


async def test_index_verification_counts_persisted_provenance_and_dimensions(
    db_session: AsyncSession,
    database_connection: AsyncConnection,
    settings: Settings,
    tmp_path: Path,
) -> None:
    del db_session
    session_factory = async_sessionmaker(
        bind=database_connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    local_settings = settings.model_copy(
        update={"app_env": "local", "local_object_store_root": tmp_path}
    )
    store = LocalObjectStore(tmp_path)
    seeded = await seed_sample_document(
        settings=local_settings,
        session_factory=session_factory,
        object_store=store,
        fixture_path=SAMPLE_PDF,
    )
    result = await DocumentIndexingService(
        session_factory=session_factory,
        object_store=store,
        embedding_client=DeterministicEmbeddingClient(),
        settings=local_settings,
    ).index_version(seeded.version_id)

    async with session_factory() as session:
        version = await session.get(DocumentVersion, seeded.version_id)
        document = await session.get(Document, seeded.document_id)
        paragraphs = list(
            await session.scalars(
                select(SourceParagraph).where(
                    SourceParagraph.document_version_id == seeded.version_id
                )
            )
        )
        chunks = list(
            await session.scalars(
                select(Chunk).where(Chunk.document_version_id == seeded.version_id)
            )
        )
        links = list(
            await session.scalars(
                select(ChunkSourceParagraph).where(
                    ChunkSourceParagraph.document_version_id == seeded.version_id
                )
            )
        )
        dimensions: set[int] = set(
            await session.scalars(
                select(func.vector_dims(Chunk.embedding)).where(
                    Chunk.document_version_id == seeded.version_id
                )
            )
        )

    assert version is not None and version.status is IngestionStatus.SUCCEEDED
    assert document is not None and document.active_version_id == seeded.version_id
    assert len(paragraphs) == result.paragraph_count > 0
    assert len(chunks) == result.chunk_count > 0
    assert len(links) >= len(chunks)
    assert dimensions == {1024}
    paragraph_ids = {paragraph.id for paragraph in paragraphs}
    assert all(link.source_paragraph_id in paragraph_ids for link in links)
    assert all(isinstance(link.chunk_id, UUID) for link in links)
