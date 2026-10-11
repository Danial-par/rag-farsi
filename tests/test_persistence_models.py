from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_assistant.db.models import (
    Chunk,
    ChunkSourceParagraph,
    Document,
    DocumentVersion,
    ParagraphKind,
    SourceParagraph,
)
from rag_assistant.db.repositories import (
    ChunkParagraphReference,
    DocumentRepository,
    InvalidPersistenceStateError,
    NewChunk,
    NewDocumentVersion,
    NewSourceParagraph,
    UserRepository,
)


async def _processing_version(
    session: AsyncSession, *, object_key: str = "documents/source.pdf"
) -> tuple[DocumentRepository, DocumentVersion]:
    user = await UserRepository(session).create(
        email=f"{object_key.replace('/', '-')}@example.com",
        password_hash="test-password-hash",
    )
    repository = DocumentRepository(session)
    document = await repository.create_owned(owner_id=user.id, title="سند آزمایشی")
    version = await repository.create_owned_version(
        owner_id=user.id,
        document_id=document.id,
        values=NewDocumentVersion(
            object_key=object_key,
            content_hash="a" * 64,
            file_size_bytes=2048,
            embedding_model="openai/text-embedding-3-large",
            embedding_dimensions=1024,
        ),
    )
    await repository.mark_processing(version.id)
    return repository, version


async def test_application_generates_uuid_and_timezone_aware_timestamps(
    db_session: AsyncSession,
) -> None:
    user = await UserRepository(db_session).create(
        email="Timestamp@Example.com",
        password_hash="test-password-hash",
    )

    assert user.id.version == 4
    assert user.created_at.utcoffset() is not None
    assert user.updated_at.utcoffset() is not None
    assert user.normalized_email == "timestamp@example.com"


async def test_paragraph_citation_identity_is_unique_per_version_and_page(
    db_session: AsyncSession,
) -> None:
    repository, version = await _processing_version(db_session)
    await repository.add_source_paragraphs(
        version_id=version.id,
        paragraphs=[
            NewSourceParagraph(
                page_number=1,
                paragraph_number=1,
                kind=ParagraphKind.PROSE,
                original_text="متن اصلی",
                normalized_text="متن اصلی",
            )
        ],
    )

    try:
        async with db_session.begin_nested():
            db_session.add(
                SourceParagraph(
                    document_version_id=version.id,
                    page_number=1,
                    paragraph_number=1,
                    kind=ParagraphKind.HEADING,
                    original_text="عنوان",
                    normalized_text="عنوان",
                    paragraph_metadata={},
                )
            )
            await db_session.flush()
    except IntegrityError:
        pass
    else:
        raise AssertionError("Duplicate citation identity was accepted")


async def test_pgvector_dimensions_and_chunk_provenance_are_persisted(
    db_session: AsyncSession,
) -> None:
    repository, version = await _processing_version(db_session, object_key="documents/vector.pdf")
    (paragraph,) = await repository.add_source_paragraphs(
        version_id=version.id,
        paragraphs=[
            NewSourceParagraph(
                page_number=2,
                paragraph_number=3,
                kind=ParagraphKind.TABLE_ROW,
                original_text="ردیف اصلی جدول",
                normalized_text="سرستون | ردیف اصلی جدول",
            )
        ],
    )
    chunk = await repository.add_chunk(
        version_id=version.id,
        values=NewChunk(
            ordinal=1,
            normalized_text="سرستون | ردیف اصلی جدول",
            token_count=7,
            embedding=[0.001] * 1024,
            paragraphs=[ChunkParagraphReference(source_paragraph_id=paragraph.id)],
        ),
    )

    dimensions = await db_session.scalar(
        select(func.vector_dims(Chunk.embedding)).where(Chunk.id == chunk.id)
    )
    link = await db_session.scalar(
        select(ChunkSourceParagraph).where(ChunkSourceParagraph.chunk_id == chunk.id)
    )

    assert dimensions == 1024
    assert link is not None
    assert link.document_version_id == version.id
    assert chunk.provenance_snapshot == [
        {
            "source_paragraph_id": str(paragraph.id),
            "page_number": 2,
            "paragraph_number": 3,
            "start_offset": None,
            "end_offset": None,
        }
    ]


async def test_repository_rejects_wrong_vector_dimensions(db_session: AsyncSession) -> None:
    repository, version = await _processing_version(
        db_session, object_key="documents/wrong-vector.pdf"
    )
    (paragraph,) = await repository.add_source_paragraphs(
        version_id=version.id,
        paragraphs=[
            NewSourceParagraph(
                page_number=1,
                paragraph_number=1,
                kind=ParagraphKind.PROSE,
                original_text="اصل",
                normalized_text="اصل",
            )
        ],
    )

    try:
        await repository.add_chunk(
            version_id=version.id,
            values=NewChunk(
                ordinal=1,
                normalized_text="اصل",
                token_count=1,
                embedding=[0.1, 0.2],
                paragraphs=[ChunkParagraphReference(source_paragraph_id=paragraph.id)],
            ),
        )
    except InvalidPersistenceStateError:
        pass
    else:
        raise AssertionError("Wrong vector dimensions were accepted")


async def test_database_rejects_cross_version_chunk_provenance(
    db_session: AsyncSession,
) -> None:
    repository, first_version = await _processing_version(
        db_session, object_key="documents/provenance-v1.pdf"
    )
    (first_paragraph,) = await repository.add_source_paragraphs(
        version_id=first_version.id,
        paragraphs=[
            NewSourceParagraph(
                page_number=1,
                paragraph_number=1,
                kind=ParagraphKind.PROSE,
                original_text="نسخه یک",
                normalized_text="نسخه یک",
            )
        ],
    )
    first_chunk = await repository.add_chunk(
        version_id=first_version.id,
        values=NewChunk(
            ordinal=1,
            normalized_text="نسخه یک",
            token_count=2,
            embedding=[0.001] * 1024,
            paragraphs=[ChunkParagraphReference(source_paragraph_id=first_paragraph.id)],
        ),
    )
    owner_id = await db_session.scalar(
        select(Document.owner_id).where(Document.id == first_version.document_id)
    )
    assert owner_id is not None
    second_version = await repository.create_owned_version(
        owner_id=owner_id,
        document_id=first_version.document_id,
        values=NewDocumentVersion(
            object_key="documents/provenance-v2.pdf",
            content_hash="a" * 64,
            file_size_bytes=2048,
            embedding_model="openai/text-embedding-3-large",
            embedding_dimensions=1024,
        ),
    )
    await repository.mark_processing(second_version.id)
    (second_paragraph,) = await repository.add_source_paragraphs(
        version_id=second_version.id,
        paragraphs=[
            NewSourceParagraph(
                page_number=1,
                paragraph_number=1,
                kind=ParagraphKind.PROSE,
                original_text="نسخه دو",
                normalized_text="نسخه دو",
            )
        ],
    )

    try:
        async with db_session.begin_nested():
            db_session.add(
                ChunkSourceParagraph(
                    chunk_id=first_chunk.id,
                    source_paragraph_id=second_paragraph.id,
                    document_version_id=first_version.id,
                    ordinal=2,
                )
            )
            await db_session.flush()
    except IntegrityError:
        pass
    else:
        raise AssertionError("Cross-version provenance was accepted")


async def test_database_preserves_immutable_original_source_text(
    db_session: AsyncSession,
) -> None:
    repository, version = await _processing_version(
        db_session, object_key="documents/immutable-source.pdf"
    )
    (paragraph,) = await repository.add_source_paragraphs(
        version_id=version.id,
        paragraphs=[
            NewSourceParagraph(
                page_number=1,
                paragraph_number=1,
                kind=ParagraphKind.PROSE,
                original_text="متن اصلی",
                normalized_text="متن اصلی",
            )
        ],
    )

    try:
        async with db_session.begin_nested():
            await db_session.execute(
                update(SourceParagraph)
                .where(SourceParagraph.id == paragraph.id)
                .values(original_text="متن تغییر یافته")
            )
    except IntegrityError:
        pass
    else:
        raise AssertionError("Immutable source text was updated")
