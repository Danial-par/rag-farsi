from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_assistant.db.models import DocumentVersion, IngestionStatus, SourceParagraph
from rag_assistant.db.repositories import (
    DocumentRepository,
    ExtractionConflictError,
    NewDocumentVersion,
    NewSourceParagraph,
    UserRepository,
)
from rag_assistant.extraction import (
    DocumentExtractionService,
    ExtractionFailureCode,
    PdfExtractionError,
)

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "documents" / "sample_doc.pdf"


def _text_page_followed_by_blank_page() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
    )
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 50 250 Td (private-content-marker) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    writer.add_blank_page(width=300, height=300)
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


async def _processing_version(
    session: AsyncSession,
    *,
    object_key: str,
) -> tuple[DocumentRepository, DocumentVersion]:
    user = await UserRepository(session).create(
        email=f"{object_key.replace('/', '-')}@example.com",
        password_hash="test-password-hash",
    )
    repository = DocumentRepository(session)
    document = await repository.create_owned(owner_id=user.id, title="سند استخراج")
    version = await repository.create_owned_version(
        owner_id=user.id,
        document_id=document.id,
        values=NewDocumentVersion(
            object_key=object_key,
            content_hash="f" * 64,
            file_size_bytes=2048,
            embedding_model="openai/text-embedding-3-large",
            embedding_dimensions=1024,
        ),
    )
    await repository.mark_processing(version.id)
    return repository, version


async def test_extraction_is_persisted_atomically_and_retries_preserve_ids(
    db_session: AsyncSession,
) -> None:
    repository, version = await _processing_version(
        db_session,
        object_key="documents/extracted-sample.pdf",
    )
    service = DocumentExtractionService(repository)
    payload = SAMPLE_PDF.read_bytes()

    first_result = await service.extract_and_persist(version_id=version.id, source=payload)
    first_rows = list(
        await db_session.scalars(
            select(SourceParagraph)
            .where(SourceParagraph.document_version_id == version.id)
            .order_by(SourceParagraph.page_number, SourceParagraph.paragraph_number)
        )
    )
    first_ids = [row.id for row in first_rows]
    second_result = await service.extract_and_persist(version_id=version.id, source=payload)
    second_rows = list(
        await db_session.scalars(
            select(SourceParagraph)
            .where(SourceParagraph.document_version_id == version.id)
            .order_by(SourceParagraph.page_number, SourceParagraph.paragraph_number)
        )
    )

    assert first_result.model_dump(mode="json") == second_result.model_dump(mode="json")
    assert [row.id for row in second_rows] == first_ids
    assert len(first_rows) == len(first_result.paragraphs)
    assert version.status is IngestionStatus.PROCESSING
    assert version.extraction_metadata["page_count"] == 13
    assert all(row.paragraph_metadata["bounding_box"] for row in first_rows)


async def test_conflicting_extraction_retry_is_rejected(
    db_session: AsyncSession,
) -> None:
    repository, version = await _processing_version(
        db_session,
        object_key="documents/conflicting-extraction.pdf",
    )
    service = DocumentExtractionService(repository)
    await service.extract_and_persist(version_id=version.id, source=SAMPLE_PDF.read_bytes())
    persisted = list(
        await db_session.scalars(
            select(SourceParagraph)
            .where(SourceParagraph.document_version_id == version.id)
            .order_by(SourceParagraph.page_number, SourceParagraph.paragraph_number)
        )
    )
    altered = [
        NewSourceParagraph(
            page_number=row.page_number,
            paragraph_number=row.paragraph_number,
            kind=row.kind,
            original_text=row.original_text,
            normalized_text=(
                f"{row.normalized_text} changed" if index == 0 else row.normalized_text
            ),
            metadata=row.paragraph_metadata,
        )
        for index, row in enumerate(persisted)
    ]

    with pytest.raises(ExtractionConflictError):
        await repository.persist_extraction(
            version_id=version.id,
            paragraphs=altered,
            extraction_metadata=version.extraction_metadata,
        )


async def test_extraction_failure_marks_version_failed_without_source_content(
    db_session: AsyncSession,
) -> None:
    repository, version = await _processing_version(
        db_session,
        object_key="documents/malformed.pdf",
    )
    service = DocumentExtractionService(repository)
    sensitive_payload = b"not-a-pdf private document words"

    with pytest.raises(PdfExtractionError) as captured:
        await service.extract_and_persist(version_id=version.id, source=sensitive_payload)

    assert captured.value.code is ExtractionFailureCode.MALFORMED
    assert version.status is IngestionStatus.FAILED
    assert version.failure_code == ExtractionFailureCode.MALFORMED.value
    assert "private document words" not in (version.failure_message or "")
    assert "private document words" not in str(version.extraction_metadata)


async def test_extraction_warnings_are_persisted_without_document_text(
    db_session: AsyncSession,
) -> None:
    repository, version = await _processing_version(
        db_session,
        object_key="documents/warning.pdf",
    )

    await DocumentExtractionService(repository).extract_and_persist(
        version_id=version.id,
        source=_text_page_followed_by_blank_page(),
    )

    assert version.extraction_metadata["warnings"] == [
        {
            "code": "blank_page",
            "message": "Physical page contains no extractable source text",
            "page_number": 2,
        }
    ]
    assert "private-content-marker" not in str(version.extraction_metadata)
