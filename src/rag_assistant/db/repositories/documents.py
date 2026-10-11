from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_assistant.db.base import utc_now
from rag_assistant.db.models import (
    Chunk,
    ChunkSourceParagraph,
    Conversation,
    ConversationDocument,
    Document,
    DocumentVersion,
    IngestionStatus,
    ParagraphKind,
    SourceParagraph,
)
from rag_assistant.db.models.document import EMBEDDING_DIMENSIONS
from rag_assistant.db.repositories.errors import (
    ExtractionConflictError,
    InvalidPersistenceStateError,
    RecordNotFoundError,
)


@dataclass(frozen=True, slots=True)
class NewDocumentVersion:
    object_key: str
    content_hash: str
    file_size_bytes: int
    embedding_model: str
    embedding_dimensions: int
    media_type: str = "application/pdf"
    ingestion_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NewSourceParagraph:
    page_number: int
    paragraph_number: int
    kind: ParagraphKind
    original_text: str
    normalized_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkParagraphReference:
    source_paragraph_id: UUID
    start_offset: int | None = None
    end_offset: int | None = None


@dataclass(frozen=True, slots=True)
class NewChunk:
    ordinal: int
    normalized_text: str
    token_count: int
    embedding: Sequence[float]
    paragraphs: Sequence[ChunkParagraphReference]
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentRepository:
    """Aggregate operations for owned and globally shared documents."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_owned(
        self,
        *,
        owner_id: UUID,
        title: str,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        document = Document(
            owner_id=owner_id,
            title=title,
            is_shared=False,
            is_default=False,
            document_metadata=metadata or {},
        )
        self._session.add(document)
        await self._session.flush()
        return document

    async def create_shared(
        self,
        *,
        title: str,
        default_for_all_users: bool,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        """Create a system-managed shared document from a trusted bootstrap service."""
        document = Document(
            owner_id=None,
            title=title,
            is_shared=True,
            is_default=default_for_all_users,
            document_metadata=metadata or {},
        )
        self._session.add(document)
        await self._session.flush()
        return document

    async def get_owned(self, *, owner_id: UUID, document_id: UUID) -> Document | None:
        return await self._session.scalar(
            select(Document).where(
                Document.id == document_id,
                Document.owner_id == owner_id,
                Document.is_shared.is_(False),
                Document.deleted_at.is_(None),
            )
        )

    async def get_accessible(self, *, user_id: UUID, document_id: UUID) -> Document | None:
        return await self._session.scalar(
            select(Document).where(
                Document.id == document_id,
                or_(Document.owner_id == user_id, Document.is_shared.is_(True)),
                Document.deleted_at.is_(None),
            )
        )

    async def list_accessible(self, *, user_id: UUID) -> list[Document]:
        documents = await self._session.scalars(
            select(Document)
            .where(
                or_(Document.owner_id == user_id, Document.is_shared.is_(True)),
                Document.deleted_at.is_(None),
            )
            .order_by(Document.title, Document.id)
        )
        return list(documents)

    async def create_owned_version(
        self,
        *,
        owner_id: UUID,
        document_id: UUID,
        values: NewDocumentVersion,
    ) -> DocumentVersion:
        document = await self._locked_owned_document(owner_id, document_id)
        return await self._create_version(document, values)

    async def create_shared_version(
        self, *, document_id: UUID, values: NewDocumentVersion
    ) -> DocumentVersion:
        document = await self._session.scalar(
            select(Document)
            .where(
                Document.id == document_id,
                Document.is_shared.is_(True),
                Document.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if document is None:
            raise RecordNotFoundError("Shared document not found")
        return await self._create_version(document, values)

    async def _create_version(
        self, document: Document, values: NewDocumentVersion
    ) -> DocumentVersion:
        if values.embedding_dimensions != EMBEDDING_DIMENSIONS:
            raise InvalidPersistenceStateError(
                f"Schema supports {EMBEDDING_DIMENSIONS}-dimension embeddings"
            )
        latest_number = await self._session.scalar(
            select(func.max(DocumentVersion.version_number)).where(
                DocumentVersion.document_id == document.id
            )
        )
        version = DocumentVersion(
            document_id=document.id,
            version_number=(latest_number or 0) + 1,
            object_key=values.object_key,
            content_hash=values.content_hash,
            file_size_bytes=values.file_size_bytes,
            media_type=values.media_type,
            embedding_model=values.embedding_model,
            embedding_dimensions=values.embedding_dimensions,
            ingestion_config=values.ingestion_config,
            extraction_metadata={},
            status=IngestionStatus.PENDING,
        )
        self._session.add(version)
        await self._session.flush()
        return version

    async def mark_processing(self, version_id: UUID) -> DocumentVersion:
        version = await self._locked_version(version_id)
        if version.status is not IngestionStatus.PENDING:
            raise InvalidPersistenceStateError("Only pending versions can start processing")
        version.status = IngestionStatus.PROCESSING
        version.started_at = utc_now()
        await self._session.flush()
        return version

    async def mark_succeeded(
        self, version_id: UUID, *, extraction_metadata: dict[str, Any]
    ) -> DocumentVersion:
        version = await self._locked_version(version_id)
        if version.status is not IngestionStatus.PROCESSING:
            raise InvalidPersistenceStateError("Only processing versions can succeed")
        version.status = IngestionStatus.SUCCEEDED
        version.extraction_metadata = extraction_metadata
        version.completed_at = utc_now()
        await self._session.flush()
        return version

    async def mark_failed(
        self,
        version_id: UUID,
        *,
        failure_code: str,
        failure_message: str,
        extraction_metadata: dict[str, Any] | None = None,
    ) -> DocumentVersion:
        version = await self._locked_version(version_id)
        if version.status not in {IngestionStatus.PENDING, IngestionStatus.PROCESSING}:
            raise InvalidPersistenceStateError("Only unfinished versions can fail")
        version.status = IngestionStatus.FAILED
        version.failed_at = utc_now()
        version.failure_code = failure_code
        version.failure_message = failure_message
        if extraction_metadata is not None:
            version.extraction_metadata = extraction_metadata
        await self._session.flush()
        return version

    async def activate_owned_version(
        self, *, owner_id: UUID, document_id: UUID, version_id: UUID
    ) -> Document:
        document = await self._locked_owned_document(owner_id, document_id)
        await self._activate(document, version_id)
        return document

    async def activate_shared_version(self, *, document_id: UUID, version_id: UUID) -> Document:
        document = await self._session.scalar(
            select(Document)
            .where(
                Document.id == document_id,
                Document.is_shared.is_(True),
                Document.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if document is None:
            raise RecordNotFoundError("Shared document not found")
        await self._activate(document, version_id)
        if document.is_default:
            await self._attach_default_document_to_conversations(document.id)
        return document

    async def _activate(self, document: Document, version_id: UUID) -> None:
        version = await self._session.scalar(
            select(DocumentVersion)
            .where(
                DocumentVersion.id == version_id,
                DocumentVersion.document_id == document.id,
            )
            .with_for_update()
        )
        if version is None:
            raise RecordNotFoundError("Document version not found")
        if version.status is not IngestionStatus.SUCCEEDED:
            raise InvalidPersistenceStateError("Only successfully ingested versions can be active")
        if version.activated_at is None:
            version.activated_at = utc_now()
            await self._session.flush()
        document.active_version_id = version.id
        await self._session.flush()

    async def add_source_paragraphs(
        self,
        *,
        version_id: UUID,
        paragraphs: Sequence[NewSourceParagraph],
    ) -> list[SourceParagraph]:
        version = await self._locked_version(version_id)
        if version.status is not IngestionStatus.PROCESSING:
            raise InvalidPersistenceStateError("Paragraphs can only be added while processing")
        models = [
            SourceParagraph(
                document_version_id=version.id,
                page_number=value.page_number,
                paragraph_number=value.paragraph_number,
                kind=value.kind,
                original_text=value.original_text,
                normalized_text=value.normalized_text,
                paragraph_metadata=value.metadata,
            )
            for value in paragraphs
        ]
        self._session.add_all(models)
        await self._session.flush()
        return models

    async def persist_extraction(
        self,
        *,
        version_id: UUID,
        paragraphs: Sequence[NewSourceParagraph],
        extraction_metadata: dict[str, Any],
    ) -> list[SourceParagraph]:
        """Atomically persist a deterministic extraction, preserving IDs on retries."""
        version = await self._locked_version(version_id)
        if version.status is not IngestionStatus.PROCESSING:
            raise InvalidPersistenceStateError(
                "Extraction can only be persisted while a version is processing"
            )
        if not paragraphs:
            raise InvalidPersistenceStateError("Extraction must contain source paragraphs")

        existing = list(
            await self._session.scalars(
                select(SourceParagraph)
                .where(SourceParagraph.document_version_id == version.id)
                .order_by(SourceParagraph.page_number, SourceParagraph.paragraph_number)
            )
        )
        if existing:
            if version.extraction_metadata == extraction_metadata and self._paragraphs_match(
                existing, paragraphs
            ):
                return existing
            raise ExtractionConflictError(
                "Persisted extraction conflicts with the deterministic retry result"
            )

        models = [
            SourceParagraph(
                document_version_id=version.id,
                page_number=value.page_number,
                paragraph_number=value.paragraph_number,
                kind=value.kind,
                original_text=value.original_text,
                normalized_text=value.normalized_text,
                paragraph_metadata=value.metadata,
            )
            for value in paragraphs
        ]
        version.extraction_metadata = extraction_metadata
        self._session.add_all(models)
        await self._session.flush()
        return models

    @staticmethod
    def _paragraphs_match(
        existing: Sequence[SourceParagraph],
        expected: Sequence[NewSourceParagraph],
    ) -> bool:
        if len(existing) != len(expected):
            return False
        return all(
            current.page_number == value.page_number
            and current.paragraph_number == value.paragraph_number
            and current.kind is value.kind
            and current.original_text == value.original_text
            and current.normalized_text == value.normalized_text
            and current.paragraph_metadata == value.metadata
            for current, value in zip(existing, expected, strict=True)
        )

    async def add_chunk(self, *, version_id: UUID, values: NewChunk) -> Chunk:
        version = await self._locked_version(version_id)
        if version.status is not IngestionStatus.PROCESSING:
            raise InvalidPersistenceStateError("Chunks can only be added while processing")
        if len(values.embedding) != version.embedding_dimensions:
            raise InvalidPersistenceStateError("Embedding dimensions do not match the version")
        paragraph_ids = [item.source_paragraph_id for item in values.paragraphs]
        if not paragraph_ids or len(paragraph_ids) != len(set(paragraph_ids)):
            raise InvalidPersistenceStateError("Chunk provenance must contain unique paragraphs")
        paragraph_models = await self._session.scalars(
            select(SourceParagraph).where(
                SourceParagraph.document_version_id == version.id,
                SourceParagraph.id.in_(paragraph_ids),
            )
        )
        paragraphs_by_id = {paragraph.id: paragraph for paragraph in paragraph_models}
        if set(paragraph_ids) != paragraphs_by_id.keys():
            raise InvalidPersistenceStateError(
                "Every chunk provenance paragraph must belong to the same version"
            )

        snapshot: list[dict[str, Any]] = []
        chunk = Chunk(
            document_version_id=version.id,
            ordinal=values.ordinal,
            normalized_text=values.normalized_text,
            token_count=values.token_count,
            embedding=list(values.embedding),
            embedding_model=version.embedding_model,
            embedding_dimensions=version.embedding_dimensions,
            chunk_metadata=values.metadata,
            provenance_snapshot=snapshot,
        )
        self._session.add(chunk)
        await self._session.flush()

        links: list[ChunkSourceParagraph] = []
        for ordinal, reference in enumerate(values.paragraphs, start=1):
            paragraph = paragraphs_by_id[reference.source_paragraph_id]
            snapshot.append(
                {
                    "source_paragraph_id": str(paragraph.id),
                    "page_number": paragraph.page_number,
                    "paragraph_number": paragraph.paragraph_number,
                    "start_offset": reference.start_offset,
                    "end_offset": reference.end_offset,
                }
            )
            links.append(
                ChunkSourceParagraph(
                    chunk_id=chunk.id,
                    source_paragraph_id=paragraph.id,
                    document_version_id=version.id,
                    ordinal=ordinal,
                    start_offset=reference.start_offset,
                    end_offset=reference.end_offset,
                )
            )
        chunk.provenance_snapshot = snapshot
        self._session.add_all(links)
        await self._session.flush()
        return chunk

    async def soft_delete_owned(self, *, owner_id: UUID, document_id: UUID) -> Document:
        document = await self._locked_owned_document(owner_id, document_id)
        if document.deleted_at is None:
            document.deleted_at = utc_now()
            await self._session.flush()
        return document

    async def hard_delete_never_activated_version(
        self, *, owner_id: UUID, document_id: UUID, version_id: UUID
    ) -> None:
        document = await self._locked_owned_document(owner_id, document_id)
        version = await self._session.scalar(
            select(DocumentVersion)
            .where(
                DocumentVersion.id == version_id,
                DocumentVersion.document_id == document.id,
            )
            .with_for_update()
        )
        if version is None:
            raise RecordNotFoundError("Document version not found")
        if document.active_version_id == version.id or version.activated_at is not None:
            raise InvalidPersistenceStateError("Activated document versions cannot be deleted")
        await self._session.delete(version)
        await self._session.flush()

    async def _locked_owned_document(self, owner_id: UUID, document_id: UUID) -> Document:
        document = await self._session.scalar(
            select(Document)
            .where(
                Document.id == document_id,
                Document.owner_id == owner_id,
                Document.is_shared.is_(False),
                Document.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if document is None:
            raise RecordNotFoundError("Owned document not found")
        return document

    async def _locked_version(self, version_id: UUID) -> DocumentVersion:
        version = await self._session.scalar(
            select(DocumentVersion).where(DocumentVersion.id == version_id).with_for_update()
        )
        if version is None:
            raise RecordNotFoundError("Document version not found")
        return version

    async def _attach_default_document_to_conversations(self, document_id: UUID) -> None:
        conversation_ids = await self._session.scalars(
            select(Conversation.id).where(Conversation.deleted_at.is_(None))
        )
        for conversation_id in conversation_ids:
            existing = await self._session.get(
                ConversationDocument,
                {"conversation_id": conversation_id, "document_id": document_id},
            )
            if existing is None:
                self._session.add(
                    ConversationDocument(
                        conversation_id=conversation_id,
                        document_id=document_id,
                    )
                )
        await self._session.flush()
