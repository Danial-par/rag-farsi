from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import and_, exists, func, literal, or_, select, text, true
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.selectable import CTE

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
from rag_assistant.retrieval.errors import RetrievalIntegrityError


@dataclass(frozen=True, slots=True)
class RetrievalScope:
    conversation_exists: bool
    active_document_count: int
    incompatible_document_count: int


@dataclass(frozen=True, slots=True)
class CitationRecord:
    source_paragraph_id: UUID
    page_number: int
    paragraph_number: int
    kind: ParagraphKind
    original_text: str
    start_offset: int | None
    end_offset: int | None


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    document_title: str
    chunk_ordinal: int
    normalized_text: str
    token_count: int
    cosine_similarity: float
    citations: tuple[CitationRecord, ...]


@dataclass(slots=True)
class _EvidenceBuilder:
    row: RowMapping
    citations: list[tuple[int, CitationRecord]]


class RetrievalRepository:
    """Run authorization-scoped exact vector retrieval and resolve provenance."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def inspect_scope(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        embedding_model: str,
        embedding_dimensions: int,
    ) -> RetrievalScope:
        authorized = self._authorized_active_documents(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        incompatible = or_(
            authorized.c.embedding_model != embedding_model,
            authorized.c.embedding_dimensions != embedding_dimensions,
        )
        statement = select(
            exists(
                select(Conversation.id).where(
                    Conversation.id == conversation_id,
                    Conversation.owner_id == user_id,
                    Conversation.deleted_at.is_(None),
                )
            ).label("conversation_exists"),
            func.count(authorized.c.document_id).label("active_document_count"),
            func.count(authorized.c.document_id)
            .filter(incompatible)
            .label("incompatible_document_count"),
        ).select_from(authorized)
        row = (await self._session.execute(statement)).one()
        return RetrievalScope(
            conversation_exists=bool(row.conversation_exists),
            active_document_count=int(row.active_document_count),
            incompatible_document_count=int(row.incompatible_document_count),
        )

    async def retrieve(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        query_embedding: list[float],
        embedding_model: str,
        embedding_dimensions: int,
        top_k: int,
        minimum_similarity: float | None,
    ) -> list[EvidenceRecord]:
        authorized = self._authorized_active_documents(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        incompatible_exists = exists(
            select(authorized.c.document_id).where(
                or_(
                    authorized.c.embedding_model != embedding_model,
                    authorized.c.embedding_dimensions != embedding_dimensions,
                )
            )
        )
        distance = Chunk.embedding.cosine_distance(query_embedding)
        candidates = (
            select(
                Chunk.id.label("chunk_id"),
                authorized.c.document_id,
                authorized.c.document_version_id,
                authorized.c.document_title,
                Chunk.ordinal.label("chunk_ordinal"),
                Chunk.normalized_text,
                Chunk.token_count,
                distance.label("cosine_distance"),
            )
            .select_from(authorized)
            .join(
                Chunk,
                and_(
                    Chunk.document_version_id == authorized.c.document_version_id,
                    Chunk.embedding_model == embedding_model,
                    Chunk.embedding_dimensions == embedding_dimensions,
                ),
            )
            .where(
                authorized.c.embedding_model == embedding_model,
                authorized.c.embedding_dimensions == embedding_dimensions,
                ~incompatible_exists,
                distance.is_not(None),
            )
        )
        if minimum_similarity is not None:
            candidates = candidates.where((literal(1.0) - distance) >= minimum_similarity)

        ranked = (
            candidates.order_by(
                distance,
                authorized.c.document_id,
                Chunk.ordinal,
                Chunk.id,
            )
            .limit(top_k)
            .cte("ranked_chunks")
        )
        statement = (
            select(
                ranked,
                ChunkSourceParagraph.ordinal.label("citation_ordinal"),
                ChunkSourceParagraph.source_paragraph_id,
                ChunkSourceParagraph.start_offset,
                ChunkSourceParagraph.end_offset,
                SourceParagraph.page_number,
                SourceParagraph.paragraph_number,
                SourceParagraph.kind,
                SourceParagraph.original_text,
            )
            .select_from(ranked)
            .outerjoin(
                ChunkSourceParagraph,
                and_(
                    ChunkSourceParagraph.chunk_id == ranked.c.chunk_id,
                    ChunkSourceParagraph.document_version_id == ranked.c.document_version_id,
                ),
            )
            .outerjoin(
                SourceParagraph,
                and_(
                    SourceParagraph.id == ChunkSourceParagraph.source_paragraph_id,
                    SourceParagraph.document_version_id == ranked.c.document_version_id,
                ),
            )
            .order_by(
                ranked.c.cosine_distance,
                ranked.c.document_id,
                ranked.c.chunk_ordinal,
                ranked.c.chunk_id,
                ChunkSourceParagraph.ordinal,
            )
        )

        # pgvector's HNSW index is approximate. Disabling index scans for this statement
        # preserves the step's exact-ranking contract, then restores the prior setting.
        previous_indexscan = await self._session.scalar(
            text("SELECT current_setting('enable_indexscan')")
        )
        await self._session.execute(text("SET LOCAL enable_indexscan = off"))
        rows = (await self._session.execute(statement)).mappings().all()
        await self._session.execute(
            text("SELECT set_config('enable_indexscan', :value, true)"),
            {"value": previous_indexscan},
        )
        return self._records(rows)

    @staticmethod
    def _authorized_active_documents(
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> CTE:
        owned_conversation = (
            select(Conversation.id)
            .where(
                Conversation.id == conversation_id,
                Conversation.owner_id == user_id,
                Conversation.deleted_at.is_(None),
            )
            .cte("owned_conversation")
        )
        explicitly_selected = exists(
            select(ConversationDocument.document_id).where(
                ConversationDocument.conversation_id == owned_conversation.c.id,
                ConversationDocument.document_id == Document.id,
            )
        )
        return (
            select(
                Document.id.label("document_id"),
                Document.title.label("document_title"),
                DocumentVersion.id.label("document_version_id"),
                DocumentVersion.embedding_model,
                DocumentVersion.embedding_dimensions,
            )
            .select_from(owned_conversation)
            .join(Document, true())
            .join(
                DocumentVersion,
                and_(
                    DocumentVersion.id == Document.active_version_id,
                    DocumentVersion.document_id == Document.id,
                ),
            )
            .where(
                Document.deleted_at.is_(None),
                or_(Document.owner_id == user_id, Document.is_shared.is_(True)),
                or_(
                    explicitly_selected,
                    and_(
                        Document.is_shared.is_(True),
                        Document.is_default.is_(True),
                    ),
                ),
                DocumentVersion.status == IngestionStatus.SUCCEEDED,
                DocumentVersion.activated_at.is_not(None),
            )
            .cte("authorized_active_documents")
        )

    @staticmethod
    def _records(rows: Sequence[RowMapping]) -> list[EvidenceRecord]:
        chunks: dict[UUID, _EvidenceBuilder] = {}
        for row in rows:
            chunk_id = row["chunk_id"]
            current = chunks.setdefault(
                chunk_id,
                _EvidenceBuilder(row=row, citations=[]),
            )
            source_paragraph_id = row["source_paragraph_id"]
            if (
                source_paragraph_id is None
                or row["citation_ordinal"] is None
                or row["page_number"] is None
                or row["paragraph_number"] is None
                or row["kind"] is None
                or row["original_text"] is None
            ):
                raise RetrievalIntegrityError(
                    "A retrieved chunk does not resolve to authoritative source text"
                )
            current.citations.append(
                (
                    int(row["citation_ordinal"]),
                    CitationRecord(
                        source_paragraph_id=source_paragraph_id,
                        page_number=int(row["page_number"]),
                        paragraph_number=int(row["paragraph_number"]),
                        kind=row["kind"],
                        original_text=row["original_text"],
                        start_offset=row["start_offset"],
                        end_offset=row["end_offset"],
                    ),
                )
            )

        records: list[EvidenceRecord] = []
        for value in chunks.values():
            row = value.row
            ordered_citations = sorted(value.citations, key=lambda item: item[0])
            if [ordinal for ordinal, _ in ordered_citations] != list(
                range(1, len(ordered_citations) + 1)
            ):
                raise RetrievalIntegrityError("Chunk citation ordering is incomplete")
            records.append(
                EvidenceRecord(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    document_version_id=row["document_version_id"],
                    document_title=row["document_title"],
                    chunk_ordinal=int(row["chunk_ordinal"]),
                    normalized_text=row["normalized_text"],
                    token_count=int(row["token_count"]),
                    cosine_similarity=1.0 - float(row["cosine_distance"]),
                    citations=tuple(citation for _, citation in ordered_citations),
                )
            )
        return records
