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
from rag_assistant.retrieval.models import SectionOutlineItem


@dataclass(frozen=True, slots=True)
class RetrievalScope:
    conversation_exists: bool
    active_document_count: int
    incompatible_document_count: int
    active_version_ids: frozenset[UUID]


@dataclass(frozen=True, slots=True)
class CitationRecord:
    source_paragraph_id: UUID
    page_number: int
    paragraph_number: int
    kind: ParagraphKind
    original_text: str
    normalized_text: str
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
    section_path: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class ExpandedSectionRecord:
    section_id: UUID
    document_id: UUID
    document_version_id: UUID
    document_title: str
    heading_path: tuple[str, ...]
    normalized_text: str
    citations: tuple[CitationRecord, ...]
    truncated: bool


@dataclass(slots=True)
class _ExpandedSectionBuilder:
    document_id: UUID
    document_version_id: UUID
    document_title: str
    heading_path: tuple[str, ...]
    citations: dict[UUID, CitationRecord]


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
            func.array_agg(authorized.c.document_version_id).label("active_version_ids"),
        ).select_from(authorized)
        row = (await self._session.execute(statement)).one()
        return RetrievalScope(
            conversation_exists=bool(row.conversation_exists),
            active_document_count=int(row.active_document_count),
            incompatible_document_count=int(row.incompatible_document_count),
            active_version_ids=frozenset(row.active_version_ids or ()),
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
                Chunk.chunk_metadata,
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
                SourceParagraph.normalized_text,
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

    async def expand_sections(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        selections: Sequence[tuple[UUID, UUID, tuple[str, ...]]],
        max_characters: int,
    ) -> list[ExpandedSectionRecord]:
        """Load source paragraphs for model-selected sections inside authorized scope."""
        if not selections:
            return []
        authorized = self._authorized_active_documents(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        section_filters = [
            and_(
                authorized.c.document_version_id == version_id,
                Chunk.document_version_id == version_id,
                Chunk.chunk_metadata.contains({"section_path": [{"id": str(section_id)}]}),
            )
            for version_id, section_id, _ in selections
        ]
        statement = (
            select(
                authorized.c.document_id,
                authorized.c.document_version_id,
                authorized.c.document_title,
                Chunk.chunk_metadata,
                SourceParagraph.id.label("source_paragraph_id"),
                SourceParagraph.page_number,
                SourceParagraph.paragraph_number,
                SourceParagraph.kind,
                SourceParagraph.original_text,
                SourceParagraph.normalized_text,
            )
            .select_from(authorized)
            .join(
                Chunk,
                and_(
                    Chunk.document_version_id == authorized.c.document_version_id,
                    Chunk.embedding_model == authorized.c.embedding_model,
                    Chunk.embedding_dimensions == authorized.c.embedding_dimensions,
                ),
            )
            .join(
                ChunkSourceParagraph,
                and_(
                    ChunkSourceParagraph.chunk_id == Chunk.id,
                    ChunkSourceParagraph.document_version_id == Chunk.document_version_id,
                ),
            )
            .join(
                SourceParagraph,
                and_(
                    SourceParagraph.id == ChunkSourceParagraph.source_paragraph_id,
                    SourceParagraph.document_version_id == Chunk.document_version_id,
                ),
            )
            .where(or_(*section_filters))
            .order_by(
                authorized.c.document_version_id,
                SourceParagraph.page_number,
                SourceParagraph.paragraph_number,
            )
        )
        rows = (await self._session.execute(statement)).mappings().all()
        sections: dict[tuple[UUID, UUID], _ExpandedSectionBuilder] = {}
        for row in rows:
            row_path = (row["chunk_metadata"] or {}).get("section_path", [])
            for version_id, section_id, heading_path in selections:
                if row["document_version_id"] != version_id or not any(
                    item.get("id") == str(section_id) for item in row_path if isinstance(item, dict)
                ):
                    continue
                key = (version_id, section_id)
                section = sections.get(key)
                if section is None:
                    section = _ExpandedSectionBuilder(
                        document_id=row["document_id"],
                        document_version_id=version_id,
                        document_title=row["document_title"],
                        heading_path=heading_path,
                        citations={},
                    )
                    sections[key] = section
                source_id = row["source_paragraph_id"]
                section.citations.setdefault(
                    source_id,
                    CitationRecord(
                        source_paragraph_id=source_id,
                        page_number=int(row["page_number"]),
                        paragraph_number=int(row["paragraph_number"]),
                        kind=row["kind"],
                        original_text=row["original_text"],
                        normalized_text=row["normalized_text"],
                        start_offset=None,
                        end_offset=None,
                    ),
                )

        expanded: list[ExpandedSectionRecord] = []
        for version_id, section_id, _ in selections:
            section = sections.get((version_id, section_id))
            if section is None:
                continue
            citations = sorted(
                section.citations.values(),
                key=lambda item: (item.page_number, item.paragraph_number),
            )
            text_parts: list[str] = []
            included: list[CitationRecord] = []
            remaining = max_characters
            truncated = False
            for citation in citations:
                if remaining <= 0:
                    truncated = True
                    break
                value = citation.normalized_text
                if len(value) > remaining:
                    text_parts.append(value[:remaining])
                    included.append(citation)
                    truncated = True
                    break
                text_parts.append(value)
                included.append(citation)
                remaining -= len(value) + 2
            expanded.append(
                ExpandedSectionRecord(
                    section_id=section_id,
                    document_id=section.document_id,
                    document_version_id=version_id,
                    document_title=section.document_title,
                    heading_path=section.heading_path,
                    normalized_text="\n\n".join(text_parts),
                    citations=tuple(included),
                    truncated=truncated,
                )
            )
        return expanded

    async def section_outline(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        roots: Sequence[tuple[UUID, UUID]],
        max_sections: int,
    ) -> tuple[tuple[SectionOutlineItem, ...], bool]:
        """Load bounded heading metadata under candidate roots within authorized scope."""
        if not roots or max_sections <= 0:
            return (), False
        authorized = self._authorized_active_documents(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        section_filters = [
            and_(
                authorized.c.document_version_id == version_id,
                Chunk.document_version_id == version_id,
                Chunk.chunk_metadata.contains({"section_path": [{"id": str(section_id)}]}),
            )
            for version_id, section_id in roots
        ]
        path_expression = Chunk.chunk_metadata["section_path"]
        statement = (
            select(
                authorized.c.document_version_id,
                authorized.c.document_title,
                path_expression.label("section_path"),
                func.min(Chunk.ordinal).label("first_ordinal"),
            )
            .select_from(authorized)
            .join(
                Chunk,
                and_(
                    Chunk.document_version_id == authorized.c.document_version_id,
                    Chunk.embedding_model == authorized.c.embedding_model,
                    Chunk.embedding_dimensions == authorized.c.embedding_dimensions,
                ),
            )
            .where(or_(*section_filters))
            .group_by(
                authorized.c.document_version_id,
                authorized.c.document_title,
                path_expression,
            )
            .order_by(authorized.c.document_version_id, func.min(Chunk.ordinal))
            .limit(max_sections * 2 + 1)
        )
        rows = (await self._session.execute(statement)).mappings().all()
        outline: dict[tuple[UUID, UUID], SectionOutlineItem] = {}
        truncated = False
        for row in rows:
            path = row["section_path"]
            if not isinstance(path, list):
                continue
            heading_path: list[str] = []
            parent_id: UUID | None = None
            version_id = row["document_version_id"]
            for index, item in enumerate(path):
                if not isinstance(item, dict):
                    continue
                raw_id = item.get("id")
                title = item.get("title")
                if not isinstance(raw_id, str) or not isinstance(title, str) or not title.strip():
                    continue
                try:
                    section_id = UUID(raw_id)
                except ValueError:
                    continue
                key = (version_id, section_id)
                heading_path.append(title[:500])
                if key not in outline:
                    raw_level = item.get("level")
                    level = raw_level if isinstance(raw_level, int) and raw_level > 0 else index + 1
                    outline[key] = SectionOutlineItem(
                        document_version_id=version_id,
                        section_id=section_id,
                        parent_section_id=parent_id,
                        document_title=row["document_title"][:500],
                        title=title[:500],
                        level=min(level, 32),
                        heading_path=tuple(heading_path),
                    )
                    if len(outline) > max_sections:
                        truncated = True
                        break
                parent_id = section_id
            if truncated:
                break
        if truncated:
            outline.pop(next(reversed(outline)), None)
        return tuple(outline.values()), truncated

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
                        normalized_text=row["normalized_text"],
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
                    section_path=tuple(
                        item
                        for item in (row["chunk_metadata"] or {}).get("section_path", [])
                        if isinstance(item, dict)
                        and isinstance(item.get("id"), str)
                        and isinstance(item.get("title"), str)
                        and isinstance(item.get("level"), int)
                    ),
                )
            )
        return records
