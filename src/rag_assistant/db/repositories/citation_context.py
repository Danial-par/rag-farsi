"""Read-only source context for citations already attached to an owned message."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import or_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from rag_assistant.db.models import (
    Conversation,
    Document,
    DocumentVersion,
    Message,
    MessageRole,
    MessageStatus,
    ParagraphKind,
    SourceParagraph,
)

CONTEXT_NEIGHBORS = 12


@dataclass(frozen=True)
class CitationContext:
    heading: str | None
    paragraphs: tuple[SourceParagraph, ...]
    truncated: bool


async def read_citation_context(
    session: AsyncSession,
    *,
    owner_id: UUID,
    conversation_id: UUID,
    message_id: UUID,
    source_paragraph_id: UUID,
) -> CitationContext | None:
    """Resolve a saved citation, then load bounded neighbors within its source section."""
    snapshot = await session.scalar(
        select(Message.citation_snapshot)
        .join(Conversation, Conversation.id == Message.conversation_id)
        .where(
            Conversation.id == conversation_id,
            Conversation.owner_id == owner_id,
            Conversation.deleted_at.is_(None),
            Message.id == message_id,
            Message.role == MessageRole.ASSISTANT,
            Message.status == MessageStatus.COMPLETED,
        )
    )
    citation = next(
        (
            item
            for item in snapshot or []
            if item.get("source_paragraph_id") == str(source_paragraph_id)
        ),
        None,
    )
    if citation is None:
        return None
    try:
        version_id = UUID(citation["document_version_id"])
        document_id = UUID(citation["document_id"])
    except (KeyError, ValueError, TypeError):
        return None

    # A saved citation is not a grant to read arbitrary documents or newer versions.
    source = await session.scalar(
        select(SourceParagraph)
        .join(DocumentVersion, DocumentVersion.id == SourceParagraph.document_version_id)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(
            SourceParagraph.id == source_paragraph_id,
            SourceParagraph.document_version_id == version_id,
            Document.id == document_id,
            Document.deleted_at.is_(None),
            DocumentVersion.activated_at.is_not(None),
            or_(Document.owner_id == owner_id, Document.is_shared.is_(True)),
        )
    )
    if source is None or (
        source.original_text != citation.get("original_text")
        or source.page_number != citation.get("page_number")
        or source.paragraph_number != citation.get("paragraph_number")
    ):
        return None

    position = tuple_(SourceParagraph.page_number, SourceParagraph.paragraph_number)
    target = (source.page_number, source.paragraph_number)
    same_version = SourceParagraph.document_version_id == version_id
    heading = await session.scalar(
        select(SourceParagraph)
        .where(same_version, SourceParagraph.kind == ParagraphKind.HEADING, position <= target)
        .order_by(position.desc())
        .limit(1)
    )
    # A cited heading can be followed by a subheading before the actual body starts.
    first_body = await session.scalar(
        select(SourceParagraph)
        .where(same_version, SourceParagraph.kind != ParagraphKind.HEADING, position >= target)
        .order_by(position)
        .limit(1)
    )
    next_heading = (
        await session.scalar(
            select(SourceParagraph)
            .where(
                same_version,
                SourceParagraph.kind == ParagraphKind.HEADING,
                position > (first_body.page_number, first_body.paragraph_number),
            )
            .order_by(position)
            .limit(1)
        )
        if first_body is not None
        else None
    )
    section = select(SourceParagraph).where(same_version)
    if heading is not None:
        section = section.where(position >= (heading.page_number, heading.paragraph_number))
    if next_heading is not None:
        section = section.where(
            position < (next_heading.page_number, next_heading.paragraph_number)
        )
    before = list(
        await session.scalars(
            section.where(position < target).order_by(position.desc()).limit(CONTEXT_NEIGHBORS + 1)
        )
    )
    after = list(
        await session.scalars(
            section.where(position > target).order_by(position).limit(CONTEXT_NEIGHBORS + 1)
        )
    )
    return CitationContext(
        heading=heading.original_text if heading is not None else None,
        paragraphs=(*reversed(before[:CONTEXT_NEIGHBORS]), source, *after[:CONTEXT_NEIGHBORS]),
        truncated=len(before) > CONTEXT_NEIGHBORS or len(after) > CONTEXT_NEIGHBORS,
    )
