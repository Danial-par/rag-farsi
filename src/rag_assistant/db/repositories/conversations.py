from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from rag_assistant.db.base import utc_now
from rag_assistant.db.models import (
    AnswerOutcome,
    Conversation,
    ConversationDocument,
    Document,
    DocumentVersion,
    Message,
    MessageRole,
    MessageStatus,
    SourceParagraph,
)
from rag_assistant.db.repositories.errors import (
    InvalidPersistenceStateError,
    RecordNotFoundError,
)


class ConversationRepository:
    """Owned conversation operations, including immutable assistant outcomes."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        owner_id: UUID,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Conversation:
        conversation = Conversation(
            owner_id=owner_id,
            title=title,
            conversation_metadata=metadata or {},
        )
        self._session.add(conversation)
        await self._session.flush()

        default_document_ids = await self._session.scalars(
            select(Document.id).where(
                Document.is_shared.is_(True),
                Document.is_default.is_(True),
                Document.active_version_id.is_not(None),
                Document.deleted_at.is_(None),
            )
        )
        self._session.add_all(
            ConversationDocument(
                conversation_id=conversation.id,
                document_id=document_id,
            )
            for document_id in default_document_ids
        )
        await self._session.flush()
        return conversation

    async def get_owned(
        self, *, owner_id: UUID, conversation_id: UUID, include_messages: bool = False
    ) -> Conversation | None:
        statement = select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.owner_id == owner_id,
            Conversation.deleted_at.is_(None),
        )
        if include_messages:
            statement = statement.options(selectinload(Conversation.messages))
        return await self._session.scalar(statement)

    async def add_document(
        self, *, owner_id: UUID, conversation_id: UUID, document_id: UUID
    ) -> ConversationDocument:
        conversation = await self._locked_owned_conversation(owner_id, conversation_id)
        document = await self._session.scalar(
            select(Document).where(
                Document.id == document_id,
                ((Document.owner_id == owner_id) | Document.is_shared.is_(True)),
                Document.deleted_at.is_(None),
            )
        )
        if document is None:
            raise RecordNotFoundError("Accessible document not found")
        existing = await self._session.get(
            ConversationDocument,
            {"conversation_id": conversation.id, "document_id": document.id},
        )
        if existing is not None:
            return existing
        link = ConversationDocument(conversation_id=conversation.id, document_id=document.id)
        self._session.add(link)
        await self._session.flush()
        return link

    async def list_active_documents(
        self, *, owner_id: UUID, conversation_id: UUID
    ) -> list[Document]:
        documents = await self._session.scalars(
            select(Document)
            .join(ConversationDocument, ConversationDocument.document_id == Document.id)
            .join(Conversation, Conversation.id == ConversationDocument.conversation_id)
            .where(
                Conversation.id == conversation_id,
                Conversation.owner_id == owner_id,
                Conversation.deleted_at.is_(None),
                Document.deleted_at.is_(None),
                Document.active_version_id.is_not(None),
                ((Document.owner_id == owner_id) | Document.is_shared.is_(True)),
            )
            .order_by(Document.title, Document.id)
        )
        return list(documents)

    async def append_user_message(
        self, *, owner_id: UUID, conversation_id: UUID, content: str
    ) -> Message:
        if not content.strip():
            raise InvalidPersistenceStateError("User message content cannot be blank")
        conversation = await self._locked_owned_conversation(owner_id, conversation_id)
        message = Message(
            conversation_id=conversation.id,
            role=MessageRole.USER,
            status=MessageStatus.COMPLETED,
            content=content,
            citation_snapshot=[],
            model_usage={},
            provider_metadata={},
        )
        conversation.updated_at = utc_now()
        self._session.add(message)
        await self._session.flush()
        return message

    async def create_pending_assistant_message(
        self, *, owner_id: UUID, conversation_id: UUID
    ) -> Message:
        conversation = await self._locked_owned_conversation(owner_id, conversation_id)
        message = Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            status=MessageStatus.PENDING,
            content=None,
            citation_snapshot=[],
            model_usage={},
            provider_metadata={},
        )
        conversation.updated_at = utc_now()
        self._session.add(message)
        await self._session.flush()
        return message

    async def complete_assistant_message(
        self,
        *,
        owner_id: UUID,
        message_id: UUID,
        content: str,
        outcome: AnswerOutcome,
        cited_paragraph_ids: Sequence[UUID],
        model_usage: dict[str, Any] | None = None,
        provider_metadata: dict[str, Any] | None = None,
    ) -> Message:
        if not content.strip():
            raise InvalidPersistenceStateError("Completed assistant content cannot be blank")
        message = await self._locked_pending_assistant_message(owner_id, message_id)
        unique_citation_ids = list(dict.fromkeys(cited_paragraph_ids))
        if len(unique_citation_ids) != len(cited_paragraph_ids):
            raise InvalidPersistenceStateError("Citation identifiers must be unique")
        if outcome is AnswerOutcome.ABSTAINED and unique_citation_ids:
            raise InvalidPersistenceStateError("Abstained messages cannot contain citations")
        if outcome is not AnswerOutcome.ABSTAINED and not unique_citation_ids:
            raise InvalidPersistenceStateError("Supported answers require at least one citation")

        citation_snapshot = await self._build_citation_snapshot(
            conversation_id=message.conversation_id,
            paragraph_ids=unique_citation_ids,
        )
        message.status = MessageStatus.COMPLETED
        message.answer_outcome = outcome
        message.content = content
        message.citation_snapshot = citation_snapshot
        message.model_usage = model_usage or {}
        message.provider_metadata = provider_metadata or {}
        await self._session.flush()
        return message

    async def fail_assistant_message(
        self,
        *,
        owner_id: UUID,
        message_id: UUID,
        error_code: str,
        error_message: str,
        provider_metadata: dict[str, Any] | None = None,
    ) -> Message:
        if not error_code or not error_message:
            raise InvalidPersistenceStateError("Failed messages require an error code and message")
        message = await self._locked_pending_assistant_message(owner_id, message_id)
        message.status = MessageStatus.FAILED
        message.error_code = error_code
        message.error_message = error_message
        message.provider_metadata = provider_metadata or {}
        await self._session.flush()
        return message

    async def _locked_owned_conversation(
        self, owner_id: UUID, conversation_id: UUID
    ) -> Conversation:
        conversation = await self._session.scalar(
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.owner_id == owner_id,
                Conversation.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if conversation is None:
            raise RecordNotFoundError("Owned conversation not found")
        return conversation

    async def _locked_pending_assistant_message(self, owner_id: UUID, message_id: UUID) -> Message:
        message = await self._session.scalar(
            select(Message)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(
                Message.id == message_id,
                Message.role == MessageRole.ASSISTANT,
                Message.status == MessageStatus.PENDING,
                Conversation.owner_id == owner_id,
                Conversation.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if message is None:
            raise RecordNotFoundError("Pending assistant message not found")
        return message

    async def _build_citation_snapshot(
        self, *, conversation_id: UUID, paragraph_ids: Sequence[UUID]
    ) -> list[dict[str, Any]]:
        if not paragraph_ids:
            return []
        rows = await self._session.execute(
            select(SourceParagraph, DocumentVersion, Document)
            .join(
                DocumentVersion,
                DocumentVersion.id == SourceParagraph.document_version_id,
            )
            .join(Document, Document.id == DocumentVersion.document_id)
            .join(
                ConversationDocument,
                ConversationDocument.document_id == Document.id,
            )
            .where(
                ConversationDocument.conversation_id == conversation_id,
                SourceParagraph.id.in_(paragraph_ids),
                Document.deleted_at.is_(None),
            )
        )
        rows_by_id: dict[UUID, tuple[SourceParagraph, DocumentVersion, Document]] = {}
        for paragraph, version, document in rows:
            rows_by_id[paragraph.id] = (paragraph, version, document)
        if set(paragraph_ids) != rows_by_id.keys():
            raise InvalidPersistenceStateError(
                "Every citation must resolve to a source paragraph in the conversation scope"
            )
        return [
            {
                "source_paragraph_id": str(paragraph.id),
                "document_id": str(document.id),
                "document_version_id": str(version.id),
                "document_title": document.title,
                "page_number": paragraph.page_number,
                "paragraph_number": paragraph.paragraph_number,
                "original_text": paragraph.original_text,
            }
            for paragraph_id in paragraph_ids
            for paragraph, version, document in [rows_by_id[paragraph_id]]
        ]
