from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_assistant.db.base import (
    Base,
    CreatedAtMixin,
    UpdatedAtMixin,
    UUIDPrimaryKeyMixin,
)
from rag_assistant.db.models.enums import AnswerOutcome, MessageRole, MessageStatus

if TYPE_CHECKING:
    from rag_assistant.db.models.document import Document
    from rag_assistant.db.models.user import User


def _enum_values(
    enum_class: type[AnswerOutcome] | type[MessageRole] | type[MessageStatus],
) -> list[str]:
    return [member.value for member in enum_class]


class Conversation(UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, Base):
    __tablename__ = "conversations"
    __table_args__ = (
        CheckConstraint("title IS NULL OR length(btrim(title)) > 0", name="title_not_blank"),
        CheckConstraint(
            "jsonb_typeof(conversation_metadata) = 'object'",
            name="conversation_metadata_is_object",
        ),
        Index("ix_conversations_owner_id_updated_at", "owner_id", "updated_at"),
    )

    owner_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    conversation_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )

    owner: Mapped[User] = relationship(back_populates="conversations")
    document_links: Mapped[list[ConversationDocument]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class ConversationDocument(CreatedAtMixin, Base):
    __tablename__ = "conversation_documents"

    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )

    conversation: Mapped[Conversation] = relationship(back_populates="document_links")
    document: Mapped[Document] = relationship(back_populates="conversation_links")


class Message(UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("jsonb_typeof(model_usage) = 'object'", name="model_usage_is_object"),
        CheckConstraint(
            "jsonb_typeof(provider_metadata) = 'object'", name="provider_metadata_is_object"
        ),
        CheckConstraint(
            "jsonb_typeof(citation_snapshot) = 'array'", name="citation_snapshot_is_array"
        ),
        CheckConstraint(
            "(role = 'user' AND status = 'completed' AND content IS NOT NULL "
            "AND length(content) > 0 AND answer_outcome IS NULL AND error_code IS NULL "
            "AND error_message IS NULL AND jsonb_array_length(citation_snapshot) = 0) OR "
            "(role = 'assistant' AND status = 'pending' AND content IS NULL "
            "AND answer_outcome IS NULL AND error_code IS NULL AND error_message IS NULL "
            "AND jsonb_array_length(citation_snapshot) = 0) OR "
            "(role = 'assistant' AND status = 'failed' AND content IS NULL "
            "AND answer_outcome IS NULL AND error_code IS NOT NULL AND error_message IS NOT NULL "
            "AND jsonb_array_length(citation_snapshot) = 0) OR "
            "(role = 'assistant' AND status = 'completed' AND content IS NOT NULL "
            "AND length(content) > 0 AND answer_outcome IS NOT NULL AND error_code IS NULL "
            "AND error_message IS NULL AND ((answer_outcome = 'abstained' "
            "AND jsonb_array_length(citation_snapshot) = 0) OR "
            "(answer_outcome IN ('fully_answered', 'partially_answered') "
            "AND jsonb_array_length(citation_snapshot) > 0)))",
            name="message_state_valid",
        ),
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
        Index(
            "uq_messages_user_idempotency_key",
            "conversation_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("role = 'user' AND idempotency_key IS NOT NULL"),
        ),
    )

    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(
        Enum(
            MessageRole,
            values_callable=_enum_values,
            native_enum=False,
            create_constraint=False,
            length=16,
        ),
        nullable=False,
    )
    status: Mapped[MessageStatus] = mapped_column(
        Enum(
            MessageStatus,
            values_callable=_enum_values,
            native_enum=False,
            create_constraint=False,
            length=16,
        ),
        nullable=False,
    )
    answer_outcome: Mapped[AnswerOutcome | None] = mapped_column(
        Enum(
            AnswerOutcome,
            values_callable=_enum_values,
            native_enum=False,
            create_constraint=False,
            length=24,
        ),
        nullable=True,
    )
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_snapshot: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    model_usage: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    provider_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
