"""Typed values exchanged by the RAG orchestration layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_assistant.db.models import AnswerOutcome, MessageRole, MessageStatus


class RAGGraphInput(BaseModel):
    """One authenticated request to answer a question in a conversation."""

    model_config = ConfigDict(frozen=True)

    user_id: UUID
    conversation_id: UUID
    question: str = Field(max_length=10_000)
    idempotency_key: str = Field(default_factory=lambda: uuid4().hex, min_length=1, max_length=128)

    @field_validator("question")
    @classmethod
    def require_meaningful_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank")
        return value


class ConversationHistoryMessage(BaseModel):
    """A prior completed message usable only as conversational context."""

    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: str = Field(min_length=1)


class ConversationContext(BaseModel):
    """Authorized conversation context loaded before request records are created."""

    model_config = ConfigDict(frozen=True)

    history: tuple[ConversationHistoryMessage, ...]


class ExecutionRecord(BaseModel):
    """The durable user and pending-assistant records for one graph invocation."""

    model_config = ConfigDict(frozen=True)

    user_message_id: UUID
    assistant_message_id: UUID
    existing_output: dict[str, Any] | None = None


class AnswerCandidate(BaseModel):
    """An answer proposed or approved by an injected generation component."""

    model_config = ConfigDict(frozen=True)

    content: str
    outcome: AnswerOutcome
    cited_chunk_ids: tuple[UUID, ...] = ()
    cited_paragraph_ids: tuple[UUID, ...] = ()
    unsupported_content: str | None = None
    model_usage: dict[str, Any] = Field(default_factory=dict)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class RAGGraphOutput(BaseModel):
    """The persisted terminal assistant execution returned by the graph."""

    model_config = ConfigDict(frozen=True)

    assistant_message_id: UUID
    status: MessageStatus
    outcome: AnswerOutcome | None = None
    content: str | None = None
    citation_snapshot: tuple[dict[str, Any], ...] = ()
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None


class WorkflowFailure(BaseModel):
    """A bounded, non-sensitive failure description carried to final persistence."""

    model_config = ConfigDict(frozen=True)

    node_name: str
    error_code: str
    safe_message: str
