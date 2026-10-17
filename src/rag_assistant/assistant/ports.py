"""Dependency ports for the RAG graph and small production adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from langfuse import get_client

from rag_assistant.assistant.models import (
    AnswerCandidate,
    ConversationContext,
    ConversationHistoryMessage,
    ExecutionRecord,
    RAGGraphOutput,
)
from rag_assistant.retrieval import RetrievalRequest, RetrievalResult


class EvidenceRetriever(Protocol):
    """Retrieve authorization-scoped evidence for one prepared question."""

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...


class AnswerGenerator(Protocol):
    """Produce a candidate answer from history and retrieved evidence only."""

    async def generate(
        self,
        *,
        question: str,
        history: tuple[ConversationHistoryMessage, ...],
        evidence: RetrievalResult,
    ) -> AnswerCandidate: ...


class AnswerValidator(Protocol):
    """Validate a candidate answer against the evidence supplied to generation."""

    async def validate(
        self,
        *,
        question: str,
        evidence: RetrievalResult,
        candidate: AnswerCandidate,
    ) -> AnswerCandidate: ...


class WorkflowPersistence(Protocol):
    """Persistence boundary with separately committed request and terminal records."""

    async def load_context(
        self, *, user_id: UUID, conversation_id: UUID, history_limit: int
    ) -> ConversationContext: ...

    async def create_execution(
        self, *, user_id: UUID, conversation_id: UUID, question: str
    ) -> ExecutionRecord: ...

    async def complete_execution(
        self, *, user_id: UUID, assistant_message_id: UUID, answer: AnswerCandidate
    ) -> RAGGraphOutput: ...

    async def fail_execution(
        self,
        *,
        user_id: UUID,
        assistant_message_id: UUID,
        error_code: str,
        error_message: str,
    ) -> RAGGraphOutput: ...


class NodeObserver(Protocol):
    """Record safe, bounded metadata for one graph-node execution."""

    async def record_node(self, *, node_name: str, metadata: Mapping[str, str | int]) -> None: ...


class LangfuseNodeObserver:
    """Attach safe node metadata to the active Langfuse observation."""

    async def record_node(self, *, node_name: str, metadata: Mapping[str, str | int]) -> None:
        get_client().update_current_span(name=node_name, metadata=dict(metadata))


class NoopNodeObserver:
    """Allow deterministic graph construction before observability is configured."""

    async def record_node(self, *, node_name: str, metadata: Mapping[str, str | int]) -> None:
        del node_name, metadata


@dataclass(frozen=True, slots=True)
class RAGDependencies:
    """Per-workflow dependencies; request data never lives on this object."""

    retriever: EvidenceRetriever
    generator: AnswerGenerator
    validator: AnswerValidator
    persistence: WorkflowPersistence
    observer: NodeObserver = field(default_factory=LangfuseNodeObserver)
