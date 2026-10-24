"""SQLAlchemy implementation of the graph persistence boundary."""

from __future__ import annotations

from uuid import UUID

from rag_assistant.assistant.errors import ConversationAuthorizationError
from rag_assistant.assistant.models import (
    AnswerCandidate,
    ConversationContext,
    ConversationHistoryMessage,
    ExecutionRecord,
    RAGGraphOutput,
)
from rag_assistant.db.models import MessageStatus
from rag_assistant.db.repositories import ConversationRepository
from rag_assistant.db.session import AsyncSessionFactory


class SqlAlchemyWorkflowPersistence:
    """Persist a workflow in short, independent database transactions."""

    def __init__(self, session_factory: AsyncSessionFactory) -> None:
        self._session_factory = session_factory

    async def load_context(
        self, *, user_id: UUID, conversation_id: UUID, history_limit: int
    ) -> ConversationContext:
        async with self._session_factory() as session:
            conversation = await ConversationRepository(session).get_owned(
                owner_id=user_id,
                conversation_id=conversation_id,
                include_messages=True,
            )
            if conversation is None:
                raise ConversationAuthorizationError("Conversation access could not be established")
            history = tuple(
                ConversationHistoryMessage(role=message.role, content=message.content)
                for message in conversation.messages
                if message.status is MessageStatus.COMPLETED and message.content is not None
            )
            return ConversationContext(history=history[-history_limit:])

    async def create_execution(
        self, *, user_id: UUID, conversation_id: UUID, question: str, idempotency_key: str
    ) -> ExecutionRecord:
        async with self._session_factory.begin() as session:
            repository = ConversationRepository(session)
            (
                user_message_id,
                assistant_message_id,
                existing_output,
            ) = await repository.create_execution(
                owner_id=user_id,
                conversation_id=conversation_id,
                question=question,
                idempotency_key=idempotency_key,
            )
            return ExecutionRecord(
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id,
                existing_output=existing_output,
            )

    async def renew_execution_lease(
        self, *, user_id: UUID, conversation_id: UUID, idempotency_key: str
    ) -> None:
        """Keep an actively running graph from being classified as abandoned."""
        async with self._session_factory.begin() as session:
            await ConversationRepository(session).renew_execution_lease(
                owner_id=user_id,
                conversation_id=conversation_id,
                idempotency_key=idempotency_key,
            )

    async def complete_execution(
        self, *, user_id: UUID, assistant_message_id: UUID, answer: AnswerCandidate
    ) -> RAGGraphOutput:
        async with self._session_factory.begin() as session:
            message = await ConversationRepository(session).complete_assistant_message(
                owner_id=user_id,
                message_id=assistant_message_id,
                content=answer.content,
                outcome=answer.outcome,
                cited_chunk_ids=answer.cited_chunk_ids,
                model_usage=answer.model_usage,
                provider_metadata=answer.provider_metadata,
            )
            return RAGGraphOutput(
                assistant_message_id=message.id,
                status=message.status,
                outcome=message.answer_outcome,
                content=message.content,
                citation_snapshot=tuple(message.citation_snapshot),
                created_at=message.created_at,
            )

    async def fail_execution(
        self,
        *,
        user_id: UUID,
        assistant_message_id: UUID,
        error_code: str,
        error_message: str,
    ) -> RAGGraphOutput:
        async with self._session_factory.begin() as session:
            message = await ConversationRepository(session).fail_assistant_message(
                owner_id=user_id,
                message_id=assistant_message_id,
                error_code=error_code,
                error_message=error_message,
            )
            return RAGGraphOutput(
                assistant_message_id=message.id,
                status=message.status,
                error_code=message.error_code,
                error_message=message.error_message,
                created_at=message.created_at,
            )
