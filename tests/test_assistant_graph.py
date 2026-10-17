from __future__ import annotations

from collections.abc import Mapping
from typing import cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker

from rag_assistant.assistant import (
    AnswerCandidate,
    ConversationContext,
    ConversationHistoryMessage,
    RAGDependencies,
    RAGGraphInput,
    RAGGraphOutput,
    RAGWorkflow,
    SqlAlchemyWorkflowPersistence,
    create_rag_workflow,
)
from rag_assistant.assistant.errors import ConversationAuthorizationError
from rag_assistant.assistant.models import ExecutionRecord
from rag_assistant.db.models import AnswerOutcome, MessageRole, MessageStatus, ParagraphKind
from rag_assistant.db.repositories import ConversationRepository, UserRepository
from rag_assistant.retrieval import CitationSource, EvidenceChunk, RetrievalRequest, RetrievalResult


def _evidence(paragraph_id: UUID | None = None) -> RetrievalResult:
    source_id = paragraph_id or uuid4()
    return RetrievalResult(
        normalized_query="پرسش",
        embedding_model="openai/text-embedding-3-large",
        embedding_dimensions=1024,
        evidence=(
            EvidenceChunk(
                chunk_id=uuid4(),
                document_id=uuid4(),
                document_version_id=uuid4(),
                document_title="راهنما",
                chunk_ordinal=1,
                normalized_text="شواهد بازیابی‌شده",
                token_count=4,
                cosine_similarity=0.9,
                citation_sources=(
                    CitationSource(
                        source_paragraph_id=source_id,
                        document_id=uuid4(),
                        document_version_id=uuid4(),
                        document_title="راهنما",
                        page_number=1,
                        paragraph_number=1,
                        kind=ParagraphKind.PROSE,
                        original_text="متن اصلی منبع",
                    ),
                ),
            ),
        ),
    )


class FakePersistence:
    def __init__(
        self,
        *,
        context: ConversationContext | None = None,
        authorized: bool = True,
    ) -> None:
        self.context = context or ConversationContext(history=())
        self.authorized = authorized
        self.history_limits: list[int] = []
        self.created_questions: list[str] = []
        self.completed_answers: list[AnswerCandidate] = []
        self.failed: list[tuple[str, str]] = []
        self._executions: list[ExecutionRecord] = []

    async def load_context(
        self, *, user_id: UUID, conversation_id: UUID, history_limit: int
    ) -> ConversationContext:
        del user_id, conversation_id
        self.history_limits.append(history_limit)
        if not self.authorized:
            raise ConversationAuthorizationError("not authorized")
        return ConversationContext(history=self.context.history[-history_limit:])

    async def create_execution(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        question: str,
        idempotency_key: str,
    ) -> ExecutionRecord:
        del user_id, conversation_id, idempotency_key
        self.created_questions.append(question)
        execution = ExecutionRecord(user_message_id=uuid4(), assistant_message_id=uuid4())
        self._executions.append(execution)
        return execution

    async def renew_execution_lease(
        self, *, user_id: UUID, conversation_id: UUID, idempotency_key: str
    ) -> None:
        del user_id, conversation_id, idempotency_key

    async def complete_execution(
        self, *, user_id: UUID, assistant_message_id: UUID, answer: AnswerCandidate
    ) -> RAGGraphOutput:
        del user_id
        self.completed_answers.append(answer)
        return RAGGraphOutput(
            assistant_message_id=assistant_message_id,
            status=MessageStatus.COMPLETED,
            outcome=answer.outcome,
            content=answer.content,
            citation_snapshot=tuple(
                {"source_paragraph_id": str(paragraph_id)}
                for paragraph_id in answer.cited_paragraph_ids
            ),
        )

    async def fail_execution(
        self,
        *,
        user_id: UUID,
        assistant_message_id: UUID,
        error_code: str,
        error_message: str,
    ) -> RAGGraphOutput:
        del user_id
        self.failed.append((error_code, error_message))
        return RAGGraphOutput(
            assistant_message_id=assistant_message_id,
            status=MessageStatus.FAILED,
            error_code=error_code,
            error_message=error_message,
        )


class FakeRetriever:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.requests: list[RetrievalRequest] = []

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.requests.append(request)
        return self.result


class FakeGenerator:
    def __init__(self, answer: AnswerCandidate | Exception) -> None:
        self.answer = answer
        self.calls: list[tuple[str, tuple[ConversationHistoryMessage, ...], RetrievalResult]] = []

    async def generate(
        self,
        *,
        question: str,
        history: tuple[ConversationHistoryMessage, ...],
        evidence: RetrievalResult,
    ) -> AnswerCandidate:
        self.calls.append((question, history, evidence))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


class FakeValidator:
    def __init__(self, answer: AnswerCandidate | Exception | None = None) -> None:
        self.answer = answer
        self.calls: list[AnswerCandidate] = []

    async def validate(
        self,
        *,
        question: str,
        evidence: RetrievalResult,
        candidate: AnswerCandidate,
    ) -> AnswerCandidate:
        del question, evidence
        self.calls.append(candidate)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer or candidate


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, str | int]]] = []

    async def record_node(self, *, node_name: str, metadata: Mapping[str, str | int]) -> None:
        self.events.append((node_name, dict(metadata)))


def _workflow(
    *,
    persistence: FakePersistence,
    retriever: FakeRetriever,
    generator: FakeGenerator,
    validator: FakeValidator,
    observer: RecordingObserver | None = None,
) -> RAGWorkflow:
    return create_rag_workflow(
        RAGDependencies(
            persistence=persistence,
            retriever=retriever,
            generator=generator,
            validator=validator,
            observer=observer or RecordingObserver(),
        )
    )


async def test_graph_generates_validated_evidence_grounded_answer() -> None:
    source_id = uuid4()
    history = tuple(
        ConversationHistoryMessage(role=MessageRole.USER, content=f"پیام {index}")
        for index in range(20)
    )
    persistence = FakePersistence(context=ConversationContext(history=history))
    retriever = FakeRetriever(_evidence(source_id))
    generator = FakeGenerator(
        AnswerCandidate(
            content="پاسخ مستند",
            outcome=AnswerOutcome.FULLY_ANSWERED,
            cited_paragraph_ids=(source_id,),
        )
    )
    validator = FakeValidator()
    observer = RecordingObserver()

    output = await _workflow(
        persistence=persistence,
        retriever=retriever,
        generator=generator,
        validator=validator,
        observer=observer,
    ).ainvoke(RAGGraphInput(user_id=uuid4(), conversation_id=uuid4(), question="پرسش"))

    assert output.status is MessageStatus.COMPLETED
    assert output.outcome is AnswerOutcome.FULLY_ANSWERED
    assert persistence.history_limits == [12]
    assert persistence.created_questions == ["پرسش"]
    assert len(generator.calls) == 1
    assert len(generator.calls[0][1]) == 12
    assert len(validator.calls) == 1
    assert persistence.failed == []
    assert [name for name, _ in observer.events] == [
        "load_conversation",
        "create_execution",
        "prepare_question",
        "retrieve_evidence",
        "assess_evidence",
        "generate_answer",
        "validate_answer",
        "persist_completed",
    ]
    assert all("conversation_id" in metadata for _, metadata in observer.events)
    assert all("پرسش" not in str(metadata) for _, metadata in observer.events)


async def test_graph_abstains_and_persists_no_citations_when_retrieval_is_empty() -> None:
    persistence = FakePersistence()
    generator = FakeGenerator(RuntimeError("must not run"))
    validator = FakeValidator(RuntimeError("must not run"))

    output = await _workflow(
        persistence=persistence,
        retriever=FakeRetriever(
            RetrievalResult(
                normalized_query="پرسش",
                embedding_model="openai/text-embedding-3-large",
                embedding_dimensions=1024,
                evidence=(),
            )
        ),
        generator=generator,
        validator=validator,
    ).ainvoke(RAGGraphInput(user_id=uuid4(), conversation_id=uuid4(), question="پرسش"))

    assert output.status is MessageStatus.COMPLETED
    assert output.outcome is AnswerOutcome.ABSTAINED
    assert output.citation_snapshot == ()
    assert persistence.completed_answers[0].cited_paragraph_ids == ()
    assert generator.calls == []
    assert validator.calls == []


async def test_graph_fails_closed_without_creating_messages_for_unauthorized_conversation() -> None:
    persistence = FakePersistence(authorized=False)
    retriever = FakeRetriever(_evidence())

    with pytest.raises(ConversationAuthorizationError):
        await _workflow(
            persistence=persistence,
            retriever=retriever,
            generator=FakeGenerator(RuntimeError("must not run")),
            validator=FakeValidator(),
        ).ainvoke(RAGGraphInput(user_id=uuid4(), conversation_id=uuid4(), question="پرسش"))

    assert persistence.created_questions == []
    assert persistence.failed == []
    assert retriever.requests == []


async def test_graph_marks_dependency_failure_without_leaking_an_unvalidated_answer() -> None:
    persistence = FakePersistence()

    output = await _workflow(
        persistence=persistence,
        retriever=FakeRetriever(_evidence()),
        generator=FakeGenerator(RuntimeError("provider unavailable")),
        validator=FakeValidator(),
    ).ainvoke(RAGGraphInput(user_id=uuid4(), conversation_id=uuid4(), question="پرسش"))

    assert output.status is MessageStatus.FAILED
    assert output.content is None
    assert output.outcome is None
    assert output.citation_snapshot == ()
    assert persistence.completed_answers == []
    assert persistence.failed == [
        ("generation_failed", "Answer generation could not be completed.")
    ]


async def test_graph_rejects_citations_that_were_not_supplied_by_retrieval() -> None:
    source_id = uuid4()
    persistence = FakePersistence()
    generated = AnswerCandidate(
        content="پاسخ",
        outcome=AnswerOutcome.FULLY_ANSWERED,
        cited_paragraph_ids=(source_id,),
    )
    invalid = generated.model_copy(update={"cited_paragraph_ids": (uuid4(),)})

    output = await _workflow(
        persistence=persistence,
        retriever=FakeRetriever(_evidence(source_id)),
        generator=FakeGenerator(generated),
        validator=FakeValidator(invalid),
    ).ainvoke(RAGGraphInput(user_id=uuid4(), conversation_id=uuid4(), question="پرسش"))

    assert output.status is MessageStatus.FAILED
    assert persistence.completed_answers == []
    assert persistence.failed == [
        ("validation_failed", "The generated answer could not be validated.")
    ]


async def test_graph_state_does_not_leak_between_invocations() -> None:
    first_source = uuid4()
    second_source = uuid4()
    persistence = FakePersistence()
    retriever = FakeRetriever(_evidence(first_source))
    generator = FakeGenerator(
        AnswerCandidate(
            content="پاسخ اول",
            outcome=AnswerOutcome.FULLY_ANSWERED,
            cited_paragraph_ids=(first_source,),
        )
    )
    workflow = _workflow(
        persistence=persistence,
        retriever=retriever,
        generator=generator,
        validator=FakeValidator(),
    )

    first = await workflow.ainvoke(
        RAGGraphInput(user_id=uuid4(), conversation_id=uuid4(), question="پرسش اول")
    )
    retriever.result = _evidence(second_source)
    generator.answer = AnswerCandidate(
        content="پاسخ دوم\n\nبخش پشتیبانی‌نشده: مقدار ذکرنشده",
        outcome=AnswerOutcome.PARTIALLY_ANSWERED,
        cited_paragraph_ids=(second_source,),
        unsupported_content="مقدار ذکرنشده",
    )
    second = await workflow.ainvoke(
        RAGGraphInput(user_id=uuid4(), conversation_id=uuid4(), question="پرسش دوم")
    )

    assert first.assistant_message_id != second.assistant_message_id
    assert [request.question for request in retriever.requests] == ["پرسش اول", "پرسش دوم"]
    assert [answer.content for answer in persistence.completed_answers] == [
        "پاسخ اول",
        "پاسخ دوم\n\nبخش پشتیبانی‌نشده: مقدار ذکرنشده",
    ]


async def test_sqlalchemy_persistence_commits_request_and_terminal_messages(
    db_session: AsyncSession,
) -> None:
    user = await UserRepository(db_session).create(
        email="workflow-persistence@example.com",
        password_hash="test-password-hash",
    )
    conversations = ConversationRepository(db_session)
    conversation = await conversations.create(owner_id=user.id)
    for index in range(3):
        await conversations.append_user_message(
            owner_id=user.id,
            conversation_id=conversation.id,
            content=f"پیام {index}",
        )

    session_factory = async_sessionmaker(
        bind=cast(AsyncConnection, db_session.bind),
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    persistence = SqlAlchemyWorkflowPersistence(session_factory)
    context = await persistence.load_context(
        user_id=user.id,
        conversation_id=conversation.id,
        history_limit=2,
    )
    execution = await persistence.create_execution(
        user_id=user.id,
        conversation_id=conversation.id,
        question="پرسش جدید",
        idempotency_key="history-context",
    )
    output = await persistence.complete_execution(
        user_id=user.id,
        assistant_message_id=execution.assistant_message_id,
        answer=AnswerCandidate(
            content="شواهد کافی موجود نیست.",
            outcome=AnswerOutcome.ABSTAINED,
        ),
    )

    assert [message.content for message in context.history] == ["پیام 1", "پیام 2"]
    assert output.status is MessageStatus.COMPLETED
    assert output.outcome is AnswerOutcome.ABSTAINED
    persisted = await conversations.get_owned(
        owner_id=user.id,
        conversation_id=conversation.id,
        include_messages=True,
    )
    assert persisted is not None
    assert [message.status for message in persisted.messages[-2:]] == [
        MessageStatus.COMPLETED,
        MessageStatus.COMPLETED,
    ]
    assert persisted.messages[-2].content == "پرسش جدید"
    assert persisted.messages[-1].citation_snapshot == []
