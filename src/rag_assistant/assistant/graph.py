"""The stateless async LangGraph workflow for one grounded answer."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from time import perf_counter
from typing import Any, NotRequired, TypedDict, cast
from uuid import UUID

import structlog
from langfuse import get_client, observe, propagate_attributes
from langgraph.graph import END, START, StateGraph

from rag_assistant.assistant.errors import ConversationAuthorizationError, WorkflowExecutionError
from rag_assistant.assistant.models import (
    AnswerCandidate,
    ConversationContext,
    ExecutionRecord,
    RAGGraphInput,
    RAGGraphOutput,
    WorkflowFailure,
)
from rag_assistant.assistant.ports import RAGDependencies
from rag_assistant.db.models import AnswerOutcome, MessageStatus
from rag_assistant.db.repositories.conversations import IdempotencyConflictError
from rag_assistant.retrieval import RetrievalRequest, RetrievalResult
from rag_assistant.retrieval.errors import RetrievalError

HISTORY_MESSAGE_LIMIT = 12


def _abstention_message(question: str) -> str:
    """Return a short citation-free abstention in Persian or English."""
    if any("\u0600" <= character <= "\u06ff" for character in question):
        return "برای پاسخ به این پرسش، شواهد کافی در اسناد انتخاب‌شده پیدا نشد."
    return "I could not find enough evidence in the selected documents to answer this question."


class RAGState(TypedDict):
    """Only request-scoped data moves between graph nodes."""

    user_id: UUID
    conversation_id: UUID
    question: str
    idempotency_key: str
    context: NotRequired[ConversationContext]
    execution: NotRequired[ExecutionRecord]
    prepared_question: NotRequired[str]
    evidence: NotRequired[RetrievalResult]
    evidence_sufficient: NotRequired[bool]
    candidate: NotRequired[AnswerCandidate]
    validated_answer: NotRequired[AnswerCandidate]
    failure: NotRequired[WorkflowFailure]
    output: NotRequired[RAGGraphOutput]


NodeResult = dict[str, object]
NodeAction = Callable[[], Awaitable[NodeResult]]


class RAGWorkflow:
    """Run a compiled graph without checkpoints or graph-level retries."""

    def __init__(self, compiled_graph: Any, dependencies: RAGDependencies) -> None:
        self._compiled_graph = compiled_graph
        self._persistence = dependencies.persistence
        self._heartbeat_seconds = dependencies.execution_heartbeat_seconds

    @observe(name="rag.request", capture_input=False, capture_output=False)
    async def ainvoke(self, graph_input: RAGGraphInput) -> RAGGraphOutput:
        """Execute one request and return only a terminal persisted result."""
        request_id = structlog.contextvars.get_contextvars().get("request_id")
        trace_metadata: dict[str, str] = {
            "conversation_id": str(graph_input.conversation_id),
        }
        if isinstance(request_id, str):
            trace_metadata["request_id"] = request_id
        client = get_client()
        client.update_current_span(input=graph_input.question)
        with propagate_attributes(metadata=trace_metadata):
            return await self._execute_graph(graph_input, client)

    async def _execute_graph(
        self,
        graph_input: RAGGraphInput,
        client: Any,
    ) -> RAGGraphOutput:
        heartbeat = asyncio.create_task(self._maintain_execution_lease(graph_input))
        try:
            final_state = cast(
                RAGState,
                await self._compiled_graph.ainvoke(
                    {
                        "user_id": graph_input.user_id,
                        "conversation_id": graph_input.conversation_id,
                        "question": graph_input.question,
                        "idempotency_key": graph_input.idempotency_key,
                    }
                ),
            )
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
        output = final_state.get("output")
        if output is not None:
            client.update_current_span(
                output=output.content,
                metadata={
                    "assistant_message_id": str(output.assistant_message_id),
                    "execution_status": output.status.value,
                    "answer_outcome": output.outcome.value if output.outcome else "none",
                    **({"error_code": output.error_code} if output.error_code else {}),
                },
            )
            return output

        failure = final_state.get("failure")
        if failure is not None and failure.error_code == "conversation_authorization_failed":
            client.update_current_span(metadata={"error_code": failure.error_code})
            raise ConversationAuthorizationError(failure.safe_message)
        if failure is not None:
            client.update_current_span(metadata={"error_code": failure.error_code})
        raise WorkflowExecutionError("The assistant execution did not reach a terminal state")

    async def _maintain_execution_lease(self, graph_input: RAGGraphInput) -> None:
        logger = structlog.stdlib.get_logger("rag_assistant.assistant.lease")
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            try:
                await self._persistence.renew_execution_lease(
                    user_id=graph_input.user_id,
                    conversation_id=graph_input.conversation_id,
                    idempotency_key=graph_input.idempotency_key,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await logger.awarning(
                    "assistant_execution_lease_renewal_failed",
                    error_type=type(exc).__name__,
                )


def create_rag_workflow(dependencies: RAGDependencies) -> RAGWorkflow:
    """Build the per-request graph around injected infrastructure dependencies."""

    async def record_node(
        state: RAGState,
        *,
        node_name: str,
        extra: dict[str, str | int] | None = None,
    ) -> None:
        metadata: dict[str, str | int] = {
            "conversation_id": str(state["conversation_id"]),
        }
        request_id = structlog.contextvars.get_contextvars().get("request_id")
        if isinstance(request_id, str):
            metadata["request_id"] = request_id
        execution = state.get("execution")
        if execution is not None:
            metadata["assistant_message_id"] = str(execution.assistant_message_id)
        if extra:
            metadata.update(extra)
        await dependencies.observer.record_node(node_name=node_name, metadata=metadata)

    async def guarded(
        state: RAGState,
        *,
        node_name: str,
        action: NodeAction,
    ) -> NodeResult:
        started = perf_counter()
        try:
            result = await action()
            get_client().update_current_span(
                metadata={"duration_ms": round((perf_counter() - started) * 1000, 2)}
            )
            return result
        except IdempotencyConflictError:
            raise
        except Exception as exc:
            failure = _classify_failure(node_name, exc)
            await structlog.stdlib.get_logger("rag_assistant.assistant.workflow").awarning(
                "rag_node_failed",
                node_name=node_name,
                error_type=type(exc).__name__,
                error_code=failure.error_code,
            )
            get_client().update_current_span(
                metadata={
                    "duration_ms": round((perf_counter() - started) * 1000, 2),
                    "error_code": failure.error_code,
                },
                level="ERROR",
                status_message=failure.error_code,
            )
            return {"failure": failure}

    @observe(name="rag.load_conversation", capture_input=False, capture_output=False)
    async def load_conversation(state: RAGState) -> NodeResult:
        async def action() -> NodeResult:
            context = await dependencies.persistence.load_context(
                user_id=state["user_id"],
                conversation_id=state["conversation_id"],
                history_limit=HISTORY_MESSAGE_LIMIT,
            )
            await record_node(
                state,
                node_name="load_conversation",
                extra={"history_message_count": len(context.history)},
            )
            return {"context": context}

        return await guarded(state, node_name="load_conversation", action=action)

    @observe(name="rag.create_execution", capture_input=False, capture_output=False)
    async def create_execution(state: RAGState) -> NodeResult:
        async def action() -> NodeResult:
            execution = await dependencies.persistence.create_execution(
                user_id=state["user_id"],
                conversation_id=state["conversation_id"],
                question=state["question"],
                idempotency_key=state["idempotency_key"],
            )
            result: NodeResult = {"execution": execution}
            if execution.existing_output is not None:
                result["output"] = RAGGraphOutput.model_validate(execution.existing_output)
            state_with_execution = {**state, "execution": execution}
            await record_node(
                cast(RAGState, state_with_execution),
                node_name="create_execution",
            )
            return result

        return await guarded(state, node_name="create_execution", action=action)

    @observe(name="rag.prepare_question", capture_input=False, capture_output=False)
    async def prepare_question(state: RAGState) -> NodeResult:
        async def action() -> NodeResult:
            prepared_question = state["question"]
            await record_node(
                state,
                node_name="prepare_question",
                extra={"question_character_count": len(prepared_question)},
            )
            return {"prepared_question": prepared_question}

        return await guarded(state, node_name="prepare_question", action=action)

    @observe(name="rag.retrieve_evidence", capture_input=False, capture_output=False)
    async def retrieve_evidence(state: RAGState) -> NodeResult:
        async def action() -> NodeResult:
            evidence = await dependencies.retriever.retrieve(
                RetrievalRequest(
                    user_id=state["user_id"],
                    conversation_id=state["conversation_id"],
                    question=state["prepared_question"],
                )
            )
            await record_node(
                state,
                node_name="retrieve_evidence",
                extra={"evidence_chunk_count": len(evidence.evidence)},
            )
            return {"evidence": evidence}

        return await guarded(state, node_name="retrieve_evidence", action=action)

    @observe(name="rag.assess_evidence", capture_input=False, capture_output=False)
    async def assess_evidence(state: RAGState) -> NodeResult:
        async def action() -> NodeResult:
            sufficient = bool(state["evidence"].evidence)
            await record_node(
                state,
                node_name="assess_evidence",
                extra={"evidence_sufficient": int(sufficient)},
            )
            return {"evidence_sufficient": sufficient}

        return await guarded(state, node_name="assess_evidence", action=action)

    @observe(name="rag.abstain", capture_input=False, capture_output=False)
    async def abstain(state: RAGState) -> NodeResult:
        async def action() -> NodeResult:
            candidate = AnswerCandidate(
                content=_abstention_message(state["question"]),
                outcome=AnswerOutcome.ABSTAINED,
            )
            await record_node(state, node_name="abstain")
            return {"candidate": candidate}

        return await guarded(state, node_name="abstain", action=action)

    @observe(name="rag.generate_answer", capture_input=False, capture_output=False)
    async def generate_answer(state: RAGState) -> NodeResult:
        async def action() -> NodeResult:
            candidate = await dependencies.generator.generate(
                question=state["prepared_question"],
                history=state["context"].history,
                evidence=state["evidence"],
            )
            usage_metadata = {
                key: value
                for key, value in candidate.model_usage.items()
                if key in {"input_tokens", "output_tokens", "total_tokens"}
                and isinstance(value, int)
                and value >= 0
            }
            await record_node(
                state,
                node_name="generate_answer",
                extra={
                    "candidate_citation_count": len(candidate.cited_chunk_ids),
                    **usage_metadata,
                },
            )
            return {"candidate": candidate}

        return await guarded(state, node_name="generate_answer", action=action)

    @observe(name="rag.validate_answer", capture_input=False, capture_output=False)
    async def validate_answer(state: RAGState) -> NodeResult:
        async def action() -> NodeResult:
            candidate = state["candidate"]
            if candidate.outcome is AnswerOutcome.ABSTAINED:
                validated_answer = candidate
            else:
                validated_answer = await dependencies.validator.validate(
                    question=state["prepared_question"],
                    evidence=state["evidence"],
                    candidate=candidate,
                )
            _enforce_evidence_binding(validated_answer, state["evidence"])
            await record_node(
                state,
                node_name="validate_answer",
                extra={"validated_citation_count": len(validated_answer.cited_chunk_ids)},
            )
            return {"validated_answer": validated_answer}

        return await guarded(state, node_name="validate_answer", action=action)

    @observe(name="rag.persist_completed", capture_input=False, capture_output=False)
    async def persist_completed(state: RAGState) -> NodeResult:
        async def action() -> NodeResult:
            output = await dependencies.persistence.complete_execution(
                user_id=state["user_id"],
                assistant_message_id=state["execution"].assistant_message_id,
                answer=state["validated_answer"],
            )
            if output.status is not MessageStatus.COMPLETED:
                raise WorkflowExecutionError("Persistence did not complete the assistant message")
            await record_node(
                state,
                node_name="persist_completed",
                extra={"citation_count": len(output.citation_snapshot)},
            )
            return {"output": output}

        return await guarded(state, node_name="persist_completed", action=action)

    @observe(name="rag.persist_failed", capture_input=False, capture_output=False)
    async def persist_failed(state: RAGState) -> NodeResult:
        failure = state["failure"]
        output = await dependencies.persistence.fail_execution(
            user_id=state["user_id"],
            assistant_message_id=state["execution"].assistant_message_id,
            error_code=failure.error_code,
            error_message=failure.safe_message,
        )
        if output.status is not MessageStatus.FAILED:
            raise WorkflowExecutionError("Persistence did not fail the assistant message")
        await record_node(
            state,
            node_name="persist_failed",
            extra={"failure_code": failure.error_code},
        )
        return {"output": output}

    graph = StateGraph(RAGState)
    graph.add_node("load_conversation", load_conversation)
    graph.add_node("create_execution", create_execution)
    graph.add_node("prepare_question", prepare_question)
    graph.add_node("retrieve_evidence", retrieve_evidence)
    graph.add_node("assess_evidence", assess_evidence)
    graph.add_node("abstain", abstain)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("validate_answer", validate_answer)
    graph.add_node("persist_completed", persist_completed)
    graph.add_node("persist_failed", persist_failed)
    graph.add_edge(START, "load_conversation")
    graph.add_conditional_edges(
        "load_conversation",
        _route_after_initialization,
        {"continue": "create_execution", "stop": END},
    )
    graph.add_conditional_edges(
        "create_execution",
        _route_after_execution_creation,
        {"continue": "prepare_question", "stop": END},
    )
    graph.add_conditional_edges(
        "prepare_question",
        _route_after_execution,
        {"continue": "retrieve_evidence", "failed": "persist_failed"},
    )
    graph.add_conditional_edges(
        "retrieve_evidence",
        _route_after_execution,
        {"continue": "assess_evidence", "failed": "persist_failed"},
    )
    graph.add_conditional_edges(
        "assess_evidence",
        _route_after_assessment,
        {
            "abstain": "abstain",
            "generate": "generate_answer",
            "failed": "persist_failed",
        },
    )
    graph.add_conditional_edges(
        "abstain",
        _route_after_execution,
        {"continue": "validate_answer", "failed": "persist_failed"},
    )
    graph.add_conditional_edges(
        "generate_answer",
        _route_after_execution,
        {"continue": "validate_answer", "failed": "persist_failed"},
    )
    graph.add_conditional_edges(
        "validate_answer",
        _route_after_execution,
        {"continue": "persist_completed", "failed": "persist_failed"},
    )
    graph.add_conditional_edges(
        "persist_completed",
        _route_after_terminal_persistence,
        {"complete": END, "failed": "persist_failed"},
    )
    graph.add_edge("persist_failed", END)
    return RAGWorkflow(graph.compile(), dependencies)


def _route_after_initialization(state: RAGState) -> str:
    return "stop" if "failure" in state else "continue"


def _route_after_execution_creation(state: RAGState) -> str:
    return "stop" if "failure" in state or "output" in state else "continue"


def _route_after_execution(state: RAGState) -> str:
    return "failed" if "failure" in state else "continue"


def _route_after_assessment(state: RAGState) -> str:
    if "failure" in state:
        return "failed"
    return "generate" if state["evidence_sufficient"] else "abstain"


def _route_after_terminal_persistence(state: RAGState) -> str:
    return "failed" if "failure" in state else "complete"


def _classify_failure(node_name: str, exc: Exception) -> WorkflowFailure:
    if isinstance(exc, ConversationAuthorizationError):
        return WorkflowFailure(
            node_name=node_name,
            error_code="conversation_authorization_failed",
            safe_message="Conversation access could not be established.",
        )
    if isinstance(exc, RetrievalError):
        return WorkflowFailure(
            node_name=node_name,
            error_code="retrieval_failed",
            safe_message="Evidence retrieval could not be completed.",
        )
    if node_name == "retrieve_evidence":
        return WorkflowFailure(
            node_name=node_name,
            error_code="retrieval_failed",
            safe_message="Evidence retrieval could not be completed.",
        )
    if node_name == "generate_answer":
        return WorkflowFailure(
            node_name=node_name,
            error_code="generation_failed",
            safe_message="Answer generation could not be completed.",
        )
    if node_name == "validate_answer":
        return WorkflowFailure(
            node_name=node_name,
            error_code="validation_failed",
            safe_message="The generated answer could not be validated.",
        )
    return WorkflowFailure(
        node_name=node_name,
        error_code="assistant_execution_failed",
        safe_message="The assistant could not complete this request.",
    )


def _enforce_evidence_binding(answer: AnswerCandidate, evidence: RetrievalResult) -> None:
    if not answer.content.strip():
        raise ValueError("Validated answer content cannot be blank")
    citation_ids = answer.cited_chunk_ids
    if len(set(citation_ids)) != len(citation_ids):
        raise ValueError("Validated answer citation identifiers must be unique")
    if answer.outcome is AnswerOutcome.ABSTAINED:
        if citation_ids or answer.unsupported_content:
            raise ValueError("Abstained answers cannot contain citations")
        return
    if not citation_ids:
        raise ValueError("Supported answers require at least one citation")
    if (
        answer.outcome is AnswerOutcome.PARTIALLY_ANSWERED
        and not (answer.unsupported_content or "").strip()
    ):
        raise ValueError("Partial answers must identify the unsupported portion")
    if (
        answer.outcome is AnswerOutcome.PARTIALLY_ANSWERED
        and (answer.unsupported_content or "") not in answer.content
    ):
        raise ValueError("Partial answer content must state its unsupported portion")
    if answer.outcome is AnswerOutcome.FULLY_ANSWERED and answer.unsupported_content:
        raise ValueError("Fully answered results cannot contain unsupported portions")
    retrieved_ids = {chunk.chunk_id for chunk in evidence.evidence}
    if not set(citation_ids).issubset(retrieved_ids):
        raise ValueError("Validated answer cited evidence that was not retrieved")
