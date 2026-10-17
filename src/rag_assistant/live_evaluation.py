"""Credential-gated end-to-end evaluation through the configured RAG stack."""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Sequence
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID

from sqlalchemy import select

from rag_assistant.assistant import (
    AnswerGenerator,
    EvidenceAnswerValidator,
    LangfuseNodeObserver,
    OpenRouterAnswerGenerator,
    RAGDependencies,
    RAGGraphInput,
    SqlAlchemyWorkflowPersistence,
    create_rag_workflow,
)
from rag_assistant.core.config import Settings
from rag_assistant.core.observability import configure_langfuse, shutdown_langfuse
from rag_assistant.core.security import PasswordManager
from rag_assistant.db.base import utc_now
from rag_assistant.db.models import (
    Conversation,
    Document,
    DocumentVersion,
    Message,
    MessageRole,
    MessageStatus,
    SourceParagraph,
)
from rag_assistant.db.repositories import ConversationRepository, UserRepository
from rag_assistant.db.session import AsyncSessionFactory, create_database_resources
from rag_assistant.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationObservation,
    EvaluationReport,
    score_evaluation,
)
from rag_assistant.indexing.embeddings import EmbeddingClient, OpenRouterEmbeddingClient
from rag_assistant.main import _SessionScopedRetriever
from rag_assistant.retrieval import RetrievalRequest, RetrievalResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "tests" / "fixtures" / "evaluation" / "sample_doc.json"
DEFAULT_ARTIFACTS = PROJECT_ROOT / ".local-data" / "evaluation-runs"
SAMPLE_PDF = PROJECT_ROOT / "tests" / "fixtures" / "documents" / "sample_doc.pdf"
SAMPLE_SEED_KEY = "sample_doc_pdf_v1"


class EvaluationExecutionError(RuntimeError):
    """A safe failure category for a case whose graph did not complete."""

    def __init__(self, case_id: str, error_code: str) -> None:
        super().__init__(f"Case {case_id} did not complete: {error_code}")
        self.case_id = case_id
        self.error_code = error_code


class _ObservedRetriever:
    """Delegate to production retrieval while retaining only safe source identities."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_result: RetrievalResult | None = None

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.last_result = await self._inner.retrieve(request)
        return self.last_result


async def run_live_evaluation(
    *,
    settings: Settings,
    dataset: EvaluationDataset,
    case_ids: Sequence[str] | None = None,
    artifacts_dir: Path = DEFAULT_ARTIFACTS,
    embedding_client: EmbeddingClient | None = None,
    workflow: Any | None = None,
    session_factory: AsyncSessionFactory | None = None,
    answer_generator: AnswerGenerator | None = None,
) -> EvaluationReport:
    """Run selected cases through live embeddings, authorized pgvector retrieval and graph."""
    if settings.openrouter_api_key is None:
        raise ValueError("Set OPENROUTER_API_KEY to run live evaluation")
    configure_langfuse(settings)
    selected = _select_cases(dataset, case_ids)
    if not selected:
        raise ValueError("At least one evaluation case must be selected")
    await asyncio.to_thread(artifacts_dir.mkdir, parents=True, exist_ok=True)
    run_path = artifacts_dir / f"{dataset.dataset_id}.json"
    completed = await asyncio.to_thread(_read_checkpoint, run_path, dataset.dataset_id)
    owns_engine = session_factory is None
    engine = None
    if session_factory is None:
        engine, session_factory = create_database_resources(settings)
    owns_embedding = embedding_client is None
    if embedding_client is None:
        embedding_client = OpenRouterEmbeddingClient(
            base_url=str(settings.openrouter_base_url),
            api_key=settings.openrouter_api_key.get_secret_value(),
            connect_timeout_seconds=settings.embedding_connect_timeout_seconds,
            read_timeout_seconds=settings.embedding_read_timeout_seconds,
        )
    observed_retriever: _ObservedRetriever | None = None
    if workflow is None:
        observed_retriever = _ObservedRetriever(
            _SessionScopedRetriever(session_factory, embedding_client, settings)
        )
        workflow = create_rag_workflow(
            RAGDependencies(
                retriever=observed_retriever,
                generator=answer_generator or OpenRouterAnswerGenerator.from_settings(settings),
                validator=EvidenceAnswerValidator(),
                persistence=SqlAlchemyWorkflowPersistence(session_factory),
                observer=LangfuseNodeObserver(),
                execution_heartbeat_seconds=settings.assistant_execution_recovery_poll_seconds,
            )
        )
    try:
        sample_id, version_id, paragraph_by_id = await _active_sample(session_factory)
        known_units = set(paragraph_by_id.values())
        invalid_expected = {
            unit for case in selected for unit in case.expected_units if unit not in known_units
        }
        if invalid_expected:
            raise ValueError(
                "Dataset citation identities do not resolve to the active persisted sample: "
                + ", ".join(sorted(invalid_expected))
            )
        user_id = await _evaluation_user(session_factory, sample_id)
        for case in selected:
            if case.case_id in completed:
                continue
            conversation_id = await _evaluation_conversation(
                session_factory,
                user_id=user_id,
                sample_id=sample_id,
                dataset_id=dataset.dataset_id,
                case=case,
            )
            started = perf_counter()
            if observed_retriever is not None:
                observed_retriever.last_result = None
            idempotency_key = await _next_idempotency_key(
                session_factory,
                settings=settings,
                user_id=user_id,
                conversation_id=conversation_id,
                base_key=f"{dataset.dataset_id}:{case.case_id}",
            )
            output = await workflow.ainvoke(
                RAGGraphInput(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    question=case.question,
                    idempotency_key=idempotency_key,
                )
            )
            retrieval = observed_retriever.last_result if observed_retriever else None
            if retrieval is None:
                # An idempotent replay can return a persisted answer without entering retrieval.
                retrieval = await _retrieve_once(
                    session_factory,
                    embedding_client,
                    settings,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    case=case,
                )
            latency_ms = (perf_counter() - started) * 1000
            retrieved = _retrieved_units(retrieval)
            persisted = _snapshot_units(output.citation_snapshot, paragraph_by_id)
            cited = persisted
            usage = await _persisted_usage(
                session_factory, assistant_message_id=output.assistant_message_id
            )
            completed[case.case_id] = EvaluationObservation(
                case_id=case.case_id,
                retrieved_units=retrieved,
                retrieved_unit_groups=_retrieved_unit_groups(retrieval),
                cited_units=cited,
                persisted_units=persisted,
                outcome=output.outcome.value if output.outcome else None,
                execution_status=(
                    "completed" if output.status is MessageStatus.COMPLETED else "failed"
                ),
                error_code=output.error_code,
                latency_ms=latency_ms,
                token_usage=usage,
            ).model_dump(mode="json")
            await asyncio.to_thread(
                _write_checkpoint,
                run_path,
                dataset.dataset_id,
                sample_id,
                version_id,
                [completed[case_id] for case_id in sorted(completed)],
            )
            if output.status is not MessageStatus.COMPLETED or output.outcome is None:
                failure_code = output.error_code or "execution_failed"
                raise EvaluationExecutionError(case.case_id, failure_code)
        selected_ids = {case.case_id for case in selected}
        scored_cases = tuple(
            case
            for case in dataset.cases
            if case.case_id in selected_ids and case.case_id in completed
        )
        observations = tuple(
            EvaluationObservation.model_validate(completed[case.case_id]) for case in scored_cases
        )
        return score_evaluation(
            EvaluationDataset(dataset_id=dataset.dataset_id, cases=scored_cases), observations
        )
    finally:
        if owns_embedding:
            await embedding_client.aclose()  # type: ignore[attr-defined]
        if owns_engine and engine is not None:
            await engine.dispose()
        shutdown_langfuse(settings)


def _select_cases(
    dataset: EvaluationDataset, case_ids: Sequence[str] | None
) -> tuple[EvaluationCase, ...]:
    if case_ids is None:
        return dataset.cases
    requested = set(case_ids)
    known = {case.case_id for case in dataset.cases}
    if unknown := requested - known:
        raise ValueError(f"Unknown evaluation case IDs: {', '.join(sorted(unknown))}")
    return tuple(case for case in dataset.cases if case.case_id in requested)


def _read_checkpoint(path: Path, dataset_id: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("dataset_id") != dataset_id:
        raise ValueError("Evaluation checkpoint belongs to a different dataset")
    if value.get("report_version") != 2:
        return {}
    observations = value.get("observations", [])
    return {
        item["case_id"]: item
        for item in observations
        if item.get("execution_status", "completed") == "completed"
        and item.get("outcome") is not None
    }


def _write_checkpoint(
    path: Path,
    dataset_id: str,
    sample_id: UUID,
    version_id: UUID,
    observations: list[dict[str, Any]],
) -> None:
    safe_report = {
        "report_version": 2,
        "dataset_id": dataset_id,
        "sample_document_id": str(sample_id),
        "sample_version_id": str(version_id),
        "observations": observations,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(safe_report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


async def _active_sample(
    session_factory: AsyncSessionFactory,
) -> tuple[UUID, UUID, dict[UUID, str]]:
    payload_hash = sha256(SAMPLE_PDF.read_bytes()).hexdigest()
    async with session_factory() as session:
        document = await session.scalar(
            select(Document).where(
                Document.is_shared.is_(True),
                Document.document_metadata["seed_key"].astext == SAMPLE_SEED_KEY,
                Document.deleted_at.is_(None),
            )
        )
        if document is None or document.active_version_id is None:
            raise ValueError("The seeded sample document has no active indexed version")
        version = await session.get(DocumentVersion, document.active_version_id)
        if (
            version is None
            or version.content_hash != payload_hash
            or version.status.value != "succeeded"
        ):
            raise ValueError("The active sample version does not match the fixture content hash")
        paragraphs = list(
            await session.scalars(
                select(SourceParagraph).where(SourceParagraph.document_version_id == version.id)
            )
        )
        paragraph_by_id = {
            item.id: f"{item.page_number}:{item.paragraph_number}" for item in paragraphs
        }
        if not paragraphs:
            raise ValueError("The active sample version contains no persisted source paragraphs")
        return document.id, version.id, paragraph_by_id


async def _evaluation_user(session_factory: AsyncSessionFactory, sample_id: UUID) -> UUID:
    email = f"rag-eval-{sample_id.hex[:16]}@local.invalid"
    async with session_factory.begin() as session:
        repository = UserRepository(session)
        user = await repository.get_by_normalized_email(email)
        if user is None:
            user = await repository.create(
                email=email,
                password_hash=PasswordManager().hash(secrets.token_urlsafe(32)),
            )
        return user.id


async def _evaluation_conversation(
    session_factory: AsyncSessionFactory,
    *,
    user_id: UUID,
    sample_id: UUID,
    dataset_id: str,
    case: EvaluationCase,
) -> UUID:
    marker = f"sample-eval:{dataset_id}:{sample_id}:{case.case_id}"
    async with session_factory.begin() as session:
        conversation = await session.scalar(
            select(Conversation).where(
                Conversation.owner_id == user_id,
                Conversation.conversation_metadata["evaluation_key"].astext == marker,
                Conversation.deleted_at.is_(None),
            )
        )
        repository = ConversationRepository(session)
        if conversation is None:
            conversation = await repository.create(
                owner_id=user_id,
                title=f"Evaluation: {case.case_id}",
                metadata={"evaluation_key": marker},
            )
        await repository.add_document(
            owner_id=user_id, conversation_id=conversation.id, document_id=sample_id
        )
        return conversation.id


async def _retrieve_once(
    session_factory: AsyncSessionFactory,
    embedding_client: EmbeddingClient,
    settings: Settings,
    *,
    user_id: UUID,
    conversation_id: UUID,
    case: EvaluationCase,
) -> RetrievalResult:
    return await _SessionScopedRetriever(session_factory, embedding_client, settings).retrieve(
        RetrievalRequest(user_id=user_id, conversation_id=conversation_id, question=case.question)
    )


async def _persisted_usage(
    session_factory: AsyncSessionFactory, *, assistant_message_id: UUID
) -> dict[str, int]:
    async with session_factory() as session:
        message = await session.get(Message, assistant_message_id)
        if message is None:
            raise ValueError("Completed assistant message could not be resolved")
        return {
            key: value
            for key, value in message.model_usage.items()
            if key in {"input_tokens", "output_tokens", "total_tokens"}
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        }


async def _next_idempotency_key(
    session_factory: AsyncSessionFactory,
    *,
    settings: Settings,
    user_id: UUID,
    conversation_id: UUID,
    base_key: str,
) -> str:
    """Reuse a completed attempt; choose a new key only after a persisted failure."""
    async with session_factory.begin() as session:
        for attempt in range(100):
            key = base_key if attempt == 0 else f"{base_key}:retry-{attempt}"
            user_message = await session.scalar(
                select(Message).where(
                    Message.conversation_id == conversation_id,
                    Message.role == MessageRole.USER,
                    Message.idempotency_key == key,
                )
            )
            if user_message is None:
                return key
            assistant_message = await session.scalar(
                select(Message).where(
                    Message.conversation_id == conversation_id,
                    Message.role == MessageRole.ASSISTANT,
                    Message.idempotency_key == key,
                )
            )
            if assistant_message is None:
                raise ValueError("Evaluation attempt is missing its assistant message")
            if assistant_message.status is MessageStatus.COMPLETED:
                return key
            if assistant_message.status is MessageStatus.PENDING:
                abandoned_before = utc_now() - timedelta(
                    seconds=settings.assistant_execution_timeout_seconds
                )
                if assistant_message.updated_at >= abandoned_before:
                    raise ValueError("A previous evaluation attempt is still pending")
                await ConversationRepository(session).fail_assistant_message(
                    owner_id=user_id,
                    message_id=assistant_message.id,
                    error_code="execution_abandoned",
                    error_message="Evaluation execution exceeded its renewal timeout.",
                )
    raise ValueError("Evaluation exceeded the safe retry limit for one case")


def _retrieved_units(result: RetrievalResult) -> tuple[str, ...]:
    identities = dict.fromkeys(unit for group in _retrieved_unit_groups(result) for unit in group)
    return tuple(identities)


def _retrieved_unit_groups(result: RetrievalResult) -> tuple[tuple[str, ...], ...]:
    """Preserve ranked retrieval chunk boundaries for correct Recall@k scoring."""
    return tuple(
        tuple(
            dict.fromkeys(
                f"{source.page_number}:{source.paragraph_number}"
                for source in chunk.citation_sources
            )
        )
        for chunk in result.evidence
    )


def _snapshot_units(
    snapshot: tuple[dict[str, Any], ...], paragraph_by_id: dict[UUID, str]
) -> tuple[str, ...]:
    units: list[str] = []
    for item in snapshot:
        source_id = UUID(str(item["source_paragraph_id"]))
        identity = paragraph_by_id.get(source_id)
        if identity is None or identity != _identity(item):
            raise ValueError("Persisted citation snapshot does not match its source paragraph")
        units.append(identity)
    return tuple(units)


def _identity(citation: dict[str, Any]) -> str:
    return f"{citation['page_number']}:{citation['paragraph_number']}"
