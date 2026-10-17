from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncConnection, async_sessionmaker

from rag_assistant.assistant import AnswerCandidate
from rag_assistant.assistant.models import ConversationHistoryMessage
from rag_assistant.core.config import Settings
from rag_assistant.db.models import AnswerOutcome
from rag_assistant.db.session import AsyncSessionFactory
from rag_assistant.evaluation import (
    EvaluationDataset,
    EvaluationObservation,
    load_dataset,
    score_evaluation,
)
from rag_assistant.extraction import PdfExtractor
from rag_assistant.indexing.service import DocumentIndexingService
from rag_assistant.indexing.storage import LocalObjectStore
from rag_assistant.live_evaluation import (
    _read_checkpoint,
    _select_cases,
    _snapshot_units,
    _write_checkpoint,
    run_live_evaluation,
)
from rag_assistant.sample_seed import seed_sample_document

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


def test_sample_evaluation_units_resolve_to_extracted_citation_paragraphs() -> None:
    dataset = load_dataset(FIXTURE_ROOT / "evaluation" / "sample_doc.json")
    extraction = PdfExtractor().extract(
        (FIXTURE_ROOT / "documents" / "sample_doc.pdf").read_bytes()
    )
    persisted_units = {
        f"{item.page_number}:{item.paragraph_number}" for item in extraction.paragraphs
    }

    assert len(dataset.cases) == 30
    assert Counter(case.category for case in dataset.cases) == {
        "persian_supported": 12,
        "english_supported": 5,
        "partial": 4,
        "unsupported": 6,
        "adversarial_noisy": 3,
    }
    assert all(set(case.expected_units) <= persisted_units for case in dataset.cases)
    assert any(any(ord(character) > 127 for character in case.question) for case in dataset.cases)
    assert any(case.question.isascii() for case in dataset.cases)


def test_evaluation_scores_retrieval_citations_outcomes_and_abstention() -> None:
    dataset = load_dataset(FIXTURE_ROOT / "evaluation" / "sample_doc.json")
    observations = tuple(
        EvaluationObservation(
            case_id=case.case_id,
            retrieved_units=case.expected_units,
            cited_units=case.expected_units if case.expected_outcome != "abstained" else (),
            persisted_units=case.expected_units,
            outcome=case.expected_outcome,
        )
        for case in dataset.cases
    )

    report = score_evaluation(dataset, observations)

    assert report.retrieval_recall_at_k == 1.0
    assert report.citation_validity == 1.0
    assert report.citation_recall == 1.0
    assert report.outcome_accuracy == 1.0
    assert report.abstention_accuracy == 1.0
    assert report.partial_answer_accuracy == 1.0
    assert report.warnings == ()


def test_evaluation_reports_advisory_retrieval_regression_and_invalid_citations() -> None:
    dataset = load_dataset(FIXTURE_ROOT / "evaluation" / "sample_doc.json")
    observations = tuple(
        EvaluationObservation(
            case_id=case.case_id,
            retrieved_units=(),
            cited_units=("99:99",) if case.expected_units else (),
            persisted_units=(),
            outcome="abstained" if case.expected_units else "fully_answered",
        )
        for case in dataset.cases
    )

    report = score_evaluation(dataset, observations)

    assert report.retrieval_recall_at_k == 0.0
    assert report.citation_validity == pytest.approx(7 / 30)
    assert report.outcome_accuracy == 0.0
    assert report.abstention_accuracy < 1.0
    assert report.warnings


def test_metric_percentiles_latency_and_token_totals() -> None:
    dataset = load_dataset(FIXTURE_ROOT / "evaluation" / "sample_doc.json")
    observations = tuple(
        EvaluationObservation(
            case_id=case.case_id,
            retrieved_units=case.expected_units,
            cited_units=case.expected_units if case.expected_outcome != "abstained" else (),
            persisted_units=case.expected_units,
            outcome=case.expected_outcome,
            latency_ms=index * 10,
            token_usage={"input_tokens": 2, "total_tokens": 3},
        )
        for index, case in enumerate(dataset.cases)
    )
    report = score_evaluation(dataset, observations)

    assert report.average_latency_ms == 145
    assert report.p50_latency_ms == 150
    assert report.p95_latency_ms == 280
    assert report.token_usage_totals == {"input_tokens": 60, "total_tokens": 90}


def test_recall_at_five_counts_ranked_chunks_not_flattened_paragraphs() -> None:
    dataset = load_dataset(FIXTURE_ROOT / "evaluation" / "sample_doc.json")
    case = dataset.cases[0]
    one_case = EvaluationDataset(dataset_id=dataset.dataset_id, cases=(case,))
    groups = (("90:1",), ("90:2",), ("90:3",), ("90:4",), case.expected_units)
    observation = EvaluationObservation(
        case_id=case.case_id,
        retrieved_units=tuple(unit for group in groups for unit in group),
        retrieved_unit_groups=groups,
        cited_units=case.expected_units,
        persisted_units=case.expected_units,
        outcome=case.expected_outcome,
    )
    assert score_evaluation(one_case, (observation,)).retrieval_recall_at_k == 1.0


def test_persisted_citation_mapping_resolves_id_and_page_paragraph() -> None:
    source_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    assert _snapshot_units(
        ({"source_paragraph_id": str(source_id), "page_number": 3, "paragraph_number": 2},),
        {source_id: "3:2"},
    ) == ("3:2",)
    with pytest.raises(ValueError, match="does not match"):
        _snapshot_units(
            ({"source_paragraph_id": str(source_id), "page_number": 3, "paragraph_number": 4},),
            {source_id: "3:2"},
        )


def test_subset_selection_and_checkpoint_resume_are_case_idempotent(tmp_path: Path) -> None:
    dataset = load_dataset(FIXTURE_ROOT / "evaluation" / "sample_doc.json")
    selected = _select_cases(dataset, ["fa-rag-stages"])
    assert [case.case_id for case in selected] == ["fa-rag-stages"]
    with pytest.raises(ValueError, match="Unknown evaluation case IDs"):
        _select_cases(dataset, ["not-a-case"])
    path = tmp_path / "checkpoint.json"
    observation = {"case_id": "fa-rag-stages", "outcome": "fully_answered"}
    _write_checkpoint(path, "sample-doc-v2", UUID(int=1), UUID(int=2), [observation])
    assert _read_checkpoint(path, "sample-doc-v2") == {"fa-rag-stages": observation}
    with pytest.raises(ValueError, match="different dataset"):
        _read_checkpoint(path, "other-dataset")
    _write_checkpoint(
        path,
        "sample-doc-v2",
        UUID(int=1),
        UUID(int=2),
        [{"case_id": "failed-case", "execution_status": "failed", "outcome": None}],
    )
    assert _read_checkpoint(path, "sample-doc-v2") == {}


def test_evaluation_requires_exactly_one_observation_per_case() -> None:
    dataset = load_dataset(FIXTURE_ROOT / "evaluation" / "sample_doc.json")

    with pytest.raises(ValueError, match="exactly one observation"):
        score_evaluation(dataset, ())


def test_execution_failures_are_not_scored_as_abstentions() -> None:
    dataset = load_dataset(FIXTURE_ROOT / "evaluation" / "sample_doc.json")
    observations = tuple(
        EvaluationObservation(
            case_id=case.case_id,
            retrieved_units=(),
            cited_units=(),
            persisted_units=(),
            outcome=None,
            execution_status="failed",
            error_code="generation_failed",
        )
        for case in dataset.cases
    )
    with pytest.raises(ValueError, match="completed assistant executions"):
        score_evaluation(dataset, observations)


async def test_live_evaluation_requires_explicit_openrouter_credentials() -> None:
    settings = Settings(
        app_env="test",
        database_url="postgresql+psycopg://user:password@localhost/database",
        jwt_secret="test-only-jwt-secret-with-at-least-32-bytes",
        _env_file=None,
    ).model_copy(update={"openrouter_api_key": None})
    dataset = load_dataset(FIXTURE_ROOT / "evaluation" / "sample_doc.json")

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        await run_live_evaluation(settings=settings, dataset=dataset)


class FixedEmbeddingClient:
    async def embed(
        self, texts: Sequence[str], *, model: str, dimensions: int
    ) -> list[list[float]]:
        del model
        return [[1.0 / dimensions] * dimensions for _ in texts]


class EvidenceEchoGenerator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        *,
        question: str,
        history: tuple[ConversationHistoryMessage, ...],
        evidence: object,
    ) -> AnswerCandidate:
        from rag_assistant.retrieval import RetrievalResult

        del question, history
        self.calls += 1
        assert isinstance(evidence, RetrievalResult)
        citation_ids = tuple(
            dict.fromkeys(
                source.source_paragraph_id
                for chunk in evidence.evidence
                for source in chunk.citation_sources
            )
        )[:1]
        return AnswerCandidate(
            content="Supported by retrieved evidence.",
            outcome=AnswerOutcome.FULLY_ANSWERED,
            cited_paragraph_ids=citation_ids,
        )


async def test_live_runner_uses_graph_retrieval_and_persists_snapshot(
    database_connection: AsyncConnection, settings: Settings, tmp_path: Path
) -> None:
    session_factory: AsyncSessionFactory = async_sessionmaker(
        bind=database_connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    local_settings = settings.model_copy(
        update={"app_env": "local", "openrouter_api_key": "mock-key"}
    )
    sample = await seed_sample_document(
        settings=local_settings,
        session_factory=session_factory,
        object_store=LocalObjectStore(tmp_path / "objects"),
        fixture_path=FIXTURE_ROOT / "documents" / "sample_doc.pdf",
    )
    await DocumentIndexingService(
        session_factory=session_factory,
        object_store=LocalObjectStore(tmp_path / "objects"),
        embedding_client=FixedEmbeddingClient(),
        settings=local_settings,
    ).index_version(sample.version_id)

    class RunnerGenerator(EvidenceEchoGenerator):
        async def generate(
            self,
            *,
            question: str,
            history: tuple[ConversationHistoryMessage, ...],
            evidence: object,
        ) -> AnswerCandidate:
            candidate = await super().generate(
                question=question, history=history, evidence=evidence
            )
            return candidate.model_copy(update={"outcome": AnswerOutcome.FULLY_ANSWERED})

    generator = RunnerGenerator()
    dataset = load_dataset(FIXTURE_ROOT / "evaluation" / "sample_doc.json")
    report = await run_live_evaluation(
        settings=local_settings,
        dataset=dataset,
        case_ids=["fa-rag-stages"],
        artifacts_dir=tmp_path / "reports",
        embedding_client=FixedEmbeddingClient(),
        session_factory=session_factory,
        answer_generator=generator,
    )

    assert report.cases_scored == 1
    assert report.citation_validity == 1.0
    report_file = tmp_path / "reports" / "sample-doc-v2.json"
    observations = json.loads(report_file.read_text(encoding="utf-8"))["observations"]
    assert observations[0]["retrieved_units"]
    assert observations[0]["cited_units"] == observations[0]["persisted_units"]
    assert observations[0]["latency_ms"] > 0
    assert any(observations[0]["persisted_units"])

    resumed = await run_live_evaluation(
        settings=local_settings,
        dataset=dataset,
        case_ids=["fa-rag-stages"],
        artifacts_dir=tmp_path / "reports",
        embedding_client=FixedEmbeddingClient(),
        session_factory=session_factory,
        answer_generator=generator,
    )
    assert resumed.cases_scored == 1
    assert generator.calls == 1

    fresh_dataset = dataset.model_copy(update={"dataset_id": "sample-doc-v2-fresh-run"})
    fresh_run = await run_live_evaluation(
        settings=local_settings,
        dataset=fresh_dataset,
        case_ids=["fa-rag-stages"],
        artifacts_dir=tmp_path / "reports",
        embedding_client=FixedEmbeddingClient(),
        session_factory=session_factory,
        answer_generator=generator,
    )
    assert fresh_run.cases_scored == 1
    assert generator.calls == 2
