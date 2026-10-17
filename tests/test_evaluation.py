from __future__ import annotations

from pathlib import Path

import pytest

from rag_assistant.core.config import Settings
from rag_assistant.evaluation import (
    EvaluationObservation,
    load_dataset,
    score_evaluation,
)
from rag_assistant.extraction import PdfExtractor
from rag_assistant.live_evaluation import run_live_evaluation

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


def test_sample_evaluation_units_resolve_to_extracted_citation_paragraphs() -> None:
    dataset = load_dataset(FIXTURE_ROOT / "evaluation" / "sample_doc.json")
    extraction = PdfExtractor().extract(
        (FIXTURE_ROOT / "documents" / "sample_doc.pdf").read_bytes()
    )
    persisted_units = {
        f"{item.page_number}:{item.paragraph_number}" for item in extraction.paragraphs
    }

    assert {"fully_answered", "partially_answered", "abstained"} == {
        case.expected_outcome for case in dataset.cases
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
    assert report.outcome_accuracy == 1.0
    assert report.abstention_accuracy == 1.0
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
    assert report.citation_validity == 0.2
    assert report.outcome_accuracy == 0.0
    assert report.abstention_accuracy < 1.0
    assert report.warnings


def test_evaluation_requires_exactly_one_observation_per_case() -> None:
    dataset = load_dataset(FIXTURE_ROOT / "evaluation" / "sample_doc.json")

    with pytest.raises(ValueError, match="exactly one observation"):
        score_evaluation(dataset, ())


async def test_live_evaluation_requires_explicit_openrouter_credentials() -> None:
    settings = Settings(
        app_env="test",
        database_url="postgresql+psycopg://user:password@localhost/database",
        jwt_secret="test-only-jwt-secret-with-at-least-32-bytes",
    )
    dataset = load_dataset(FIXTURE_ROOT / "evaluation" / "sample_doc.json")

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        await run_live_evaluation(settings=settings, dataset=dataset)
