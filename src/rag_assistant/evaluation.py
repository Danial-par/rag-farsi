"""Deterministic scoring for recorded RAG evaluation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Outcome = Literal["fully_answered", "partially_answered", "abstained"]


class EvaluationCase(BaseModel):
    """One question and its human-reviewed expected citation units and outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_units: tuple[str, ...]
    expected_outcome: Outcome


class EvaluationDataset(BaseModel):
    """Versioned evaluation set tied to the committed sample document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    cases: tuple[EvaluationCase, ...] = Field(min_length=1)


class EvaluationObservation(BaseModel):
    """Safe, content-free metrics captured from one completed RAG execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    retrieved_units: tuple[str, ...]
    cited_units: tuple[str, ...]
    persisted_units: tuple[str, ...]
    outcome: Outcome


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregate deterministic metrics; warnings are advisory by default."""

    retrieval_recall_at_k: float
    citation_validity: float
    outcome_accuracy: float
    abstention_accuracy: float
    cases_scored: int
    warnings: tuple[str, ...]


def load_dataset(path: Path) -> EvaluationDataset:
    """Read and validate an evaluation dataset JSON file."""
    return EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))


def load_observations(path: Path) -> tuple[EvaluationObservation, ...]:
    """Load a recorded deterministic or live execution result set."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation observations must be a JSON array")
    return tuple(EvaluationObservation.model_validate(item) for item in raw)


def score_evaluation(
    dataset: EvaluationDataset,
    observations: tuple[EvaluationObservation, ...],
    *,
    k: int = 5,
    recall_threshold: float = 0.9,
) -> EvaluationReport:
    """Score retrieval, citation resolution, outcomes, and abstention deterministically."""
    if k < 1:
        raise ValueError("k must be positive")
    cases = {case.case_id: case for case in dataset.cases}
    observed = {item.case_id: item for item in observations}
    if len(cases) != len(dataset.cases) or len(observed) != len(observations):
        raise ValueError("Evaluation case identifiers must be unique")
    if cases.keys() != observed.keys():
        raise ValueError("Every dataset case must have exactly one observation")

    recall_scores: list[float] = []
    citation_valid = 0
    outcome_correct = 0
    abstention_correct = 0
    for case_id, case in cases.items():
        result = observed[case_id]
        expected = set(case.expected_units)
        if expected:
            recall_scores.append(len(expected & set(result.retrieved_units[:k])) / len(expected))
        cited = set(result.cited_units)
        citations_resolve = cited.issubset(set(result.persisted_units))
        citations_retrieved = cited.issubset(set(result.retrieved_units))
        if citations_resolve and citations_retrieved:
            citation_valid += 1
        if result.outcome == case.expected_outcome:
            outcome_correct += 1
        should_abstain = case.expected_outcome == "abstained"
        did_abstain = result.outcome == "abstained"
        if should_abstain == did_abstain:
            abstention_correct += 1

    recall = sum(recall_scores) / len(recall_scores) if recall_scores else 1.0
    denominator = len(cases)
    warnings: list[str] = []
    if recall < recall_threshold:
        warnings.append(
            f"retrieval Recall@{k} {recall:.3f} is below advisory threshold {recall_threshold:.3f}"
        )
    if citation_valid / denominator < 1.0:
        warnings.append("citation validity is below the advisory threshold 1.000")
    if outcome_correct / denominator < 1.0:
        warnings.append("outcome accuracy is below the advisory threshold 1.000")
    if abstention_correct / denominator < 1.0:
        warnings.append("abstention accuracy is below the advisory threshold 1.000")
    return EvaluationReport(
        retrieval_recall_at_k=recall,
        citation_validity=citation_valid / denominator,
        outcome_accuracy=outcome_correct / denominator,
        abstention_accuracy=abstention_correct / denominator,
        cases_scored=denominator,
        warnings=tuple(warnings),
    )


def report_json(report: EvaluationReport) -> str:
    """Serialize a compact machine-readable evaluation report."""
    return json.dumps(
        {
            "cases_scored": report.cases_scored,
            "retrieval_recall_at_k": report.retrieval_recall_at_k,
            "citation_validity": report.citation_validity,
            "outcome_accuracy": report.outcome_accuracy,
            "abstention_accuracy": report.abstention_accuracy,
            "warnings": list(report.warnings),
        },
        ensure_ascii=False,
        indent=2,
    )
