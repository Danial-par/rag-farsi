"""Offline dataset validation and deterministic metrics for RAG evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Outcome = Literal["fully_answered", "partially_answered", "abstained"]
Category = Literal[
    "persian_supported", "english_supported", "partial", "unsupported", "adversarial_noisy"
]


class EvaluationCase(BaseModel):
    """One human-reviewed question and its physical-page citation identities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    category: Category
    question: str = Field(min_length=1)
    expected_units: tuple[str, ...]
    expected_outcome: Outcome
    ground_truth_note: str = Field(min_length=1)

    @model_validator(mode="after")
    def outcome_matches_category(self) -> EvaluationCase:
        if (self.category == "partial") != (self.expected_outcome == "partially_answered"):
            raise ValueError("partial cases must expect partially_answered")
        if self.category == "unsupported" and self.expected_outcome != "abstained":
            raise ValueError("unsupported cases must expect abstained")
        if self.expected_outcome == "abstained" and self.expected_units:
            raise ValueError("abstention cases cannot expect citation units")
        if self.expected_outcome != "abstained" and not self.expected_units:
            raise ValueError("supported cases must expect citation units")
        return self


class EvaluationDataset(BaseModel):
    """Versioned evaluation set tied to the committed sample document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    cases: tuple[EvaluationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_identifiers(self) -> EvaluationDataset:
        identifiers = [case.case_id for case in self.cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Evaluation case identifiers must be unique")
        return self


class EvaluationObservation(BaseModel):
    """Safe observation from a completed graph execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    retrieved_units: tuple[str, ...]
    retrieved_unit_groups: tuple[tuple[str, ...], ...] = ()
    cited_units: tuple[str, ...]
    persisted_units: tuple[str, ...]
    outcome: Outcome | None
    execution_status: Literal["completed", "failed"] = "completed"
    error_code: str | None = None
    latency_ms: float = Field(default=0, ge=0)
    token_usage: dict[str, int] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregate deterministic metrics; warnings are advisory by default."""

    retrieval_recall_at_k: float
    context_source_recall: float
    citation_validity: float
    citation_recall: float
    outcome_accuracy: float
    abstention_accuracy: float
    partial_answer_accuracy: float
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    token_usage_totals: dict[str, int]
    cases_scored: int
    warnings: tuple[str, ...]


def load_dataset(path: Path) -> EvaluationDataset:
    """Read and validate a human-reviewed evaluation dataset."""
    return EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8-sig"))


def load_observations(path: Path) -> tuple[EvaluationObservation, ...]:
    """Load a recorded offline or live execution result set."""
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation observations must be a JSON array")
    return tuple(EvaluationObservation.model_validate(item) for item in raw)


def score_evaluation(
    dataset: EvaluationDataset,
    observations: tuple[EvaluationObservation, ...],
    *,
    k: int = 5,
    recall_threshold: float = 0.80,
) -> EvaluationReport:
    """Score retrieval, citation integrity, outcomes, latency and token use."""
    if k < 1:
        raise ValueError("k must be positive")
    cases = {case.case_id: case for case in dataset.cases}
    observed = {item.case_id: item for item in observations}
    if len(observed) != len(observations):
        raise ValueError("Evaluation observation identifiers must be unique")
    if not observed.keys() <= cases.keys():
        raise ValueError("Observations contain unknown evaluation case identifiers")
    if observed.keys() != cases.keys():
        raise ValueError("Every dataset case must have exactly one observation")
    if any(item.execution_status != "completed" or item.outcome is None for item in observations):
        raise ValueError("Evaluation metrics require completed assistant executions")

    recall_scores: list[float] = []
    context_recall_scores: list[float] = []
    citation_recall_scores: list[float] = []
    citation_valid = outcome_correct = abstention_correct = partial_correct = 0
    partial_count = 0
    latencies: list[float] = []
    token_totals: dict[str, int] = {}
    for case_id, case in cases.items():
        result = observed[case_id]
        expected = set(case.expected_units)
        if expected:
            top_k_units = (
                {unit for group in result.retrieved_unit_groups[:k] for unit in group}
                if result.retrieved_unit_groups
                else set(result.retrieved_units[:k])
            )
            recall_scores.append(len(expected & top_k_units) / len(expected))
            context_recall_scores.append(
                len(expected & set(result.retrieved_units)) / len(expected)
            )
            citation_recall_scores.append(len(expected & set(result.cited_units)) / len(expected))
        cited = set(result.cited_units)
        if cited.issubset(set(result.persisted_units)) and cited.issubset(
            set(result.retrieved_units)
        ):
            citation_valid += 1
        matches = result.outcome == case.expected_outcome
        outcome_correct += matches
        if case.expected_outcome == "partially_answered":
            partial_count += 1
            partial_correct += matches
        if (case.expected_outcome == "abstained") == (result.outcome == "abstained"):
            abstention_correct += 1
        latencies.append(result.latency_ms)
        for name, count in result.token_usage.items():
            if count >= 0:
                token_totals[name] = token_totals.get(name, 0) + count

    denominator = len(cases)
    recall = _mean(recall_scores)
    context_recall = _mean(context_recall_scores)
    citation_recall = _mean(citation_recall_scores)
    latencies.sort()
    warnings: list[str] = []
    targets = (
        (recall, recall_threshold, f"retrieval Recall@{k}"),
        (context_recall, recall_threshold, "retrieval context source recall"),
        (citation_valid / denominator, 1.0, "citation validity"),
        (citation_recall, 0.80, "citation recall"),
        (outcome_correct / denominator, 0.80, "outcome accuracy"),
        (abstention_correct / denominator, 0.80, "abstention accuracy"),
    )
    warnings.extend(
        f"{name} {value:.3f} is below target {threshold:.3f}"
        for value, threshold, name in targets
        if value < threshold
    )
    return EvaluationReport(
        retrieval_recall_at_k=recall,
        context_source_recall=context_recall,
        citation_validity=citation_valid / denominator,
        citation_recall=citation_recall,
        outcome_accuracy=outcome_correct / denominator,
        abstention_accuracy=abstention_correct / denominator,
        partial_answer_accuracy=partial_correct / partial_count if partial_count else 1.0,
        average_latency_ms=_mean(latencies),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        token_usage_totals=token_totals,
        cases_scored=denominator,
        warnings=tuple(warnings),
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return values[min(len(values) - 1, max(0, int((len(values) - 1) * quantile + 0.5)))]


def report_json(report: EvaluationReport) -> str:
    """Serialize aggregate scores without provider payloads or credentials."""
    return json.dumps(
        {
            "cases_scored": report.cases_scored,
            "retrieval_recall_at_k": report.retrieval_recall_at_k,
            "context_source_recall": report.context_source_recall,
            "citation_validity": report.citation_validity,
            "citation_recall": report.citation_recall,
            "outcome_accuracy": report.outcome_accuracy,
            "abstention_accuracy": report.abstention_accuracy,
            "partial_answer_accuracy": report.partial_answer_accuracy,
            "average_latency_ms": report.average_latency_ms,
            "p50_latency_ms": report.p50_latency_ms,
            "p95_latency_ms": report.p95_latency_ms,
            "token_usage_totals": report.token_usage_totals,
            "warnings": list(report.warnings),
        },
        ensure_ascii=False,
        indent=2,
    )
