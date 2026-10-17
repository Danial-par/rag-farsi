"""Manual OpenRouter evaluation against the committed, non-sensitive PDF fixture."""

from __future__ import annotations

import re
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from rag_assistant.assistant.generation import OpenRouterAnswerGenerator
from rag_assistant.assistant.models import AnswerCandidate
from rag_assistant.core.config import Settings
from rag_assistant.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    EvaluationObservation,
    EvaluationReport,
    score_evaluation,
)
from rag_assistant.extraction import PdfExtractor, PersianTextNormalizer
from rag_assistant.extraction.domain import SourceBlock
from rag_assistant.retrieval import CitationSource, EvidenceChunk, RetrievalResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PDF = PROJECT_ROOT / "tests" / "fixtures" / "documents" / "sample_doc.pdf"
DEFAULT_DATASET = PROJECT_ROOT / "tests" / "fixtures" / "evaluation" / "sample_doc.json"
_WORD_PATTERN = re.compile(r"[\w]+", re.UNICODE)


async def run_live_evaluation(
    *, settings: Settings, dataset: EvaluationDataset
) -> EvaluationReport:
    """Run model generation manually over deterministic sample-PDF retrieval results."""
    if settings.openrouter_api_key is None:
        raise ValueError("Set OPENROUTER_API_KEY to run live evaluation")
    payload = SAMPLE_PDF.read_bytes()
    extracted = PdfExtractor().extract(payload)
    digest = sha256(payload).hexdigest()
    document_id = uuid5(NAMESPACE_URL, f"rag-evaluation:{digest}")
    version_id = uuid5(document_id, "sample-version")
    all_units = {
        f"{paragraph.page_number}:{paragraph.paragraph_number}": paragraph
        for paragraph in extracted.paragraphs
    }
    generator = OpenRouterAnswerGenerator.from_settings(settings)
    observations: list[EvaluationObservation] = []
    for case in dataset.cases:
        evidence, unit_by_id = _retrieve_fixture_units(
            case=case,
            paragraphs=all_units,
            document_id=document_id,
            version_id=version_id,
        )
        candidate = await generator.generate(question=case.question, history=(), evidence=evidence)
        observations.append(
            _observation(
                case=case,
                candidate=candidate,
                retrieved_ids=tuple(
                    unit_id for chunk in evidence.evidence for unit_id in chunk.citation_sources
                ),
                unit_by_id=unit_by_id,
                persisted_units=tuple(all_units),
            )
        )
    return score_evaluation(dataset, tuple(observations))


def _retrieve_fixture_units(
    *,
    case: EvaluationCase,
    paragraphs: dict[str, SourceBlock],
    document_id: UUID,
    version_id: UUID,
) -> tuple[RetrievalResult, dict[UUID, str]]:
    query = set(_WORD_PATTERN.findall(PersianTextNormalizer().normalize(case.question)))
    ranked: list[tuple[float, str, SourceBlock]] = []
    for identity, paragraph in paragraphs.items():
        text = paragraph.normalized_text
        words = set(_WORD_PATTERN.findall(text))
        overlap = len(query & words)
        if overlap:
            ranked.append((overlap / max(1, len(query)), identity, paragraph))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    chunks: list[EvidenceChunk] = []
    unit_by_id: dict[UUID, str] = {}
    for ordinal, (score, identity, paragraph) in enumerate(ranked[:5], start=1):
        page_number = paragraph.page_number
        paragraph_number = paragraph.paragraph_number
        source_id = uuid5(version_id, identity)
        unit_by_id[source_id] = identity
        source = CitationSource(
            source_paragraph_id=source_id,
            document_id=document_id,
            document_version_id=version_id,
            document_title="Sample Persian RAG document",
            page_number=page_number,
            paragraph_number=paragraph_number,
            kind=paragraph.kind,
            original_text=paragraph.original_text,
        )
        normalized_text = paragraph.normalized_text
        chunks.append(
            EvidenceChunk(
                chunk_id=uuid5(version_id, f"chunk:{identity}"),
                document_id=document_id,
                document_version_id=version_id,
                document_title="Sample Persian RAG document",
                chunk_ordinal=ordinal,
                normalized_text=normalized_text,
                token_count=max(1, len(normalized_text.split())),
                cosine_similarity=score,
                citation_sources=(source,),
            )
        )
    return (
        RetrievalResult(
            normalized_query=PersianTextNormalizer().normalize(case.question),
            embedding_model="deterministic-lexical-evaluation",
            embedding_dimensions=1,
            evidence=tuple(chunks),
        ),
        unit_by_id,
    )


def _observation(
    *,
    case: EvaluationCase,
    candidate: AnswerCandidate,
    retrieved_ids: Sequence[CitationSource],
    unit_by_id: dict[UUID, str],
    persisted_units: tuple[str, ...],
) -> EvaluationObservation:
    retrieved = tuple(unit_by_id[item.source_paragraph_id] for item in retrieved_ids)
    cited = tuple(unit_by_id[item] for item in candidate.cited_paragraph_ids if item in unit_by_id)
    return EvaluationObservation(
        case_id=case.case_id,
        retrieved_units=retrieved,
        cited_units=cited,
        persisted_units=persisted_units,
        outcome=candidate.outcome.value,
    )
