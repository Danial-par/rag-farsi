"""Shared non-database model stages for retrieval ranking and context decisions."""

from __future__ import annotations

from uuid import UUID

import structlog
from langfuse import get_client, observe

from rag_assistant.core.config import Settings
from rag_assistant.retrieval.models import (
    RankedCandidate,
    RankedCandidates,
    RetrievalCandidates,
    RetrievalDecision,
    SectionSelection,
)
from rag_assistant.retrieval.planner import RetrievalPlanner
from rag_assistant.retrieval.reranker import CandidateReranker

_logger = structlog.stdlib.get_logger("rag_assistant.retrieval")


class RetrievalModelStages:
    """Apply the cross-encoder and structured context planner with safe fallbacks."""

    def __init__(
        self,
        *,
        settings: Settings,
        reranker: CandidateReranker | None,
        planner: RetrievalPlanner | None,
    ) -> None:
        self._settings = settings
        self._reranker = reranker
        self._planner = planner

    @observe(name="rag.retrieval.rerank_candidates", capture_input=False, capture_output=False)
    async def rerank_candidates(self, candidates: RetrievalCandidates) -> RankedCandidates:
        if not candidates.candidates:
            return RankedCandidates(candidates=())
        if self._reranker is None:
            return vector_fallback(candidates)
        try:
            return await self._reranker.rerank(
                question=candidates.normalized_query,
                candidates=candidates,
                top_k=candidates.final_top_k,
            )
        except Exception as exc:
            _logger.warning(
                "retrieval_cross_encoder_fell_back",
                error_type=type(exc).__name__,
                candidate_count=len(candidates.candidates),
            )
            get_client().update_current_span(
                metadata={
                    "reranker_status": "fallback",
                    "reranker_error_type": type(exc).__name__,
                }
            )
            return vector_fallback(candidates)

    @observe(name="rag.retrieval.assess_context", capture_input=False, capture_output=False)
    async def assess_context(
        self,
        *,
        question: str,
        candidates: RetrievalCandidates,
        ranked: RankedCandidates,
    ) -> RetrievalDecision:
        if not ranked.candidates or self._planner is None:
            return RetrievalDecision(action="answer")
        try:
            return await self._planner.assess(
                question=question,
                ranked=ranked,
                section_outline=candidates.section_outline,
                section_outline_truncated=candidates.section_outline_truncated,
            )
        except Exception as exc:
            _logger.warning(
                "retrieval_context_planner_fell_back",
                error_type=type(exc).__name__,
                candidate_count=len(ranked.candidates),
            )
            get_client().update_current_span(
                metadata={
                    "context_planner_status": "fallback",
                    "context_planner_error_type": type(exc).__name__,
                }
            )
            return fallback_expansion_decision(
                candidates,
                ranked,
                max_sections=self._settings.retrieval_max_expanded_sections,
            )


def vector_fallback(candidates: RetrievalCandidates) -> RankedCandidates:
    return RankedCandidates(
        candidates=tuple(
            RankedCandidate(evidence=item, score=item.cosine_similarity)
            for item in candidates.candidates[: candidates.final_top_k]
        ),
        fallback_used=True,
    )


def fallback_expansion_decision(
    candidates: RetrievalCandidates,
    ranked: RankedCandidates,
    *,
    max_sections: int,
) -> RetrievalDecision:
    """Use the top hit's persisted root section if the planner is unavailable."""
    if not max_sections:
        return RetrievalDecision(action="answer")
    for item in ranked.candidates:
        evidence = item.evidence
        if not evidence.section_path:
            continue
        raw_section_id = evidence.section_path[0].get("id")
        if not isinstance(raw_section_id, str):
            continue
        try:
            section_id = UUID(raw_section_id)
        except ValueError:
            continue
        outline_item = next(
            (
                outline
                for outline in candidates.section_outline
                if outline.document_version_id == evidence.document_version_id
                and outline.section_id == section_id
            ),
            None,
        )
        if outline_item is not None:
            return RetrievalDecision(
                action="expand_section",
                sections=(
                    SectionSelection(
                        document_version_id=outline_item.document_version_id,
                        section_id=outline_item.section_id,
                        parent_section_id=outline_item.parent_section_id,
                        heading_path=outline_item.heading_path,
                    ),
                ),
            )
    return RetrievalDecision(action="answer")
