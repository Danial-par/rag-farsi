"""Authorization-scoped retrieval for persisted document evidence."""

from rag_assistant.retrieval.errors import (
    IncompatibleEmbeddingConfigurationError,
    InvalidRetrievalQueryError,
    RetrievalError,
    RetrievalIntegrityError,
    RetrievalScopeError,
)
from rag_assistant.retrieval.models import (
    CitationSource,
    EvidenceChunk,
    EvidenceSection,
    ExpandedRetrievalContext,
    RankedCandidate,
    RankedCandidates,
    RetrievalCandidates,
    RetrievalDecision,
    RetrievalRequest,
    RetrievalResult,
    SectionOutlineItem,
    SectionSelection,
)
from rag_assistant.retrieval.reranker import (
    CandidateReranker,
    RerankerError,
    TextEmbeddingsInferenceReranker,
)
from rag_assistant.retrieval.service import RetrievalService

__all__ = [
    "CitationSource",
    "CandidateReranker",
    "EvidenceChunk",
    "EvidenceSection",
    "ExpandedRetrievalContext",
    "IncompatibleEmbeddingConfigurationError",
    "InvalidRetrievalQueryError",
    "RetrievalError",
    "RetrievalCandidates",
    "RetrievalDecision",
    "RetrievalIntegrityError",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievalScopeError",
    "RetrievalService",
    "RankedCandidate",
    "RankedCandidates",
    "RerankerError",
    "SectionOutlineItem",
    "SectionSelection",
    "TextEmbeddingsInferenceReranker",
]
