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
    RetrievalRequest,
    RetrievalResult,
)
from rag_assistant.retrieval.service import RetrievalService

__all__ = [
    "CitationSource",
    "EvidenceChunk",
    "EvidenceSection",
    "IncompatibleEmbeddingConfigurationError",
    "InvalidRetrievalQueryError",
    "RetrievalError",
    "RetrievalIntegrityError",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievalScopeError",
    "RetrievalService",
]
