"""Provenance-aware chunking, embeddings, source loading, and indexing orchestration."""

from rag_assistant.indexing.chunking import (
    CHUNKING_VERSION,
    ChunkDraft,
    ChunkingError,
    ChunkingParagraph,
    ParagraphChunker,
    TiktokenTokenCounter,
)
from rag_assistant.indexing.embeddings import (
    EmbeddingBatchPolicy,
    EmbeddingClient,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingTimeoutError,
    OpenRouterEmbeddingClient,
    PermanentEmbeddingError,
    RetryableEmbeddingError,
    embed_in_batches,
)
from rag_assistant.indexing.service import (
    DocumentIndexingService,
    IndexingResult,
    SourceIntegrityError,
)
from rag_assistant.indexing.storage import (
    LocalObjectStore,
    ObjectStore,
    ObjectStoreError,
    S3ObjectStore,
)

__all__ = [
    "CHUNKING_VERSION",
    "ChunkDraft",
    "ChunkingError",
    "ChunkingParagraph",
    "DocumentIndexingService",
    "EmbeddingBatchPolicy",
    "EmbeddingClient",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingTimeoutError",
    "IndexingResult",
    "LocalObjectStore",
    "ObjectStore",
    "ObjectStoreError",
    "OpenRouterEmbeddingClient",
    "ParagraphChunker",
    "PermanentEmbeddingError",
    "RetryableEmbeddingError",
    "S3ObjectStore",
    "SourceIntegrityError",
    "TiktokenTokenCounter",
    "embed_in_batches",
]
