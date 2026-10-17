from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from rag_assistant.core.config import Settings
from rag_assistant.extraction import PersianTextNormalizer
from rag_assistant.indexing import EmbeddingBatchPolicy, EmbeddingClient, embed_in_batches
from rag_assistant.retrieval.errors import (
    IncompatibleEmbeddingConfigurationError,
    InvalidRetrievalQueryError,
    RetrievalScopeError,
)
from rag_assistant.retrieval.models import (
    CitationSource,
    EvidenceChunk,
    RetrievalRequest,
    RetrievalResult,
)
from rag_assistant.retrieval.repository import RetrievalRepository, RetrievalScope


class RetrievalService:
    """Normalize, embed, authorize, rank, and resolve evidence for one question."""

    def __init__(
        self,
        *,
        session: AsyncSession,
        embedding_client: EmbeddingClient,
        settings: Settings,
        normalizer: PersianTextNormalizer | None = None,
    ) -> None:
        self._repository = RetrievalRepository(session)
        self._embedding_client = embedding_client
        self._settings = settings
        self._normalizer = normalizer or PersianTextNormalizer()

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        normalized_query = self._normalizer.normalize(request.question)
        if not normalized_query:
            raise InvalidRetrievalQueryError(
                "The query has no searchable content after normalization"
            )

        scope = await self._inspect_scope(request)
        self._validate_scope(scope)
        if scope.active_document_count == 0:
            return self._empty_result(normalized_query)

        vectors = await embed_in_batches(
            self._embedding_client,
            [normalized_query],
            model=self._settings.embedding_model,
            dimensions=self._settings.embedding_dimensions,
            policy=EmbeddingBatchPolicy(
                batch_size=1,
                concurrency=1,
                max_retries=self._settings.embedding_max_retries,
            ),
        )
        records = await self._repository.retrieve(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            query_embedding=vectors[0],
            embedding_model=self._settings.embedding_model,
            embedding_dimensions=self._settings.embedding_dimensions,
            top_k=request.top_k,
            minimum_similarity=request.minimum_similarity,
        )

        # Recheck after the external embedding call so ownership or configuration
        # changes cannot turn into a silently partial result.
        final_scope = await self._inspect_scope(request)
        self._validate_scope(final_scope)
        if final_scope.active_document_count == 0:
            return self._empty_result(normalized_query)

        evidence = tuple(
            EvidenceChunk(
                chunk_id=record.chunk_id,
                document_id=record.document_id,
                document_version_id=record.document_version_id,
                document_title=record.document_title,
                chunk_ordinal=record.chunk_ordinal,
                normalized_text=record.normalized_text,
                token_count=record.token_count,
                cosine_similarity=record.cosine_similarity,
                citation_sources=tuple(
                    CitationSource(
                        source_paragraph_id=citation.source_paragraph_id,
                        document_id=record.document_id,
                        document_version_id=record.document_version_id,
                        document_title=record.document_title,
                        page_number=citation.page_number,
                        paragraph_number=citation.paragraph_number,
                        kind=citation.kind,
                        original_text=citation.original_text,
                        start_offset=citation.start_offset,
                        end_offset=citation.end_offset,
                    )
                    for citation in record.citations
                ),
            )
            for record in records
        )
        return RetrievalResult(
            normalized_query=normalized_query,
            embedding_model=self._settings.embedding_model,
            embedding_dimensions=self._settings.embedding_dimensions,
            evidence=evidence,
        )

    async def _inspect_scope(self, request: RetrievalRequest) -> RetrievalScope:
        return await self._repository.inspect_scope(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            embedding_model=self._settings.embedding_model,
            embedding_dimensions=self._settings.embedding_dimensions,
        )

    @staticmethod
    def _validate_scope(scope: RetrievalScope) -> None:
        if not scope.conversation_exists:
            raise RetrievalScopeError("Owned conversation scope could not be established")
        if scope.incompatible_document_count:
            raise IncompatibleEmbeddingConfigurationError(
                "A selected active document uses an incompatible embedding configuration"
            )

    def _empty_result(self, normalized_query: str) -> RetrievalResult:
        return RetrievalResult(
            normalized_query=normalized_query,
            embedding_model=self._settings.embedding_model,
            embedding_dimensions=self._settings.embedding_dimensions,
            evidence=(),
        )
