from __future__ import annotations

from uuid import UUID

import structlog
from langfuse import get_client, observe
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
    EvidenceSection,
    RetrievalRequest,
    RetrievalResult,
)
from rag_assistant.retrieval.planner import RetrievalPlan, RetrievalPlanner
from rag_assistant.retrieval.repository import (
    EvidenceRecord,
    ExpandedSectionRecord,
    RetrievalRepository,
    RetrievalScope,
)

_logger = structlog.stdlib.get_logger("rag_assistant.retrieval")


class RetrievalService:
    """Normalize, embed, authorize, rank, and resolve evidence for one question."""

    def __init__(
        self,
        *,
        session: AsyncSession,
        embedding_client: EmbeddingClient,
        settings: Settings,
        normalizer: PersianTextNormalizer | None = None,
        planner: RetrievalPlanner | None = None,
    ) -> None:
        self._repository = RetrievalRepository(session)
        self._embedding_client = embedding_client
        self._settings = settings
        self._normalizer = normalizer or PersianTextNormalizer()
        self._planner = planner

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
            top_k=(
                min(self._settings.retrieval_candidate_count, request.top_k * 5)
                if self._planner is not None
                else request.top_k
            ),
            minimum_similarity=request.minimum_similarity,
        )
        get_client().update_current_span(
            metadata={
                "retrieval_vector_candidate_count": len(records),
                "retrieval_planner_status": (
                    "pending" if self._planner is not None and records else "skipped"
                ),
            }
        )

        context_records = []
        if self._planner is not None and records:
            candidate_records = tuple(records)
            # Release the read transaction while the independent reranking call runs.
            await self._repository.release_read_transaction()
            try:
                plan = await self._planner.plan(
                    question=request.question,
                    candidates=records,
                    top_k=request.top_k,
                )
            except Exception as exc:
                _logger.warning(
                    "retrieval_planner_fell_back",
                    error_type=type(exc).__name__,
                    planner_validation_failure=getattr(exc, "failure_reason", None),
                    candidate_count=len(records),
                )
                get_client().update_current_span(
                    metadata={
                        "retrieval_planner_status": "fallback",
                        "retrieval_planner_error_type": type(exc).__name__,
                        "retrieval_planner_validation_failure": getattr(
                            exc, "failure_reason", None
                        ),
                    }
                )
                plan = RetrievalPlan(
                    chunk_ids=tuple(record.chunk_id for record in records[: request.top_k]),
                    section_selections=self._fallback_section_selections(records),
                )
                get_client().update_current_span(
                    metadata={
                        "retrieval_fallback_section_count": len(plan.section_selections),
                    }
                )
            else:
                get_client().update_current_span(
                    metadata={
                        "retrieval_planner_status": "completed",
                        "retrieval_selected_chunk_count": len(plan.chunk_ids),
                        "retrieval_selected_section_count": len(plan.section_selections),
                    }
                )
            records_by_id = {record.chunk_id: record for record in candidate_records}
            records = [records_by_id[item] for item in plan.chunk_ids if item in records_by_id]
            if not records:
                records = list(records_by_id.values())[: request.top_k]
                plan = plan.model_copy(update={"section_selections": ()})

            final_scope = await self._inspect_scope(request)
            self._validate_scope(final_scope)
            if final_scope.active_document_count == 0:
                return self._empty_result(normalized_query)

            selections_with_paths: list[tuple[UUID, UUID, tuple[str, ...], tuple[str, ...]]] = []
            for version_id, section_id in plan.section_selections:
                record = next(
                    (
                        candidate
                        for candidate in candidate_records
                        if candidate.document_version_id == version_id
                        and any(
                            item.get("id") == str(section_id) for item in candidate.section_path
                        )
                    ),
                    None,
                )
                if record is None:
                    continue
                indexed_path = next(
                    (
                        (index, record.section_path[: index + 1])
                        for index, item in enumerate(record.section_path)
                        if item.get("id") == str(section_id)
                    ),
                    None,
                )
                if indexed_path is None:
                    continue
                _, path = indexed_path
                selections_with_paths.append(
                    (
                        version_id,
                        section_id,
                        tuple(str(item["title"]) for item in path),
                        tuple(str(item["id"]) for item in path),
                    )
                )
            selections: list[tuple[UUID, UUID, tuple[str, ...]]] = []
            covered_paths: dict[UUID, set[str]] = {}
            if self._settings.retrieval_max_expanded_sections:
                for version_id, section_id, titles, path_ids in sorted(
                    selections_with_paths, key=lambda item: len(item[3])
                ):
                    if str(section_id) in covered_paths.get(version_id, set()):
                        continue
                    selections.append((version_id, section_id, titles))
                    covered_paths.setdefault(version_id, set()).update(path_ids)
                    if len(selections) >= self._settings.retrieval_max_expanded_sections:
                        break
            context_records = await self._expand_sections(
                request=request,
                selections=selections,
            )

        # Recheck after external model calls so ownership or configuration
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
                        normalized_text=citation.normalized_text,
                        start_offset=citation.start_offset,
                        end_offset=citation.end_offset,
                    )
                    for citation in record.citations
                ),
                section_path=record.section_path,
            )
            for record in records
        )

        return RetrievalResult(
            normalized_query=normalized_query,
            embedding_model=self._settings.embedding_model,
            embedding_dimensions=self._settings.embedding_dimensions,
            evidence=evidence,
            context_sections=tuple(
                EvidenceSection(
                    section_id=record.section_id,
                    document_id=record.document_id,
                    document_version_id=record.document_version_id,
                    document_title=record.document_title,
                    heading_path=record.heading_path,
                    normalized_text=record.normalized_text,
                    truncated=record.truncated,
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
                            normalized_text=citation.normalized_text,
                            start_offset=None,
                            end_offset=None,
                        )
                        for citation in record.citations
                    ),
                )
                for record in context_records
            ),
        )

    @observe(name="rag.retrieval.expand_sections", capture_input=False, capture_output=False)
    async def _expand_sections(
        self,
        *,
        request: RetrievalRequest,
        selections: list[tuple[UUID, UUID, tuple[str, ...]]],
    ) -> list[ExpandedSectionRecord]:
        records = await self._repository.expand_sections(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            selections=selections,
            max_characters=self._settings.retrieval_section_max_characters,
        )
        get_client().update_current_span(
            metadata={
                "selected_section_count": len(selections),
                "expanded_context_section_count": len(records),
                "expanded_citation_source_count": sum(len(record.citations) for record in records),
            }
        )
        return records

    @staticmethod
    def _fallback_section_selections(
        records: tuple[EvidenceRecord, ...] | list[EvidenceRecord],
    ) -> tuple[tuple[UUID, UUID], ...]:
        """Expand the top vector hit's highest available parent if reranking fails."""
        for record in records:
            section_path = record.section_path
            if not section_path:
                continue
            section_id = section_path[0].get("id")
            if not isinstance(section_id, str):
                continue
            try:
                parsed_section_id = UUID(section_id)
            except ValueError:
                continue
            return ((record.document_version_id, parsed_section_id),)
        return ()

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
