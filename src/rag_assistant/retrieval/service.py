from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

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
    ExpandedRetrievalContext,
    RankedCandidates,
    RetrievalCandidates,
    RetrievalDecision,
    RetrievalRequest,
    RetrievalResult,
    SectionOutlineItem,
    SectionSelection,
)
from rag_assistant.retrieval.planner import RetrievalPlanner
from rag_assistant.retrieval.repository import EvidenceRecord, RetrievalRepository, RetrievalScope
from rag_assistant.retrieval.reranker import CandidateReranker
from rag_assistant.retrieval.stages import RetrievalModelStages


class RetrievalService:
    """Run the typed retrieval stages for one authorized conversation question."""

    def __init__(
        self,
        *,
        session: AsyncSession,
        embedding_client: EmbeddingClient,
        settings: Settings,
        normalizer: PersianTextNormalizer | None = None,
        reranker: CandidateReranker | None = None,
        planner: RetrievalPlanner | None = None,
        model_stages: RetrievalModelStages | None = None,
    ) -> None:
        self._repository = RetrievalRepository(session)
        self._embedding_client = embedding_client
        self._settings = settings
        self._normalizer = normalizer or PersianTextNormalizer()
        self._model_stages = model_stages or RetrievalModelStages(
            settings=settings,
            reranker=reranker,
            planner=planner,
        )

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Execute all retrieval stages for callers outside the LangGraph workflow."""
        candidates = await self.retrieve_candidates(request)
        ranked = await self.rerank_candidates(candidates)
        decision = await self.assess_context(
            question=request.question,
            candidates=candidates,
            ranked=ranked,
        )
        expanded = await self.expand_context(
            request=request,
            candidates=candidates,
            ranked=ranked,
            decision=decision,
        )
        return self.assemble_result(candidates, ranked, expanded)

    @observe(name="rag.retrieval.vector_search", capture_input=False, capture_output=False)
    async def retrieve_candidates(self, request: RetrievalRequest) -> RetrievalCandidates:
        normalized_query = self._normalizer.normalize(request.question)
        if not normalized_query:
            raise InvalidRetrievalQueryError(
                "The query has no searchable content after normalization"
            )

        scope = await self._inspect_scope(request)
        self._validate_scope(scope)
        if scope.active_document_count == 0:
            candidates = self._empty_candidates(normalized_query, request.top_k)
            await self._repository.release_read_transaction()
            return candidates

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
            top_k=self._settings.retrieval_candidate_count,
            minimum_similarity=request.minimum_similarity,
        )
        root_ids: set[tuple[UUID, UUID]] = set()
        for record in records:
            if not record.section_path:
                continue
            raw_root_id = record.section_path[0].get("id")
            if isinstance(raw_root_id, str) and _is_uuid(raw_root_id):
                root_ids.add((record.document_version_id, UUID(raw_root_id)))
        outline, outline_truncated = await self._repository.section_outline(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            roots=tuple(sorted(root_ids, key=lambda item: (str(item[0]), str(item[1])))),
            max_sections=self._settings.retrieval_outline_max_sections,
        )
        await self._repository.release_read_transaction()
        get_client().update_current_span(
            metadata={
                "retrieval_vector_candidate_count": len(records),
                "retrieval_section_outline_count": len(outline),
                "retrieval_section_outline_truncated": int(outline_truncated),
            }
        )
        return RetrievalCandidates(
            normalized_query=normalized_query,
            embedding_model=self._settings.embedding_model,
            embedding_dimensions=self._settings.embedding_dimensions,
            final_top_k=request.top_k,
            candidates=tuple(_to_evidence(record) for record in records),
            section_outline=outline,
            section_outline_truncated=outline_truncated,
        )

    async def rerank_candidates(self, candidates: RetrievalCandidates) -> RankedCandidates:
        return await self._model_stages.rerank_candidates(candidates)

    async def assess_context(
        self,
        *,
        question: str,
        candidates: RetrievalCandidates,
        ranked: RankedCandidates,
    ) -> RetrievalDecision:
        return await self._model_stages.assess_context(
            question=question,
            candidates=candidates,
            ranked=ranked,
        )

    @observe(name="rag.retrieval.validate_scope", capture_input=False, capture_output=False)
    async def validate_scope(
        self,
        *,
        request: RetrievalRequest,
        ranked: RankedCandidates,
        decision: RetrievalDecision,
    ) -> None:
        scope = await self._inspect_scope(request)
        self._validate_scope(scope)
        referenced_versions = {item.evidence.document_version_id for item in ranked.candidates} | {
            section.document_version_id for section in decision.sections
        }
        if not referenced_versions.issubset(scope.active_version_ids):
            raise RetrievalScopeError("Retrieved evidence is no longer in authorized scope")

    @observe(name="rag.retrieval.expand_context", capture_input=False, capture_output=False)
    async def expand_context(
        self,
        *,
        request: RetrievalRequest,
        candidates: RetrievalCandidates,
        ranked: RankedCandidates,
        decision: RetrievalDecision,
    ) -> ExpandedRetrievalContext:
        await self.validate_scope(request=request, ranked=ranked, decision=decision)
        selections = self._validated_selections(candidates.section_outline, decision)
        max_sections = self._settings.retrieval_max_expanded_sections
        if not max_sections:
            selections = ()
        elif len(selections) > max_sections:
            selections = selections[:max_sections]

        records = await self._repository.expand_sections(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            selections=tuple(
                (item.document_version_id, item.section_id, item.heading_path)
                for item in selections
            ),
            max_characters=(
                max(1, self._settings.retrieval_section_max_characters // len(selections))
                if selections
                else self._settings.retrieval_section_max_characters
            ),
        )
        await self.validate_scope(request=request, ranked=ranked, decision=decision)

        sections = tuple(
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
            for record in records
        )
        get_client().update_current_span(
            metadata={
                "context_action": decision.action,
                "selected_section_count": len(selections),
                "expanded_context_section_count": len(sections),
                "expanded_citation_source_count": sum(
                    len(section.citation_sources) for section in sections
                ),
            }
        )
        return ExpandedRetrievalContext(sections=sections)

    @staticmethod
    def assemble_result(
        candidates: RetrievalCandidates,
        ranked: RankedCandidates,
        expanded: ExpandedRetrievalContext,
    ) -> RetrievalResult:
        return RetrievalResult(
            normalized_query=candidates.normalized_query,
            embedding_model=candidates.embedding_model,
            embedding_dimensions=candidates.embedding_dimensions,
            evidence=tuple(item.evidence for item in ranked.candidates),
            context_sections=expanded.sections,
        )

    @staticmethod
    def _validated_selections(
        outline: Sequence[SectionOutlineItem], decision: RetrievalDecision
    ) -> tuple[SectionSelection, ...]:
        available = {(item.document_version_id, item.section_id): item for item in outline}
        for selection in decision.sections:
            item = available.get((selection.document_version_id, selection.section_id))
            if (
                item is None
                or item.parent_section_id != selection.parent_section_id
                or item.heading_path != selection.heading_path
            ):
                raise RetrievalScopeError("Expansion target is outside retrieved section metadata")
        return decision.sections

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

    def _empty_candidates(self, normalized_query: str, final_top_k: int) -> RetrievalCandidates:
        return RetrievalCandidates(
            normalized_query=normalized_query,
            embedding_model=self._settings.embedding_model,
            embedding_dimensions=self._settings.embedding_dimensions,
            final_top_k=final_top_k,
            candidates=(),
        )


def _to_evidence(record: EvidenceRecord) -> EvidenceChunk:
    return EvidenceChunk(
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


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
    except ValueError:
        return False
    return True
