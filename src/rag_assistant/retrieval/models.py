from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_assistant.db.models import ParagraphKind


class RetrievalRequest(BaseModel):
    """Internal request for evidence within one user's conversation scope."""

    model_config = ConfigDict(frozen=True)

    user_id: UUID
    conversation_id: UUID
    question: str
    top_k: int = Field(default=6, ge=1, le=20)
    minimum_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)


class SectionOutlineItem(BaseModel):
    """One persisted section heading available for bounded context expansion."""

    model_config = ConfigDict(frozen=True)

    document_version_id: UUID
    section_id: UUID
    parent_section_id: UUID | None = None
    document_title: str = Field(min_length=1, max_length=500)
    title: str = Field(min_length=1, max_length=500)
    level: int = Field(ge=1, le=32)
    heading_path: tuple[str, ...] = Field(min_length=1)


class SectionSelection(BaseModel):
    """A planner-selected section, resolved from an available outline alias."""

    model_config = ConfigDict(frozen=True)

    document_version_id: UUID
    section_id: UUID
    parent_section_id: UUID | None = None
    heading_path: tuple[str, ...] = Field(min_length=1)


class RetrievalDecision(BaseModel):
    """A bounded choice to use current evidence or fetch source-backed context."""

    model_config = ConfigDict(frozen=True)

    action: Literal["answer", "expand_section", "fetch_siblings"]
    sections: tuple[SectionSelection, ...] = ()

    @model_validator(mode="after")
    def validate_action_sections(self) -> RetrievalDecision:
        if self.action == "answer" and self.sections:
            raise ValueError("An answer decision cannot include expansion sections")
        if self.action == "expand_section" and len(self.sections) != 1:
            raise ValueError("A section expansion must select exactly one section")
        if self.action == "fetch_siblings":
            if len(self.sections) < 2:
                raise ValueError("Sibling retrieval must select at least two sections")
            parent_ids = {
                (section.document_version_id, section.parent_section_id)
                for section in self.sections
            }
            if len(parent_ids) != 1 or None in {parent for _, parent in parent_ids}:
                raise ValueError("Sibling sections must share one persisted parent")
        return self


class CitationSource(BaseModel):
    """A persisted citation unit with original and normalized retrieval text."""

    model_config = ConfigDict(frozen=True)

    source_paragraph_id: UUID
    document_id: UUID
    document_version_id: UUID
    document_title: str
    page_number: int = Field(ge=1)
    paragraph_number: int = Field(ge=1)
    kind: ParagraphKind
    original_text: str
    normalized_text: str
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=1)


class EvidenceChunk(BaseModel):
    """A ranked retrieval chunk and its complete citation provenance."""

    model_config = ConfigDict(frozen=True)

    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    document_title: str
    chunk_ordinal: int = Field(ge=1)
    normalized_text: str
    token_count: int = Field(ge=1)
    cosine_similarity: float
    citation_sources: tuple[CitationSource, ...] = Field(min_length=1)
    section_path: tuple[dict[str, object], ...] = ()


class RetrievalCandidates(BaseModel):
    """Authorized vector candidates and their source-backed section outline."""

    model_config = ConfigDict(frozen=True)

    normalized_query: str
    embedding_model: str
    embedding_dimensions: int = Field(ge=1)
    final_top_k: int = Field(default=6, ge=1, le=20)
    candidates: tuple[EvidenceChunk, ...]
    section_outline: tuple[SectionOutlineItem, ...] = ()
    section_outline_truncated: bool = False


class RankedCandidate(BaseModel):
    """One cross-encoder score attached to its original authorized evidence."""

    model_config = ConfigDict(frozen=True)

    evidence: EvidenceChunk
    score: float


class RankedCandidates(BaseModel):
    """The final top-K passages after pairwise reranking."""

    model_config = ConfigDict(frozen=True)

    candidates: tuple[RankedCandidate, ...]
    fallback_used: bool = False


class EvidenceSection(BaseModel):
    """A reconstructed, source-backed parent section included for generation."""

    model_config = ConfigDict(frozen=True)

    section_id: UUID
    document_id: UUID
    document_version_id: UUID
    document_title: str
    heading_path: tuple[str, ...]
    normalized_text: str
    citation_sources: tuple[CitationSource, ...] = Field(min_length=1)
    truncated: bool = False


class ExpandedRetrievalContext(BaseModel):
    """Authorized section text loaded after a planner expansion decision."""

    model_config = ConfigDict(frozen=True)

    sections: tuple[EvidenceSection, ...] = ()


class RetrievalResult(BaseModel):
    """Authorized evidence returned for a normalized query."""

    model_config = ConfigDict(frozen=True)

    normalized_query: str
    embedding_model: str
    embedding_dimensions: int = Field(ge=1)
    evidence: tuple[EvidenceChunk, ...]
    context_sections: tuple[EvidenceSection, ...] = ()
