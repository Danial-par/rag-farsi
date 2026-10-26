from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from rag_assistant.db.models import ParagraphKind


class RetrievalRequest(BaseModel):
    """Internal request for evidence within one user's conversation scope."""

    model_config = ConfigDict(frozen=True)

    user_id: UUID
    conversation_id: UUID
    question: str
    top_k: int = Field(default=6, ge=1, le=20)
    minimum_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)


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


class RetrievalResult(BaseModel):
    """Authorized evidence returned for a normalized query."""

    model_config = ConfigDict(frozen=True)

    normalized_query: str
    embedding_model: str
    embedding_dimensions: int = Field(ge=1)
    evidence: tuple[EvidenceChunk, ...]
    context_sections: tuple[EvidenceSection, ...] = ()
