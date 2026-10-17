from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

import tiktoken

from rag_assistant.db.models import ParagraphKind

CHUNKING_VERSION = "paragraph-token-v1"
_BOUNDARY_PATTERN = re.compile(r"[.!?؟؛:\n]\s*|\s+")


class TokenCounter(Protocol):
    """Count model tokens without coupling chunk construction to one tokenizer package."""

    def count(self, text: str) -> int: ...


class TiktokenTokenCounter:
    """Count tokens with the embedding model encoding and a deterministic fallback."""

    def __init__(self, model: str) -> None:
        model_name = model.split("/", maxsplit=1)[-1]
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))


@dataclass(frozen=True, slots=True)
class ChunkingParagraph:
    source_paragraph_id: UUID
    page_number: int
    paragraph_number: int
    kind: ParagraphKind
    normalized_text: str


@dataclass(frozen=True, slots=True)
class ChunkReference:
    source_paragraph_id: UUID
    page_number: int
    paragraph_number: int
    start_offset: int | None = None
    end_offset: int | None = None


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    ordinal: int
    normalized_text: str
    token_count: int
    references: tuple[ChunkReference, ...]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _Segment:
    paragraph: ChunkingParagraph
    text: str
    start_offset: int | None = None
    end_offset: int | None = None

    @property
    def is_complete_paragraph(self) -> bool:
        return self.start_offset is None and self.end_offset is None

    def reference(self) -> ChunkReference:
        return ChunkReference(
            source_paragraph_id=self.paragraph.source_paragraph_id,
            page_number=self.paragraph.page_number,
            paragraph_number=self.paragraph.paragraph_number,
            start_offset=self.start_offset,
            end_offset=self.end_offset,
        )


class ChunkingError(ValueError):
    """The source paragraphs cannot produce safe, provenance-aware chunks."""


class ParagraphChunker:
    """Pack ordered source paragraphs into deterministic token-bounded chunks."""

    def __init__(
        self,
        token_counter: TokenCounter,
        *,
        target_tokens: int = 400,
        max_tokens: int = 600,
        paragraph_overlap: int = 1,
    ) -> None:
        if target_tokens <= 0 or max_tokens < target_tokens:
            raise ValueError("Chunk token budgets are invalid")
        if paragraph_overlap not in {0, 1}:
            raise ValueError("The current chunker supports zero or one paragraph of overlap")
        self._tokens = token_counter
        self._target_tokens = target_tokens
        self._max_tokens = max_tokens
        self._paragraph_overlap = paragraph_overlap

    def chunk(self, paragraphs: Sequence[ChunkingParagraph]) -> list[ChunkDraft]:
        self._validate_source(paragraphs)
        chunks: list[ChunkDraft] = []
        active_heading: _Segment | None = None
        active_heading_used = False
        content: list[_Segment] = []

        def emit(*, allow_overlap: bool) -> None:
            nonlocal active_heading_used, content
            if not content:
                return
            chunks.append(self._build_chunk(len(chunks) + 1, active_heading, content))
            active_heading_used = active_heading is not None
            overlap: list[_Segment] = []
            if allow_overlap and self._paragraph_overlap and content[-1].is_complete_paragraph:
                overlap = [content[-1]]
            content = overlap

        for paragraph in paragraphs:
            if paragraph.kind is ParagraphKind.HEADING:
                emit(allow_overlap=False)
                if active_heading is not None and not active_heading_used:
                    chunks.append(self._build_chunk(len(chunks) + 1, None, [active_heading]))
                active_heading = self._complete_segment(paragraph)
                active_heading_used = False
                continue

            heading_tokens = self._context_token_count(active_heading)
            available = self._max_tokens - heading_tokens
            if available <= 0:
                raise ChunkingError("A heading consumes the complete chunk token budget")
            segments = self._segments(paragraph, available)
            for segment in segments:
                if any(
                    item.paragraph.source_paragraph_id == segment.paragraph.source_paragraph_id
                    for item in content
                ):
                    emit(allow_overlap=False)
                candidate = [*content, segment]
                candidate_tokens = self._chunk_token_count(active_heading, candidate)
                if content and candidate_tokens > self._target_tokens:
                    emit(allow_overlap=True)
                    candidate = [*content, segment]
                    candidate_tokens = self._chunk_token_count(active_heading, candidate)
                if candidate_tokens > self._max_tokens:
                    if content:
                        # The retained overlap already belongs to the chunk just emitted.
                        # Drop it when it would prevent the next paragraph from fitting.
                        content = []
                        candidate = [segment]
                        candidate_tokens = self._chunk_token_count(active_heading, candidate)
                    if candidate_tokens > self._max_tokens:
                        raise ChunkingError("A split paragraph fragment exceeds the chunk limit")
                content = candidate

        emit(allow_overlap=False)
        if active_heading is not None and not active_heading_used:
            chunks.append(self._build_chunk(len(chunks) + 1, None, [active_heading]))
        if not chunks:
            raise ChunkingError("No non-empty chunks were produced")
        return chunks

    @staticmethod
    def _validate_source(paragraphs: Sequence[ChunkingParagraph]) -> None:
        if not paragraphs:
            raise ChunkingError("At least one source paragraph is required")
        identities: set[UUID] = set()
        previous_position = (0, 0)
        for paragraph in paragraphs:
            if paragraph.source_paragraph_id in identities:
                raise ChunkingError("Source paragraph identifiers must be unique")
            identities.add(paragraph.source_paragraph_id)
            position = (paragraph.page_number, paragraph.paragraph_number)
            if position <= previous_position:
                raise ChunkingError("Source paragraphs must be in citation order")
            previous_position = position
            if not paragraph.normalized_text.strip():
                raise ChunkingError("Source paragraphs must contain normalized text")

    @staticmethod
    def _complete_segment(paragraph: ChunkingParagraph) -> _Segment:
        return _Segment(paragraph=paragraph, text=paragraph.normalized_text)

    def _segments(self, paragraph: ChunkingParagraph, budget: int) -> list[_Segment]:
        text = paragraph.normalized_text
        if self._tokens.count(text) <= budget:
            return [self._complete_segment(paragraph)]

        segments: list[_Segment] = []
        start = 0
        while start < len(text):
            end = self._largest_fitting_end(text, start, budget)
            if end <= start:
                raise ChunkingError("A source character cannot fit within the token budget")
            preferred = self._preferred_boundary(text, start, end)
            if preferred > start:
                end = preferred
            raw = text[start:end]
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            segment_start = start + leading
            segment_end = start + trailing
            if segment_end > segment_start:
                segment_text = text[segment_start:segment_end]
                if self._tokens.count(segment_text) > budget:
                    raise ChunkingError("A split paragraph fragment exceeds its token budget")
                segments.append(
                    _Segment(
                        paragraph=paragraph,
                        text=segment_text,
                        start_offset=segment_start,
                        end_offset=segment_end,
                    )
                )
            start = end
        if not segments:
            raise ChunkingError("An oversized paragraph produced no usable fragments")
        return segments

    def _largest_fitting_end(self, text: str, start: int, budget: int) -> int:
        low = start + 1
        high = len(text)
        best = start
        while low <= high:
            middle = (low + high) // 2
            if self._tokens.count(text[start:middle]) <= budget:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        return best

    @staticmethod
    def _preferred_boundary(text: str, start: int, maximum_end: int) -> int:
        minimum_end = start + max(1, int((maximum_end - start) * 0.6))
        boundaries = [
            match.end()
            for match in _BOUNDARY_PATTERN.finditer(text, start, maximum_end)
            if match.end() >= minimum_end
        ]
        return boundaries[-1] if boundaries else maximum_end

    def _context_token_count(self, heading: _Segment | None) -> int:
        if heading is None:
            return 0
        return self._tokens.count(f"{heading.text}\n\n")

    def _chunk_token_count(self, heading: _Segment | None, content: Sequence[_Segment]) -> int:
        return self._tokens.count(self._join_text(heading, content))

    def _build_chunk(
        self,
        ordinal: int,
        heading: _Segment | None,
        content: Sequence[_Segment],
    ) -> ChunkDraft:
        segments = ([heading] if heading is not None else []) + list(content)
        text = self._join_text(heading, content)
        token_count = self._tokens.count(text)
        if not text.strip() or token_count <= 0 or token_count > self._max_tokens:
            raise ChunkingError("A constructed chunk is empty or outside its token budget")
        references: list[ChunkReference] = []
        seen: set[UUID] = set()
        for segment in segments:
            source_paragraph_id = segment.paragraph.source_paragraph_id
            if source_paragraph_id in seen:
                continue
            seen.add(source_paragraph_id)
            references.append(segment.reference())
        return ChunkDraft(
            ordinal=ordinal,
            normalized_text=text,
            token_count=token_count,
            references=tuple(references),
            metadata={
                "chunking_version": CHUNKING_VERSION,
                "target_tokens": self._target_tokens,
                "max_tokens": self._max_tokens,
                "paragraph_overlap": self._paragraph_overlap,
                "has_heading_context": heading is not None,
            },
        )

    @staticmethod
    def _join_text(heading: _Segment | None, content: Sequence[_Segment]) -> str:
        parts = ([heading.text] if heading is not None else []) + [item.text for item in content]
        return "\n\n".join(parts)
