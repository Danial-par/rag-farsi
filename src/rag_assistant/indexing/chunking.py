from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

import tiktoken

from rag_assistant.db.models import ParagraphKind

CHUNKING_VERSION = "structure-aware-v2"
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
    """Pack source paragraphs, keeping list groups and consecutive headings together."""

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
        for headings, body in self._sections(paragraphs):
            for context, content in self._pack_groups(headings, self._groups(body)):
                chunks.append(self._build_chunk(len(chunks) + 1, context, content))
        if not chunks:
            raise ChunkingError("No non-empty chunks were produced")
        return chunks

    def _sections(
        self, paragraphs: Sequence[ChunkingParagraph]
    ) -> Iterator[tuple[list[_Segment], list[_Segment]]]:
        headings: list[_Segment] = []
        body: list[_Segment] = []
        for paragraph in paragraphs:
            segment = self._complete_segment(paragraph)
            if paragraph.kind is ParagraphKind.HEADING:
                if body:
                    yield headings, body
                    headings, body = [], []
                # Adjacent headings often represent a section followed by a subsection.
                # Keep both without guessing hierarchy from unreliable PDF numbering.
                headings.append(segment)
            else:
                body.append(segment)
        if body:
            yield headings, body
        elif headings:
            # A document ending in headings still needs all its source text represented.
            yield [], headings

    @staticmethod
    def _groups(body: Sequence[_Segment]) -> list[list[_Segment]]:
        groups: list[list[_Segment]] = []
        index = 0
        while index < len(body):
            group = [body[index]]
            index += 1
            if group[0].paragraph.kind in {ParagraphKind.PROSE, ParagraphKind.LIST_ITEM}:
                while index < len(body) and body[index].paragraph.kind is ParagraphKind.LIST_ITEM:
                    group.append(body[index])
                    index += 1
            groups.append(group)
        return groups

    def _pack_groups(
        self, context: list[_Segment], groups: Sequence[list[_Segment]]
    ) -> Iterator[tuple[list[_Segment], list[_Segment]]]:
        content: list[_Segment] = []
        for group in groups:
            if self._chunk_token_count(context, group) > self._max_tokens:
                if content:
                    yield context, content
                    content = []
                if len(group) == 1:
                    fragments = self._segments(group[0].paragraph, context)
                    for segment in fragments[:-1]:
                        yield context, [segment]
                    # The last fragment can share a chunk with following source units.
                    content = [fragments[-1]]
                else:
                    # Split long lists at item boundaries, repeating the introduction
                    # as source-backed context whenever it leaves room for item text.
                    intro = group[0]
                    prefix = [*context, intro]
                    can_repeat_intro = intro.paragraph.kind is ParagraphKind.PROSE and all(
                        self._tokens.count(self._join_text(prefix, []) + "\n\n" + item.text[0])
                        <= self._max_tokens
                        for item in group[1:]
                    )
                    if can_repeat_intro:
                        yield from self._pack_groups(prefix, [[item] for item in group[1:]])
                    else:
                        yield from self._pack_groups(context, [[item] for item in group])
                continue

            candidate = [*content, *group]
            candidate_tokens = self._chunk_token_count(context, candidate)
            keep_group = len(group) > 1 and candidate_tokens <= self._max_tokens
            if content and candidate_tokens > self._target_tokens and not keep_group:
                yield context, content
                last = content[-1]
                # Repeating a lone bullet out of its list loses its defining context.
                content = (
                    [last]
                    if self._paragraph_overlap
                    and last.is_complete_paragraph
                    and last.paragraph.kind is ParagraphKind.PROSE
                    else []
                )
                candidate = [*content, *group]
            if self._chunk_token_count(context, candidate) > self._max_tokens:
                candidate = group  # Drop optional overlap before breaking a logical group.
            content = candidate
        if content:
            yield context, content

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
            if min(position) < 1 or position <= previous_position:
                raise ChunkingError("Source paragraphs must be in citation order")
            previous_position = position
            if not paragraph.normalized_text.strip():
                raise ChunkingError("Source paragraphs must contain normalized text")

    @staticmethod
    def _complete_segment(paragraph: ChunkingParagraph) -> _Segment:
        return _Segment(paragraph=paragraph, text=paragraph.normalized_text)

    def _segments(
        self, paragraph: ChunkingParagraph, context: Sequence[_Segment]
    ) -> list[_Segment]:
        text = paragraph.normalized_text
        prefix = self._join_text(context, []) + "\n\n" if context else ""
        if self._tokens.count(prefix + text) <= self._max_tokens:
            return [self._complete_segment(paragraph)]

        segments: list[_Segment] = []
        start = 0
        while start < len(text):
            end = self._largest_fitting_end(text, start, prefix)
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
                if self._tokens.count(prefix + segment_text) > self._max_tokens:
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

    def _largest_fitting_end(self, text: str, start: int, prefix: str) -> int:
        low = start + 1
        high = len(text)
        best = start
        while low <= high:
            middle = (low + high) // 2
            if self._tokens.count(prefix + text[start:middle]) <= self._max_tokens:
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

    def _chunk_token_count(self, context: Sequence[_Segment], content: Sequence[_Segment]) -> int:
        return self._tokens.count(self._join_text(context, content))

    def _build_chunk(
        self,
        ordinal: int,
        context: Sequence[_Segment],
        content: Sequence[_Segment],
    ) -> ChunkDraft:
        segments = [*context, *content]
        text = self._join_text(context, content)
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
                "has_heading_context": any(
                    item.paragraph.kind is ParagraphKind.HEADING for item in context
                ),
                "has_list_context": any(
                    item.paragraph.kind is ParagraphKind.PROSE for item in context
                ),
            },
        )

    @staticmethod
    def _join_text(context: Sequence[_Segment], content: Sequence[_Segment]) -> str:
        return "\n\n".join(item.text for item in (*context, *content))
