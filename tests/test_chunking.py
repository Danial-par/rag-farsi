from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from rag_assistant.db.models import ParagraphKind
from rag_assistant.indexing.chunking import (
    ChunkingError,
    ChunkingParagraph,
    ParagraphChunker,
    TiktokenTokenCounter,
)


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


class CharacterCounter:
    def count(self, text: str) -> int:
        return len(text)


class FakeEncoding:
    def encode(self, text: str, *, disallowed_special: tuple[object, ...]) -> list[int]:
        del disallowed_special
        return list(range(len(text.split())))


def _paragraph(
    text: str,
    *,
    page: int,
    number: int,
    kind: ParagraphKind = ParagraphKind.PROSE,
    paragraph_id: UUID | None = None,
) -> ChunkingParagraph:
    return ChunkingParagraph(
        source_paragraph_id=paragraph_id or uuid4(),
        page_number=page,
        paragraph_number=number,
        kind=kind,
        normalized_text=text,
    )


def test_chunks_are_deterministic_ordered_and_within_token_limit() -> None:
    paragraphs = [
        _paragraph("یک دو سه", page=1, number=1),
        _paragraph("چهار پنج شش", page=1, number=2),
        _paragraph("هفت هشت نه", page=2, number=1),
    ]
    chunker = ParagraphChunker(WordCounter(), target_tokens=4, max_tokens=8)

    first = chunker.chunk(paragraphs)
    second = chunker.chunk(paragraphs)

    assert first == second
    assert [chunk.ordinal for chunk in first] == list(range(1, len(first) + 1))
    assert all(0 < chunk.token_count <= 8 for chunk in first)


def test_neighboring_chunks_overlap_one_complete_paragraph() -> None:
    first = _paragraph("یک دو سه", page=1, number=1)
    second = _paragraph("چهار پنج شش", page=1, number=2)
    third = _paragraph("هفت هشت نه", page=1, number=3)

    chunks = ParagraphChunker(
        WordCounter(), target_tokens=4, max_tokens=8, paragraph_overlap=1
    ).chunk([first, second, third])

    assert [[ref.source_paragraph_id for ref in chunk.references] for chunk in chunks] == [
        [first.source_paragraph_id],
        [first.source_paragraph_id, second.source_paragraph_id],
        [second.source_paragraph_id, third.source_paragraph_id],
    ]


def test_nearest_heading_is_repeated_as_provenance_backed_context() -> None:
    heading = _paragraph("عنوان بخش", page=1, number=1, kind=ParagraphKind.HEADING)
    first = _paragraph("متن نخست این بخش", page=1, number=2)
    second = _paragraph("متن دوم این بخش", page=2, number=1)

    chunks = ParagraphChunker(WordCounter(), target_tokens=5, max_tokens=9).chunk(
        [heading, first, second]
    )

    assert len(chunks) == 2
    assert all(chunk.normalized_text.startswith("عنوان بخش\n\n") for chunk in chunks)
    assert all(
        chunk.references[0].source_paragraph_id == heading.source_paragraph_id for chunk in chunks
    )
    assert chunks[1].references[-1].page_number == 2


def test_oversized_paragraph_splits_at_valid_normalized_character_offsets() -> None:
    paragraph = _paragraph("بخش اول. بخش دوم. بخش سوم.", page=3, number=2)

    chunks = ParagraphChunker(
        CharacterCounter(), target_tokens=10, max_tokens=12, paragraph_overlap=1
    ).chunk([paragraph])

    assert len(chunks) >= 3
    rebuilt = "".join(
        paragraph.normalized_text[reference.start_offset : reference.end_offset]
        for chunk in chunks
        for reference in chunk.references
        if reference.start_offset is not None and reference.end_offset is not None
    )
    assert rebuilt.replace(" ", "") == paragraph.normalized_text.replace(" ", "")
    assert all(chunk.token_count <= 12 for chunk in chunks)


def test_empty_duplicate_and_out_of_order_sources_are_rejected() -> None:
    paragraph_id = uuid4()
    valid = _paragraph("متن", page=1, number=1, paragraph_id=paragraph_id)
    duplicate = _paragraph("متن دیگر", page=1, number=2, paragraph_id=paragraph_id)
    out_of_order = _paragraph("قبلی", page=1, number=1)
    chunker = ParagraphChunker(WordCounter(), target_tokens=4, max_tokens=8)

    with pytest.raises(ChunkingError):
        chunker.chunk([])
    with pytest.raises(ChunkingError):
        chunker.chunk([valid, duplicate])
    with pytest.raises(ChunkingError):
        chunker.chunk([duplicate, out_of_order])


def test_tiktoken_counter_strips_openrouter_provider_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_models: list[str] = []

    def encoding_for_model(model: str) -> FakeEncoding:
        requested_models.append(model)
        return FakeEncoding()

    monkeypatch.setattr("tiktoken.encoding_for_model", encoding_for_model)

    counter = TiktokenTokenCounter("openai/text-embedding-3-large")

    assert counter.count("یک دو سه") == 3
    assert requested_models == ["text-embedding-3-large"]
