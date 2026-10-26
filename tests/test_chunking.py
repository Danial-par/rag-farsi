from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from rag_assistant.db.models import ParagraphKind
from rag_assistant.extraction import PdfExtractor
from rag_assistant.indexing.chunking import (
    ChunkingError,
    ChunkingParagraph,
    ParagraphChunker,
    SectionPathItem,
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


def test_list_and_introduction_stay_together_above_soft_target() -> None:
    paragraphs = [
        _paragraph("مدل های رایج عبارتند از", page=1, number=1),
        _paragraph("• مدل اول", page=1, number=2, kind=ParagraphKind.LIST_ITEM),
        _paragraph("• مدل دوم", page=1, number=3, kind=ParagraphKind.LIST_ITEM),
    ]

    chunks = ParagraphChunker(WordCounter(), target_tokens=6, max_tokens=12).chunk(paragraphs)

    assert len(chunks) == 1
    assert chunks[0].token_count == 11
    assert [ref.source_paragraph_id for ref in chunks[0].references] == [
        paragraph.source_paragraph_id for paragraph in paragraphs
    ]


@pytest.mark.parametrize("overlap", [0, 1])
def test_long_list_repeats_heading_and_intro_across_pages(overlap: int) -> None:
    heading = _paragraph("مدل ها", page=1, number=1, kind=ParagraphKind.HEADING)
    intro = _paragraph("مدل های پیشنهادی", page=1, number=2)
    items = [
        _paragraph("• مدل شماره یک", page=1, number=3, kind=ParagraphKind.LIST_ITEM),
        _paragraph("• مدل شماره دو", page=2, number=1, kind=ParagraphKind.LIST_ITEM),
        _paragraph("• مدل شماره سه", page=2, number=2, kind=ParagraphKind.LIST_ITEM),
    ]
    chunks = ParagraphChunker(
        WordCounter(), target_tokens=10, max_tokens=14, paragraph_overlap=overlap
    ).chunk([heading, intro, *items])

    assert len(chunks) == 3
    for chunk, item in zip(chunks, items, strict=True):
        assert chunk.normalized_text == "\n\n".join(
            [heading.normalized_text, intro.normalized_text, item.normalized_text]
        )
        assert [ref.source_paragraph_id for ref in chunk.references] == [
            heading.source_paragraph_id,
            intro.source_paragraph_id,
            item.source_paragraph_id,
        ]
        assert chunk.metadata["has_list_context"] is True
        assert chunk.token_count <= 14


def test_consecutive_headings_have_content_and_do_not_leak_into_next_section() -> None:
    paragraphs = [
        _paragraph("بخش اصلی", page=1, number=1, kind=ParagraphKind.HEADING),
        _paragraph("زیربخش", page=1, number=2, kind=ParagraphKind.HEADING),
        _paragraph("محتوای بخش", page=1, number=3),
        _paragraph("بخش بعدی", page=1, number=4, kind=ParagraphKind.HEADING),
        _paragraph("محتوای بعدی", page=1, number=5),
    ]

    chunks = ParagraphChunker(WordCounter(), target_tokens=6, max_tokens=12).chunk(paragraphs)

    assert len(chunks) == 2
    assert [ref.source_paragraph_id for ref in chunks[0].references] == [
        paragraph.source_paragraph_id for paragraph in paragraphs[:3]
    ]
    assert [ref.source_paragraph_id for ref in chunks[1].references] == [
        paragraph.source_paragraph_id for paragraph in paragraphs[3:]
    ]


def test_chunk_metadata_preserves_the_complete_parent_child_section_path() -> None:
    parent_id = uuid4()
    child_id = uuid4()
    parent_path = (SectionPathItem(parent_id, "اجزای اصلی", 1),)
    child_path = (
        *parent_path,
        SectionPathItem(child_id, "ورود اسناد", 2),
    )
    paragraphs = [
        ChunkingParagraph(
            source_paragraph_id=parent_id,
            page_number=1,
            paragraph_number=1,
            kind=ParagraphKind.HEADING,
            normalized_text="اجزای اصلی",
            heading_level=1,
            section_path=parent_path,
        ),
        ChunkingParagraph(
            source_paragraph_id=child_id,
            page_number=1,
            paragraph_number=2,
            kind=ParagraphKind.HEADING,
            normalized_text="ورود اسناد",
            heading_level=2,
            section_path=child_path,
        ),
        ChunkingParagraph(
            source_paragraph_id=uuid4(),
            page_number=1,
            paragraph_number=3,
            kind=ParagraphKind.PROSE,
            normalized_text="متن توضیحی",
            section_path=child_path,
        ),
    ]

    chunk = ParagraphChunker(WordCounter(), target_tokens=20, max_tokens=30).chunk(paragraphs)[0]

    assert chunk.metadata["section_path"] == [
        {"id": str(parent_id), "title": "اجزای اصلی", "level": 1},
        {"id": str(child_id), "title": "ورود اسناد", "level": 2},
    ]


def test_oversized_list_item_keeps_intro_and_exact_source_offsets() -> None:
    intro = _paragraph("گزینه ها", page=1, number=1)
    item = _paragraph(
        "• یک دو سه چهار پنج شش هفت هشت نه ده", page=1, number=2, kind=ParagraphKind.LIST_ITEM
    )
    chunks = ParagraphChunker(WordCounter(), target_tokens=5, max_tokens=7).chunk([intro, item])

    assert len(chunks) > 1
    fragments: list[str] = []
    for chunk in chunks:
        intro_ref, item_ref = chunk.references
        assert intro_ref.source_paragraph_id == intro.source_paragraph_id
        assert item_ref.source_paragraph_id == item.source_paragraph_id
        assert item_ref.start_offset is not None and item_ref.end_offset is not None
        fragment = item.normalized_text[item_ref.start_offset : item_ref.end_offset]
        assert chunk.normalized_text == intro.normalized_text + "\n\n" + fragment
        assert chunk.token_count <= 7
        fragments.append(fragment)
    assert "".join(fragments).replace(" ", "") == item.normalized_text.replace(" ", "")


def test_oversized_intro_falls_back_without_dropping_any_source_text() -> None:
    intro = _paragraph("یک دو سه چهار پنج شش هفت هشت نه ده", page=1, number=1)
    item = _paragraph("• گزینه", page=1, number=2, kind=ParagraphKind.LIST_ITEM)
    chunks = ParagraphChunker(WordCounter(), target_tokens=4, max_tokens=5).chunk([intro, item])

    assert all(0 < chunk.token_count <= 5 for chunk in chunks)
    assert {ref.source_paragraph_id for chunk in chunks for ref in chunk.references} == {
        intro.source_paragraph_id,
        item.source_paragraph_id,
    }


def test_list_overlap_does_not_repeat_an_isolated_item_into_unrelated_prose() -> None:
    first = _paragraph("• گزینه اول", page=1, number=1, kind=ParagraphKind.LIST_ITEM)
    second = _paragraph("• گزینه دوم", page=1, number=2, kind=ParagraphKind.LIST_ITEM)
    prose = _paragraph("توضیح مستقل بعدی", page=1, number=3)

    chunks = ParagraphChunker(WordCounter(), target_tokens=5, max_tokens=8).chunk(
        [first, second, prose]
    )

    assert len(chunks) == 2
    assert [ref.source_paragraph_id for ref in chunks[1].references] == [prose.source_paragraph_id]


def test_heading_context_and_separators_are_included_in_split_budget() -> None:
    heading = _paragraph("عنوان", page=1, number=1, kind=ParagraphKind.HEADING)
    paragraph = _paragraph("متن بلند برای آزمایش حد توکن", page=1, number=2)
    chunks = ParagraphChunker(CharacterCounter(), target_tokens=10, max_tokens=15).chunk(
        [heading, paragraph]
    )

    assert len(chunks) > 1
    assert all(len(chunk.normalized_text) == chunk.token_count <= 15 for chunk in chunks)


def test_trailing_heading_is_preserved_and_oversized_context_fails_explicitly() -> None:
    heading = _paragraph("عنوان", page=1, number=1, kind=ParagraphKind.HEADING)
    chunker = ParagraphChunker(CharacterCounter(), target_tokens=5, max_tokens=8)
    assert (
        chunker.chunk([heading])[0].references[0].source_paragraph_id == heading.source_paragraph_id
    )

    with pytest.raises(ChunkingError):
        chunker.chunk(
            [
                _paragraph("عنوان بسیار بلند", page=1, number=1, kind=ParagraphKind.HEADING),
                _paragraph("متن", page=1, number=2),
            ]
        )


def test_sample_pdf_keeps_reranking_list_and_consecutive_headings_together() -> None:
    fixture = Path(__file__).parent / "fixtures" / "documents" / "sample_doc.pdf"
    extracted = PdfExtractor().extract(fixture.read_bytes())
    paragraphs = [
        _paragraph(
            block.normalized_text,
            page=block.page_number,
            number=block.paragraph_number,
            kind=block.kind,
        )
        for block in extracted.paragraphs
    ]
    # A local counter keeps the regression offline; notebook QA uses the real tokenizer.
    chunker = ParagraphChunker(CharacterCounter(), target_tokens=400, max_tokens=1000)
    chunks = chunker.chunk(paragraphs)
    positions = [
        {(ref.page_number, ref.paragraph_number) for ref in chunk.references} for chunk in chunks
    ]

    assert any({(4, number) for number in range(3, 8)} <= refs for refs in positions)
    assert any({(9, 7), (9, 8), (9, 9)} <= refs for refs in positions)
    assert {ref.source_paragraph_id for chunk in chunks for ref in chunk.references} == {
        paragraph.source_paragraph_id for paragraph in paragraphs
    }
    by_id = {paragraph.source_paragraph_id: paragraph for paragraph in paragraphs}
    for chunk in chunks:
        assert chunk.token_count == len(chunk.normalized_text) <= 1000
        assert len(chunk.references) == len({ref.source_paragraph_id for ref in chunk.references})
        # Every displayed character comes from the cited normalized source slices.
        assert chunk.normalized_text == "\n\n".join(
            by_id[ref.source_paragraph_id].normalized_text[ref.start_offset : ref.end_offset]
            for ref in chunk.references
        )
        assert any(
            by_id[ref.source_paragraph_id].kind is not ParagraphKind.HEADING
            for ref in chunk.references
        )
    assert chunks == chunker.chunk(paragraphs)
