from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    NumberObject,
)

from rag_assistant.db.models import ParagraphKind
from rag_assistant.extraction import (
    ExtractionFailureCode,
    PdfExtractionError,
    PdfExtractor,
    PersianTextNormalizer,
)
from rag_assistant.extraction.pdf import classify_source_text

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "documents" / "sample_doc.pdf"


def _serialize(writer: PdfWriter) -> bytes:
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


def _blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    return _serialize(writer)


def _encrypted_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    writer.encrypt("test-password")
    return _serialize(writer)


def _rotated_pdf() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    page.rotate(90)
    return _serialize(writer)


def _image_only_pdf() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=100)
    image = DecodedStreamObject()
    image.set_data(b"\xff\xff\xff")
    image.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(1),
            NameObject("/Height"): NumberObject(1),
            NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
            NameObject("/BitsPerComponent"): NumberObject(8),
        }
    )
    image_reference = writer._add_object(image)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/XObject"): DictionaryObject({NameObject("/Im0"): image_reference})}
    )
    content = DecodedStreamObject()
    content.set_data(b"q 100 0 0 100 0 0 cm /Im0 Do Q")
    page[NameObject("/Contents")] = writer._add_object(content)
    return _serialize(writer)


def _ruled_table_pdf() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
    )
    commands = b"\n".join(
        [
            b"q 1 w",
            b"50 200 m 250 200 l S",
            b"50 150 m 250 150 l S",
            b"50 100 m 250 100 l S",
            b"50 100 m 50 200 l S",
            b"150 100 m 150 200 l S",
            b"250 100 m 250 200 l S",
            b"Q",
            b"BT /F1 12 Tf 60 170 Td (Name) Tj ET",
            b"BT /F1 12 Tf 160 170 Td (Value) Tj ET",
            b"BT /F1 12 Tf 60 120 Td (alpha) Tj ET",
            b"BT /F1 12 Tf 160 120 Td (42) Tj ET",
        ]
    )
    content = DecodedStreamObject()
    content.set_data(commands)
    page[NameObject("/Contents")] = writer._add_object(content)
    return _serialize(writer)


def test_persian_normalization_is_deterministic_and_retrieval_oriented() -> None:
    normalizer = PersianTextNormalizer()
    source = "  يَكـ\tمی\u200c\u200cروم \u200f۱۲٣\nكاربرد  "

    normalized = normalizer.normalize(source)

    assert normalized == "یک می‌روم 123 کاربرد"
    assert normalizer.normalize(normalized) == normalized
    assert normalizer.normalize("الف \u200c ب") == "الف ب"


def test_source_classification_covers_supported_paragraph_kinds() -> None:
    assert (
        classify_source_text("۱. مقدمه", font_size=20, body_font_size=16) is ParagraphKind.HEADING
    )
    assert (
        classify_source_text("۲.۳ پردازش سند", font_size=16, body_font_size=16)
        is ParagraphKind.HEADING
    )
    assert (
        classify_source_text("ب) تقسیم‌بندی معنایی", font_size=16, body_font_size=16)
        is ParagraphKind.HEADING
    )
    assert (
        classify_source_text("• مورد نخست", font_size=16, body_font_size=16)
        is ParagraphKind.LIST_ITEM
    )
    assert (
        classify_source_text("شکل ۱: معماری سامانه", font_size=16, body_font_size=16)
        is ParagraphKind.CAPTION
    )
    assert (
        classify_source_text("این یک بند عادی است.", font_size=16, body_font_size=16)
        is ParagraphKind.PROSE
    )


def test_golden_persian_pdf_has_stable_page_local_citations() -> None:
    payload = SAMPLE_PDF.read_bytes()
    extractor = PdfExtractor()

    first = extractor.extract(payload)
    second = extractor.extract(BytesIO(payload))

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.metadata.page_count == 13
    assert [page.page_number for page in first.pages] == list(range(1, 14))
    assert all(page.blocks for page in first.pages)
    assert all(
        [block.paragraph_number for block in page.blocks] == list(range(1, len(page.blocks) + 1))
        for page in first.pages
    )
    identities = {(block.page_number, block.paragraph_number) for block in first.paragraphs}
    assert len(identities) == len(first.paragraphs)
    assert all(
        block.page_number == page.page_number for page in first.pages for block in page.blocks
    )
    assert "مقدمه و پیشینه" in first.pages[0].blocks[0].original_text
    assert first.pages[0].blocks[0].normalized_text.startswith("1")
    assert {block.kind for block in first.paragraphs} >= {
        ParagraphKind.HEADING,
        ParagraphKind.PROSE,
        ParagraphKind.LIST_ITEM,
    }


def test_ruled_table_rows_preserve_original_and_add_header_context() -> None:
    result = PdfExtractor().extract(_ruled_table_pdf())

    rows = [block for block in result.paragraphs if block.kind is ParagraphKind.TABLE_ROW]

    assert len(rows) == 2
    assert rows[0].original_text == "Name | Value"
    assert rows[0].normalized_text == "Name | Value"
    assert rows[1].original_text == "alpha | 42"
    assert rows[1].normalized_text == "Name | Value | alpha | 42"
    assert rows[0].metadata["is_header"] is True
    assert rows[1].metadata["is_header"] is False


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (b"not a pdf", ExtractionFailureCode.MALFORMED),
        (_blank_pdf(), ExtractionFailureCode.EMPTY),
        (_encrypted_pdf(), ExtractionFailureCode.ENCRYPTED),
        (_image_only_pdf(), ExtractionFailureCode.OCR_REQUIRED),
        (_rotated_pdf(), ExtractionFailureCode.UNSUPPORTED_LAYOUT),
    ],
)
def test_unsupported_pdf_inputs_fail_with_stable_codes(
    payload: bytes,
    expected_code: ExtractionFailureCode,
) -> None:
    with pytest.raises(PdfExtractionError) as captured:
        PdfExtractor().extract(payload)

    assert captured.value.code is expected_code
    assert "original_text" not in str(captured.value.metadata)
