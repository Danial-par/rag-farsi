from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from statistics import median
from typing import Any, BinaryIO, cast

import pdfplumber
import pypdf
from pdfminer.pdfparser import PDFSyntaxError
from pdfplumber.page import Page as LayoutPage
from pdfplumber.table import Table
from pypdf import PageObject, PdfReader
from pypdf.errors import PdfReadError
from pypdf.generic import DictionaryObject

from rag_assistant.db.models import ParagraphKind
from rag_assistant.extraction.domain import (
    BoundingBox,
    ExtractedDocument,
    ExtractedPage,
    ExtractionFailureCode,
    ExtractionMetadata,
    ExtractionWarning,
    ExtractionWarningCode,
    PageExtractionSummary,
    ParserMetadata,
    PdfExtractionError,
    SourceBlock,
)
from rag_assistant.extraction.normalization import PersianTextNormalizer

EXTRACTION_VERSION = "pdf-source-v1"
LAYOUT_PROFILE = "single-column-born-digital-with-ruled-tables-v1"

_DIGITS = "0-9\u0660-\u0669\u06f0-\u06f9"
_BULLET_PATTERN = re.compile(r"^\s*[\u2022\u2023\u2043\u25aa\u25e6-]\s*")
_NUMBERED_LIST_PATTERN = re.compile(rf"^\s*[{_DIGITS}]+[.)]\s*")
_SECTION_NUMBER_PATTERN = re.compile(rf"^\s*[{_DIGITS}]+(?:\s*[.]\s*[{_DIGITS}]+)+\s*")
_LETTERED_HEADING_PATTERN = re.compile(r"^\s*[اآبپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]\s*[.)]\s*")
_CAPTION_PATTERN = re.compile(
    rf"^\s*(?:شکل|جدول|نمودار|تصویر|figure|table)\s*[{_DIGITS}]*\s*[:.)-]?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _TextFragment:
    text: str
    x: float
    y: float
    top: float
    font_size: float
    font_name: str
    sequence: int


@dataclass(frozen=True, slots=True)
class _GeometryLine:
    box: BoundingBox
    has_column_gap: bool = False


@dataclass(frozen=True, slots=True)
class _TextLine:
    text: str
    font_size: float
    box: BoundingBox
    fragments: tuple[_TextFragment, ...]


@dataclass(frozen=True, slots=True)
class _RawBlock:
    kind: ParagraphKind
    original_text: str
    normalized_text: str
    box: BoundingBox
    metadata: dict[str, Any] = field(default_factory=dict)


def classify_source_text(
    text: str,
    *,
    font_size: float,
    body_font_size: float,
) -> ParagraphKind:
    """Classify a logical source line using deterministic layout and text cues."""
    if _CAPTION_PATTERN.match(text):
        return ParagraphKind.CAPTION
    if _BULLET_PATTERN.match(text):
        return ParagraphKind.LIST_ITEM
    if _SECTION_NUMBER_PATTERN.match(text) or _LETTERED_HEADING_PATTERN.match(text):
        return ParagraphKind.HEADING
    if font_size >= body_font_size * 1.15:
        return ParagraphKind.HEADING
    if _NUMBERED_LIST_PATTERN.match(text):
        return ParagraphKind.LIST_ITEM
    return ParagraphKind.PROSE


class PdfExtractor:
    """Extract stable page-local citation blocks from a supported PDF."""

    def __init__(self, normalizer: PersianTextNormalizer | None = None) -> None:
        self._normalizer = normalizer or PersianTextNormalizer()

    def extract(self, source: bytes | BinaryIO) -> ExtractedDocument:
        payload = self._read_source(source)
        reader = self._open_text_reader(payload)
        page_count = len(reader.pages)
        if page_count == 0:
            self._fail(ExtractionFailureCode.EMPTY, "PDF contains no physical pages")

        try:
            layout_document = pdfplumber.open(BytesIO(payload))
        except (PDFSyntaxError, ValueError, TypeError) as exc:
            raise self._error(
                ExtractionFailureCode.MALFORMED,
                "PDF layout could not be parsed",
                page_count=page_count,
            ) from exc

        warnings: list[ExtractionWarning] = []
        pages: list[ExtractedPage] = []
        try:
            if len(layout_document.pages) != page_count:
                self._fail(
                    ExtractionFailureCode.MALFORMED,
                    "PDF parsers disagreed about the physical page count",
                    page_count=page_count,
                )
            for page_index, (text_page, layout_page) in enumerate(
                zip(reader.pages, layout_document.pages, strict=True),
                start=1,
            ):
                page = self._extract_page(
                    page_number=page_index,
                    text_page=text_page,
                    layout_page=layout_page,
                    page_count=page_count,
                    accumulated_warnings=warnings,
                )
                pages.append(page)
                warnings.extend(page.warnings)
        finally:
            layout_document.close()

        if not any(page.blocks for page in pages):
            self._fail(
                ExtractionFailureCode.EMPTY,
                "PDF contains no extractable source text",
                page_count=page_count,
                warnings=warnings,
            )

        metadata = self._metadata(
            page_count=page_count,
            warnings=warnings,
            pages=pages,
        )
        return ExtractedDocument(pages=tuple(pages), metadata=metadata)

    @staticmethod
    def _read_source(source: bytes | BinaryIO) -> bytes:
        payload = source if isinstance(source, bytes) else source.read()
        if not isinstance(payload, bytes) or not payload:
            raise PdfExtractionError(
                ExtractionFailureCode.MALFORMED,
                "PDF input is empty or is not a binary stream",
            )
        return payload

    def _open_text_reader(self, payload: bytes) -> PdfReader:
        try:
            reader = PdfReader(BytesIO(payload), strict=False)
        except (PdfReadError, ValueError, TypeError, OSError) as exc:
            raise self._error(
                ExtractionFailureCode.MALFORMED,
                "PDF could not be parsed",
            ) from exc
        if reader.is_encrypted:
            self._fail(
                ExtractionFailureCode.ENCRYPTED,
                "Encrypted PDFs are not supported",
            )
        return reader

    def _extract_page(
        self,
        *,
        page_number: int,
        text_page: PageObject,
        layout_page: LayoutPage,
        page_count: int,
        accumulated_warnings: Sequence[ExtractionWarning],
    ) -> ExtractedPage:
        if int(text_page.rotation or 0) % 360 != 0:
            self._fail(
                ExtractionFailureCode.UNSUPPORTED_LAYOUT,
                "Rotated PDF pages are outside the supported layout profile",
                page_count=page_count,
                warnings=accumulated_warnings,
            )

        width = float(layout_page.width)
        height = float(layout_page.height)
        fragments, logical_text_lines = self._extract_fragments(text_page, height)
        geometry_lines = self._extract_geometry_lines(layout_page, width)
        text_lines = self._build_text_lines(
            fragments,
            logical_text_lines,
            geometry_lines,
            width,
            height,
        )
        page_warnings: list[ExtractionWarning] = []

        image_coverage = self._image_coverage(layout_page)
        extracted_character_count = sum(len(line.text.strip()) for line in text_lines)
        if image_coverage > 0 and not text_lines:
            self._fail(
                ExtractionFailureCode.OCR_REQUIRED,
                "A physical page contains image content without extractable text",
                page_count=page_count,
                warnings=accumulated_warnings,
            )
        if image_coverage >= 0.35 and extracted_character_count < 80:
            self._fail(
                ExtractionFailureCode.OCR_REQUIRED,
                "A physical page is materially image-based and requires OCR",
                page_count=page_count,
                warnings=accumulated_warnings,
            )
        if image_coverage > 0:
            page_warnings.append(
                ExtractionWarning(
                    code=ExtractionWarningCode.NON_TEXT_IMAGE_IGNORED,
                    message="Non-text image content was not included in source paragraphs",
                    page_number=page_number,
                )
            )

        tables = self._find_ruled_tables(
            layout_page,
            page_count=page_count,
            warnings=accumulated_warnings,
        )
        table_boxes = [self._box_from_tuple(table.bbox) for table in tables]
        if self._has_ambiguous_columns(geometry_lines, table_boxes):
            self._fail(
                ExtractionFailureCode.UNSUPPORTED_LAYOUT,
                "A physical page has an ambiguous multi-column reading order",
                page_count=page_count,
                warnings=accumulated_warnings,
            )

        non_table_lines = [
            line
            for line in text_lines
            if not any(self._center_inside(line.box, table_box) for table_box in table_boxes)
        ]
        raw_blocks = self._group_text_lines(non_table_lines)
        for table_index, table in enumerate(tables, start=1):
            raw_blocks.extend(
                self._extract_table_blocks(
                    table,
                    table_index=table_index,
                    fragments=fragments,
                    page_count=page_count,
                    warnings=accumulated_warnings,
                )
            )
        raw_blocks.sort(key=lambda block: (block.box.top, block.box.left))

        blocks = tuple(
            SourceBlock(
                page_number=page_number,
                paragraph_number=paragraph_number,
                kind=block.kind,
                original_text=block.original_text,
                normalized_text=block.normalized_text,
                bounding_box=block.box,
                metadata=block.metadata,
            )
            for paragraph_number, block in enumerate(raw_blocks, start=1)
            if block.original_text and block.normalized_text
        )
        if not blocks and image_coverage == 0:
            page_warnings.append(
                ExtractionWarning(
                    code=ExtractionWarningCode.BLANK_PAGE,
                    message="Physical page contains no extractable source text",
                    page_number=page_number,
                )
            )
        return ExtractedPage(
            page_number=page_number,
            width_points=width,
            height_points=height,
            blocks=blocks,
            warnings=tuple(page_warnings),
        )

    @staticmethod
    def _extract_fragments(
        page: PageObject, page_height: float
    ) -> tuple[list[_TextFragment], list[str]]:
        fragments: list[_TextFragment] = []

        def visit_text(
            text: str,
            current_matrix: list[float],
            text_matrix: list[float],
            font_dictionary: DictionaryObject | None,
            font_size: float,
        ) -> None:
            del current_matrix
            cleaned = text.replace("\r", "").replace("\n", "")
            if not cleaned.strip():
                return
            font = ""
            if font_dictionary is not None:
                font = str(font_dictionary.get("/BaseFont", ""))
            y = float(text_matrix[5])
            size = abs(float(font_size))
            fragments.append(
                _TextFragment(
                    text=cleaned,
                    x=float(text_matrix[4]),
                    y=y,
                    top=max(0.0, page_height - y - size * 0.5),
                    font_size=size,
                    font_name=font,
                    sequence=len(fragments),
                )
            )

        extracted_text = page.extract_text(visitor_text=visit_text) or ""
        logical_lines = [line.strip() for line in extracted_text.splitlines() if line.strip()]
        return fragments, logical_lines

    @staticmethod
    def _extract_geometry_lines(page: LayoutPage, page_width: float) -> list[_GeometryLine]:
        raw_lines = page.extract_text_lines(char_dir_render="rtl", return_chars=True)
        lines: list[_GeometryLine] = []
        for raw_line in raw_lines:
            characters = cast(list[dict[str, Any]], raw_line.get("chars", []))
            ordered_characters = sorted(characters, key=lambda character: float(character["x0"]))
            gaps = [
                float(right["x0"]) - float(left["x1"])
                for left, right in zip(
                    ordered_characters,
                    ordered_characters[1:],
                    strict=False,
                )
            ]
            lines.append(
                _GeometryLine(
                    box=BoundingBox(
                        left=max(0.0, float(raw_line["x0"])),
                        top=max(0.0, float(raw_line["top"])),
                        right=max(0.0, float(raw_line["x1"])),
                        bottom=max(0.0, float(raw_line["bottom"])),
                    ),
                    has_column_gap=bool(gaps and max(gaps) >= page_width * 0.18),
                )
            )
        return lines

    @staticmethod
    def _build_text_lines(
        fragments: Sequence[_TextFragment],
        logical_text_lines: Sequence[str],
        geometry_lines: Sequence[_GeometryLine],
        page_width: float,
        page_height: float,
    ) -> list[_TextLine]:
        groups: list[list[_TextFragment]] = []
        for fragment in fragments:
            matching_group = next(
                (group for group in groups if abs(group[0].y - fragment.y) <= 1.0),
                None,
            )
            if matching_group is None:
                groups.append([fragment])
            else:
                matching_group.append(fragment)
        groups.sort(key=lambda group: -group[0].y)

        unused_geometry = list(geometry_lines)
        lines: list[_TextLine] = []
        for group_index, group in enumerate(groups):
            ordered = tuple(sorted(group, key=lambda fragment: fragment.sequence))
            if len(logical_text_lines) == len(groups):
                text = logical_text_lines[group_index]
            else:
                text = "".join(fragment.text for fragment in ordered).strip()
            if not text:
                continue
            font_size = max(fragment.font_size for fragment in ordered)
            expected_top = min(fragment.top for fragment in ordered)
            geometry = min(
                unused_geometry,
                key=lambda candidate: abs(candidate.box.top - expected_top),
                default=None,
            )
            if geometry is not None and abs(geometry.box.top - expected_top) <= font_size * 1.5:
                box = geometry.box
                unused_geometry.remove(geometry)
            else:
                left = max(0.0, min(fragment.x for fragment in ordered))
                right = min(
                    page_width,
                    max(fragment.x + max(fragment.font_size, 1.0) for fragment in ordered),
                )
                top = max(0.0, expected_top)
                box = BoundingBox(
                    left=left,
                    top=top,
                    right=max(left, right),
                    bottom=min(page_height, top + max(font_size, 1.0)),
                )
            lines.append(
                _TextLine(
                    text=text,
                    font_size=font_size,
                    box=box,
                    fragments=ordered,
                )
            )
        return lines

    def _group_text_lines(self, lines: Sequence[_TextLine]) -> list[_RawBlock]:
        if not lines:
            return []
        body_font_size = float(median(line.font_size for line in lines))
        top_steps = [
            current.box.top - previous.box.top
            for previous, current in zip(lines, lines[1:], strict=False)
            if current.box.top > previous.box.top
        ]
        ordinary_step = float(median(top_steps)) if top_steps else body_font_size * 1.7
        paragraph_gap = max(ordinary_step * 1.55, body_font_size * 2.2)

        blocks: list[_RawBlock] = []
        index = 0
        while index < len(lines):
            first = lines[index]
            kind = classify_source_text(
                first.text,
                font_size=first.font_size,
                body_font_size=body_font_size,
            )
            grouped = [first]
            index += 1
            if kind not in {ParagraphKind.HEADING, ParagraphKind.CAPTION}:
                while index < len(lines):
                    candidate = lines[index]
                    candidate_kind = classify_source_text(
                        candidate.text,
                        font_size=candidate.font_size,
                        body_font_size=body_font_size,
                    )
                    top_step = candidate.box.top - grouped[-1].box.top
                    if top_step > paragraph_gap:
                        break
                    if candidate_kind in {
                        ParagraphKind.HEADING,
                        ParagraphKind.CAPTION,
                        ParagraphKind.LIST_ITEM,
                    }:
                        break
                    grouped.append(candidate)
                    index += 1

            original_text = "\n".join(line.text for line in grouped).strip()
            normalized_text = self._normalizer.normalize(original_text)
            if normalized_text:
                blocks.append(
                    _RawBlock(
                        kind=kind,
                        original_text=original_text,
                        normalized_text=normalized_text,
                        box=self._union_boxes([line.box for line in grouped]),
                        metadata={
                            "font_size": round(max(line.font_size for line in grouped), 3),
                            "line_count": len(grouped),
                        },
                    )
                )
        return blocks

    def _find_ruled_tables(
        self,
        page: LayoutPage,
        *,
        page_count: int,
        warnings: Sequence[ExtractionWarning],
    ) -> list[Table]:
        try:
            tables = page.find_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise self._error(
                ExtractionFailureCode.UNSUPPORTED_LAYOUT,
                "Ruled table geometry could not be interpreted unambiguously",
                page_count=page_count,
                warnings=warnings,
            ) from exc
        boxes = [self._box_from_tuple(table.bbox) for table in tables]
        for index, first in enumerate(boxes):
            if any(self._boxes_overlap(first, second) for second in boxes[index + 1 :]):
                self._fail(
                    ExtractionFailureCode.UNSUPPORTED_LAYOUT,
                    "Overlapping table regions are not supported",
                    page_count=page_count,
                    warnings=warnings,
                )
        return tables

    def _extract_table_blocks(
        self,
        table: Table,
        *,
        table_index: int,
        fragments: Sequence[_TextFragment],
        page_count: int,
        warnings: Sequence[ExtractionWarning],
    ) -> list[_RawBlock]:
        if len(table.rows) < 2:
            self._fail(
                ExtractionFailureCode.UNSUPPORTED_LAYOUT,
                "A ruled table must contain a header and at least one data row",
                page_count=page_count,
                warnings=warnings,
            )
        column_count = len(table.rows[0].cells)
        if column_count < 2 or any(len(row.cells) != column_count for row in table.rows):
            self._fail(
                ExtractionFailureCode.UNSUPPORTED_LAYOUT,
                "A ruled table has inconsistent row geometry",
                page_count=page_count,
                warnings=warnings,
            )

        rows: list[tuple[str, BoundingBox]] = []
        for row in table.rows:
            if any(cell is None for cell in row.cells):
                self._fail(
                    ExtractionFailureCode.UNSUPPORTED_LAYOUT,
                    "Merged or missing ruled table cells are not supported",
                    page_count=page_count,
                    warnings=warnings,
                )
            cells = cast(list[tuple[float, float, float, float]], row.cells)
            cell_texts = [self._text_in_cell(cell, fragments) for cell in cells]
            original_text = " | ".join(cell_texts).strip()
            if not original_text.replace("|", "").strip():
                continue
            rows.append(
                (original_text, self._union_boxes([self._box_from_tuple(cell) for cell in cells]))
            )

        if len(rows) < 2:
            self._fail(
                ExtractionFailureCode.UNSUPPORTED_LAYOUT,
                "A ruled table does not contain a usable header and data row",
                page_count=page_count,
                warnings=warnings,
            )
        header_text = self._normalizer.normalize(rows[0][0])
        if not header_text:
            self._fail(
                ExtractionFailureCode.UNSUPPORTED_LAYOUT,
                "A ruled table has no usable header text",
                page_count=page_count,
                warnings=warnings,
            )

        blocks: list[_RawBlock] = []
        for row_index, (original_text, box) in enumerate(rows, start=1):
            row_text = self._normalizer.normalize(original_text)
            normalized_text = row_text if row_index == 1 else f"{header_text} | {row_text}"
            blocks.append(
                _RawBlock(
                    kind=ParagraphKind.TABLE_ROW,
                    original_text=original_text,
                    normalized_text=normalized_text,
                    box=box,
                    metadata={
                        "table_index": table_index,
                        "table_row_number": row_index,
                        "is_header": row_index == 1,
                        "column_count": column_count,
                    },
                )
            )
        return blocks

    @staticmethod
    def _text_in_cell(
        cell: tuple[float, float, float, float],
        fragments: Sequence[_TextFragment],
    ) -> str:
        left, top, right, bottom = cell
        selected = [
            fragment
            for fragment in fragments
            if left - 2 <= fragment.x <= right + 2 and top - 2 <= fragment.top <= bottom + 2
        ]
        if not selected:
            return ""
        line_groups: list[list[_TextFragment]] = []
        for fragment in sorted(selected, key=lambda item: (-item.y, item.sequence)):
            group = next(
                (candidate for candidate in line_groups if abs(candidate[0].y - fragment.y) <= 1.0),
                None,
            )
            if group is None:
                line_groups.append([fragment])
            else:
                group.append(fragment)
        return "\n".join(
            "".join(
                fragment.text for fragment in sorted(group, key=lambda item: item.sequence)
            ).strip()
            for group in line_groups
        ).strip()

    @staticmethod
    def _has_ambiguous_columns(
        lines: Sequence[_GeometryLine],
        table_boxes: Sequence[BoundingBox],
    ) -> bool:
        non_table_lines = [
            line
            for line in lines
            if not any(PdfExtractor._center_inside(line.box, box) for box in table_boxes)
        ]
        return sum(line.has_column_gap for line in non_table_lines) >= 3

    @staticmethod
    def _image_coverage(page: LayoutPage) -> float:
        page_area = float(page.width) * float(page.height)
        if page_area <= 0:
            return 0.0
        covered = sum(
            max(0.0, float(image["x1"]) - float(image["x0"]))
            * max(0.0, float(image["bottom"]) - float(image["top"]))
            for image in page.images
        )
        return min(1.0, covered / page_area)

    def _metadata(
        self,
        *,
        page_count: int,
        warnings: Sequence[ExtractionWarning],
        pages: Sequence[ExtractedPage] = (),
    ) -> ExtractionMetadata:
        return ExtractionMetadata(
            extraction_version=EXTRACTION_VERSION,
            normalization_version=self._normalizer.version,
            text_parser=ParserMetadata(
                name="pypdf",
                version=pypdf.__version__,
                role="authoritative_text",
            ),
            layout_parser=ParserMetadata(
                name="pdfplumber",
                version=pdfplumber.__version__,
                role="geometry_and_ruled_tables",
            ),
            layout_profile=LAYOUT_PROFILE,
            page_count=page_count,
            pages=tuple(
                PageExtractionSummary(
                    page_number=page.page_number,
                    width_points=page.width_points,
                    height_points=page.height_points,
                    paragraph_count=len(page.blocks),
                )
                for page in pages
            ),
            warnings=tuple(warnings),
        )

    def _error(
        self,
        code: ExtractionFailureCode,
        message: str,
        *,
        page_count: int = 0,
        warnings: Sequence[ExtractionWarning] = (),
    ) -> PdfExtractionError:
        return PdfExtractionError(
            code,
            message,
            metadata=self._metadata(
                page_count=page_count,
                warnings=warnings,
            ).model_dump(mode="json"),
        )

    def _fail(
        self,
        code: ExtractionFailureCode,
        message: str,
        *,
        page_count: int = 0,
        warnings: Sequence[ExtractionWarning] = (),
    ) -> None:
        raise self._error(
            code,
            message,
            page_count=page_count,
            warnings=warnings,
        )

    @staticmethod
    def _box_from_tuple(values: tuple[float, float, float, float]) -> BoundingBox:
        left, top, right, bottom = values
        return BoundingBox(left=left, top=top, right=right, bottom=bottom)

    @staticmethod
    def _union_boxes(boxes: Sequence[BoundingBox]) -> BoundingBox:
        return BoundingBox(
            left=min(box.left for box in boxes),
            top=min(box.top for box in boxes),
            right=max(box.right for box in boxes),
            bottom=max(box.bottom for box in boxes),
        )

    @staticmethod
    def _center_inside(inner: BoundingBox, outer: BoundingBox) -> bool:
        center_x = (inner.left + inner.right) / 2
        center_y = (inner.top + inner.bottom) / 2
        return outer.left <= center_x <= outer.right and outer.top <= center_y <= outer.bottom

    @staticmethod
    def _boxes_overlap(first: BoundingBox, second: BoundingBox) -> bool:
        return not (
            first.right <= second.left
            or second.right <= first.left
            or first.bottom <= second.top
            or second.bottom <= first.top
        )
