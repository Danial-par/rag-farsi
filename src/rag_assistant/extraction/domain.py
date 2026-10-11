from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_assistant.db.models import ParagraphKind


class ExtractionFailureCode(StrEnum):
    """Stable machine-readable reasons why a PDF cannot be extracted safely."""

    ENCRYPTED = "pdf_encrypted"
    MALFORMED = "pdf_malformed"
    EMPTY = "pdf_empty"
    OCR_REQUIRED = "pdf_ocr_required"
    UNSUPPORTED_LAYOUT = "pdf_unsupported_layout"


class ExtractionWarningCode(StrEnum):
    """Stable non-fatal extraction diagnostics."""

    BLANK_PAGE = "blank_page"
    NON_TEXT_IMAGE_IGNORED = "non_text_image_ignored"


class BoundingBox(BaseModel):
    """A PDF-space rectangle expressed from the page's top-left corner."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    left: float = Field(ge=0)
    top: float = Field(ge=0)
    right: float = Field(ge=0)
    bottom: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_edges(self) -> Self:
        if self.right < self.left or self.bottom < self.top:
            raise ValueError("Bounding-box edges are reversed")
        return self


class ExtractionWarning(BaseModel):
    """A safe diagnostic that never contains extracted document text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: ExtractionWarningCode
    message: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)


class ParserMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    role: str = Field(min_length=1)


class PageExtractionSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(ge=1)
    width_points: float = Field(gt=0)
    height_points: float = Field(gt=0)
    paragraph_count: int = Field(ge=0)


class ExtractionMetadata(BaseModel):
    """Deterministic metadata required to reproduce an extraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    extraction_version: str = Field(min_length=1)
    normalization_version: str = Field(min_length=1)
    text_parser: ParserMetadata
    layout_parser: ParserMetadata
    layout_profile: str = Field(min_length=1)
    page_count: int = Field(ge=0)
    pages: tuple[PageExtractionSummary, ...] = ()
    warnings: tuple[ExtractionWarning, ...] = ()


class SourceBlock(BaseModel):
    """One immutable, page-bounded citation paragraph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(ge=1)
    paragraph_number: int = Field(ge=1)
    kind: ParagraphKind
    original_text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    bounding_box: BoundingBox
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedPage(BaseModel):
    """The ordered citation blocks extracted from one physical PDF page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: int = Field(ge=1)
    width_points: float = Field(gt=0)
    height_points: float = Field(gt=0)
    blocks: tuple[SourceBlock, ...] = ()
    warnings: tuple[ExtractionWarning, ...] = ()

    @model_validator(mode="after")
    def validate_block_identity(self) -> Self:
        expected_numbers = list(range(1, len(self.blocks) + 1))
        if [block.paragraph_number for block in self.blocks] != expected_numbers:
            raise ValueError("Paragraph numbers must be consecutive and one-based per page")
        if any(block.page_number != self.page_number for block in self.blocks):
            raise ValueError("A source block cannot cross its physical page boundary")
        return self


class ExtractedDocument(BaseModel):
    """Validated deterministic output of the PDF source-processing stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pages: tuple[ExtractedPage, ...]
    metadata: ExtractionMetadata

    @model_validator(mode="after")
    def validate_page_identity(self) -> Self:
        expected_numbers = list(range(1, len(self.pages) + 1))
        if [page.page_number for page in self.pages] != expected_numbers:
            raise ValueError("Physical pages must be consecutive and one-based")
        if self.metadata.page_count != len(self.pages):
            raise ValueError("Extraction metadata page count does not match extracted pages")
        return self

    @property
    def paragraphs(self) -> tuple[SourceBlock, ...]:
        return tuple(block for page in self.pages for block in page.blocks)


class PdfExtractionError(RuntimeError):
    """A safe extraction failure with a stable persistence code."""

    def __init__(
        self,
        code: ExtractionFailureCode,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.metadata = metadata or {}
