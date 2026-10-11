"""Deterministic source extraction for citation-ready documents."""

from rag_assistant.extraction.domain import (
    BoundingBox,
    ExtractedDocument,
    ExtractedPage,
    ExtractionFailureCode,
    ExtractionMetadata,
    ExtractionWarning,
    ExtractionWarningCode,
    PdfExtractionError,
    SourceBlock,
)
from rag_assistant.extraction.normalization import (
    NORMALIZATION_VERSION,
    PersianTextNormalizer,
)
from rag_assistant.extraction.pdf import EXTRACTION_VERSION, PdfExtractor
from rag_assistant.extraction.service import DocumentExtractionService

__all__ = [
    "EXTRACTION_VERSION",
    "NORMALIZATION_VERSION",
    "BoundingBox",
    "DocumentExtractionService",
    "ExtractedDocument",
    "ExtractedPage",
    "ExtractionFailureCode",
    "ExtractionMetadata",
    "ExtractionWarning",
    "ExtractionWarningCode",
    "PdfExtractionError",
    "PdfExtractor",
    "PersianTextNormalizer",
    "SourceBlock",
]
