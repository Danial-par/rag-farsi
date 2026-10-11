from __future__ import annotations

import asyncio
from typing import BinaryIO
from uuid import UUID

from rag_assistant.db.repositories import DocumentRepository, NewSourceParagraph
from rag_assistant.extraction.domain import ExtractedDocument, PdfExtractionError
from rag_assistant.extraction.pdf import PdfExtractor


class DocumentExtractionService:
    """Run PDF extraction off the event loop and persist its immutable result."""

    def __init__(
        self,
        repository: DocumentRepository,
        extractor: PdfExtractor | None = None,
    ) -> None:
        self._repository = repository
        self._extractor = extractor or PdfExtractor()

    async def extract_and_persist(
        self,
        *,
        version_id: UUID,
        source: bytes | BinaryIO,
    ) -> ExtractedDocument:
        try:
            result = await asyncio.to_thread(self._extractor.extract, source)
        except PdfExtractionError as exc:
            await self._repository.mark_failed(
                version_id,
                failure_code=exc.code.value,
                failure_message=str(exc),
                extraction_metadata=exc.metadata,
            )
            raise

        paragraphs = [
            NewSourceParagraph(
                page_number=block.page_number,
                paragraph_number=block.paragraph_number,
                kind=block.kind,
                original_text=block.original_text,
                normalized_text=block.normalized_text,
                metadata={
                    **block.metadata,
                    "bounding_box": block.bounding_box.model_dump(mode="json"),
                    "extraction_version": result.metadata.extraction_version,
                    "normalization_version": result.metadata.normalization_version,
                },
            )
            for block in result.paragraphs
        ]
        await self._repository.persist_extraction(
            version_id=version_id,
            paragraphs=paragraphs,
            extraction_metadata=result.metadata.model_dump(mode="json"),
        )
        return result
