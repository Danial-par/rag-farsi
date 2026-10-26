from __future__ import annotations

import json
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from langchain_core.messages import BaseMessage

from rag_assistant.db.models import ParagraphKind, SourceParagraph
from rag_assistant.indexing.structure import (
    AnnotatedParagraph,
    OpenRouterStructureAnnotator,
    StructureAnnotationResponse,
    heuristic_heading_levels,
)


class AliasAwareModel:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def with_structured_output(
        self, _schema: type[StructureAnnotationResponse], *, include_raw: bool
    ) -> AliasAwareModel:
        assert include_raw is True
        return self

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> dict[str, object]:
        payload = json.loads(str(messages[-1].content))
        paragraphs = payload["paragraphs"]
        self.batches.append([item["alias"] for item in paragraphs])
        return {
            "parsed": StructureAnnotationResponse(
                paragraphs=[
                    AnnotatedParagraph(
                        alias=item["alias"],
                        heading_level=(1 if item["text"].startswith(("عنوان", "•")) else None),
                    )
                    for item in paragraphs
                ]
            )
        }


def _paragraph(index: int, *, kind: ParagraphKind, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        page_number=1 + index // 10,
        kind=kind,
        paragraph_metadata={"font_size": 12.0, "line_count": 1},
        original_text=text,
        normalized_text=text,
    )


@pytest.mark.asyncio
async def test_structure_annotation_uses_local_aliases_and_keeps_citation_ids_fixed() -> None:
    paragraphs = [
        _paragraph(
            index,
            kind=ParagraphKind.HEADING if index == 0 else ParagraphKind.PROSE,
            text="عنوان بخش" if index == 0 else f"متن پاراگراف {index}",
        )
        for index in range(34)
    ]
    paragraphs[2] = _paragraph(2, kind=ParagraphKind.LIST_ITEM, text="• مورد مستقل")
    model = AliasAwareModel()

    labels = await OpenRouterStructureAnnotator(model).annotate(
        cast(Sequence[SourceParagraph], paragraphs)
    )

    assert model.batches == [
        [f"P{index}" for index in range(1, 33)],
        ["P1", "P2"],
    ]
    assert labels[paragraphs[0].id] == 1
    assert paragraphs[1].id in labels and labels[paragraphs[1].id] is None
    assert paragraphs[2].id not in labels
    assert set(labels) == {
        item.id for item in paragraphs if item.kind is not ParagraphKind.LIST_ITEM
    }


def test_heuristic_heading_levels_preserve_detected_hierarchy() -> None:
    top = _paragraph(0, kind=ParagraphKind.HEADING, text="1. معماری")
    nested = _paragraph(1, kind=ParagraphKind.HEADING, text="1.2. بازیابی")
    body = _paragraph(2, kind=ParagraphKind.PROSE, text="متن توضیحی")

    labels = heuristic_heading_levels(cast(Sequence[SourceParagraph], [top, nested, body]))

    assert labels == {top.id: 1, nested.id: 2, body.id: None}
