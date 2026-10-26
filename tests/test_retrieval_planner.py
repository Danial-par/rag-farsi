from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

import pytest
from langchain_core.messages import BaseMessage

from rag_assistant.retrieval.planner import (
    OpenRouterRetrievalPlanner,
    RankedRetrieval,
)


class Candidate:
    def __init__(self, *, chunk_id, version_id, section_id) -> None:
        self.chunk_id = chunk_id
        self.document_version_id = version_id
        self.normalized_text = "متن مستند"
        self.section_path = ({"id": str(section_id), "title": "بخش اصلی", "level": 1},)


class FakePlannerModel:
    def __init__(self, parsed: RankedRetrieval) -> None:
        self.parsed = parsed
        self.calls = 0

    def with_structured_output(
        self, _schema: type[RankedRetrieval], *, include_raw: bool
    ) -> FakePlannerModel:
        assert include_raw is True
        return self

    async def ainvoke(self, _messages: Sequence[BaseMessage]) -> dict[str, object]:
        self.calls += 1
        return {"parsed": self.parsed}


@pytest.mark.asyncio
async def test_planner_reranks_only_supplied_chunks_and_maps_section_to_version() -> None:
    version_id = uuid4()
    section_id = uuid4()
    first = Candidate(chunk_id=uuid4(), version_id=version_id, section_id=section_id)
    second = Candidate(chunk_id=uuid4(), version_id=version_id, section_id=section_id)
    model = FakePlannerModel(
        RankedRetrieval(chunk_aliases=["E2", "E1"], expand_section_aliases=["S1"])
    )

    plan = await OpenRouterRetrievalPlanner(model).plan(
        question="همه اجزای اصلی کدام‌اند؟",
        candidates=[first, second],  # type: ignore[list-item]
        top_k=2,
    )

    assert plan.chunk_ids == (second.chunk_id, first.chunk_id)
    assert plan.section_selections == ((version_id, section_id),)
    assert model.calls == 1


@pytest.mark.asyncio
async def test_planner_retries_unknown_alias_once_then_fails_closed() -> None:
    candidate = Candidate(chunk_id=uuid4(), version_id=uuid4(), section_id=uuid4())
    model = FakePlannerModel(RankedRetrieval(chunk_aliases=["E9"], expand_section_aliases=["S9"]))

    with pytest.raises(ValueError, match="unknown or duplicate aliases"):
        await OpenRouterRetrievalPlanner(model).plan(
            question="پرسش",
            candidates=[candidate],  # type: ignore[list-item]
            top_k=1,
        )

    assert model.calls == 2
