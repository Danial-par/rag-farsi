from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import BaseMessage
from pydantic import ValidationError

from rag_assistant.retrieval.planner import (
    OpenRouterRetrievalPlanner,
    RankedRetrieval,
    RetrievalPlannerOutputError,
    _ranked_retrieval_schema,
)


class Candidate:
    def __init__(self, *, chunk_id: UUID, version_id: UUID, section_id: UUID) -> None:
        self.chunk_id = chunk_id
        self.document_version_id = version_id
        self.normalized_text = "متن مستند"
        self.section_path = ({"id": str(section_id), "title": "بخش اصلی", "level": 1},)


class FakePlannerModel:
    def __init__(self, parsed: RankedRetrieval) -> None:
        self.parsed = parsed
        self.calls = 0
        self.schema: type[RankedRetrieval] | None = None

    def with_structured_output(
        self, schema: type[RankedRetrieval], *, include_raw: bool
    ) -> FakePlannerModel:
        assert include_raw is True
        self.schema = schema
        return self

    async def ainvoke(self, _messages: Sequence[BaseMessage]) -> dict[str, object]:
        self.calls += 1
        assert self.schema is not None
        try:
            parsed = self.schema(**self.parsed.model_dump())
        except ValidationError:
            parsed = None
        return {"parsed": parsed}


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
        candidates=[first, second],
        top_k=2,
    )

    assert plan.chunk_ids == (second.chunk_id, first.chunk_id)
    assert plan.section_selections == ((version_id, section_id),)
    assert model.calls == 1


@pytest.mark.asyncio
async def test_planner_retries_unknown_alias_once_then_fails_closed() -> None:
    candidate = Candidate(chunk_id=uuid4(), version_id=uuid4(), section_id=uuid4())
    model = FakePlannerModel(RankedRetrieval(chunk_aliases=["E9"], expand_section_aliases=["S9"]))

    with pytest.raises(RetrievalPlannerOutputError, match="invalid structured output") as error:
        await OpenRouterRetrievalPlanner(model).plan(
            question="پرسش",
            candidates=[candidate],
            top_k=1,
        )

    assert model.calls == 2
    assert error.value.failure_reason in {
        "unknown_chunk_alias",
        "unknown_section_alias",
    }


def test_planner_schema_enforces_request_specific_candidate_limits() -> None:
    schema = _ranked_retrieval_schema(top_k=2, max_sections=1)

    assert schema.model_json_schema()["properties"]["chunk_aliases"]["maxItems"] == 2
    assert schema.model_json_schema()["properties"]["expand_section_aliases"]["maxItems"] == 1
    with pytest.raises(ValidationError):
        schema(chunk_aliases=["E1", "E2", "E3"], expand_section_aliases=[])
