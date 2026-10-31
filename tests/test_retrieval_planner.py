from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

import pytest
from langchain_core.messages import BaseMessage
from pydantic import ValidationError

from rag_assistant.db.models import ParagraphKind
from rag_assistant.retrieval import (
    CitationSource,
    EvidenceChunk,
    RankedCandidate,
    RankedCandidates,
    SectionOutlineItem,
)
from rag_assistant.retrieval.models import RetrievalDecision
from rag_assistant.retrieval.planner import (
    ContextAssessment,
    OpenRouterContextPlanner,
    RetrievalPlannerOutputError,
    _context_assessment_schema,
)


class FakePlannerModel:
    def __init__(self, parsed: ContextAssessment) -> None:
        self.parsed = parsed
        self.calls = 0
        self.schema: type[ContextAssessment] | None = None

    def with_structured_output(
        self, schema: type[ContextAssessment], *, include_raw: bool
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


def _ranked() -> RankedCandidates:
    chunk_id = uuid4()
    doc_id = uuid4()
    version_id = uuid4()
    source_id = uuid4()
    evidence = EvidenceChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        document_version_id=version_id,
        document_title="راهنما",
        chunk_ordinal=1,
        normalized_text="متن مرتبط",
        token_count=4,
        cosine_similarity=0.9,
        citation_sources=(
            CitationSource(
                source_paragraph_id=source_id,
                document_id=doc_id,
                document_version_id=version_id,
                document_title="راهنما",
                page_number=1,
                paragraph_number=1,
                kind=ParagraphKind.PROSE,
                original_text="متن مرتبط",
                normalized_text="متن مرتبط",
            ),
        ),
    )
    return RankedCandidates(candidates=(RankedCandidate(evidence=evidence, score=0.8),))


def _outline() -> tuple[SectionOutlineItem, ...]:
    version_id = uuid4()
    parent_id, first_id, second_id = uuid4(), uuid4(), uuid4()
    return (
        SectionOutlineItem(
            document_version_id=version_id,
            section_id=parent_id,
            document_title="RAG Handbook",
            title="اجزای RAG",
            level=1,
            heading_path=("اجزای RAG",),
        ),
        SectionOutlineItem(
            document_version_id=version_id,
            section_id=first_id,
            parent_section_id=parent_id,
            document_title="RAG Handbook",
            title="بارگذاری اسناد",
            level=2,
            heading_path=("اجزای RAG", "بارگذاری اسناد"),
        ),
        SectionOutlineItem(
            document_version_id=version_id,
            section_id=second_id,
            parent_section_id=parent_id,
            document_title="RAG Handbook",
            title="تقسیم‌بندی متن",
            level=2,
            heading_path=("اجزای RAG", "تقسیم‌بندی متن"),
        ),
    )


@pytest.mark.asyncio
async def test_context_planner_maps_a_parent_expansion_alias_to_persisted_ids() -> None:
    outline = _outline()
    model = FakePlannerModel(ContextAssessment(action="expand_section", section_aliases=["S1"]))
    planner = OpenRouterContextPlanner(model)

    decision = await planner.assess(
        question="همه اجزای اصلی کدام‌اند؟",
        ranked=_ranked(),
        section_outline=outline,
    )

    assert decision == RetrievalDecision(
        action="expand_section",
        sections=(
            {
                "document_version_id": outline[0].document_version_id,
                "section_id": outline[0].section_id,
                "parent_section_id": None,
                "heading_path": ("اجزای RAG",),
            },
        ),
    )
    assert model.calls == 1


@pytest.mark.asyncio
async def test_context_planner_accepts_only_available_sibling_sections() -> None:
    outline = _outline()
    model = FakePlannerModel(
        ContextAssessment(action="fetch_siblings", section_aliases=["S2", "S3"])
    )

    decision = await OpenRouterContextPlanner(model).assess(
        question="دو زیربخش مرتبط را توضیح بده",
        ranked=_ranked(),
        section_outline=outline,
    )

    assert decision.action == "fetch_siblings"
    assert tuple(item.section_id for item in decision.sections) == (
        outline[1].section_id,
        outline[2].section_id,
    )


@pytest.mark.asyncio
async def test_context_planner_rejects_unknown_aliases_after_one_retry() -> None:
    model = FakePlannerModel(ContextAssessment(action="expand_section", section_aliases=["S99"]))
    with pytest.raises(RetrievalPlannerOutputError, match="invalid structured output") as error:
        await OpenRouterContextPlanner(model).assess(
            question="پرسش",
            ranked=_ranked(),
            section_outline=_outline(),
        )

    assert model.calls == 2
    assert error.value.failure_reason == "unknown_section_alias"


def test_context_planner_schema_enforces_section_count() -> None:
    schema = _context_assessment_schema(max_sections=2)

    assert schema.model_json_schema()["properties"]["section_aliases"]["maxItems"] == 2
    with pytest.raises(ValidationError):
        schema(action="fetch_siblings", section_aliases=["S1", "S2", "S3"])
