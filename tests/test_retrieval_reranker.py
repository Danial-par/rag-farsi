from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest

from rag_assistant.db.models import ParagraphKind
from rag_assistant.retrieval import (
    CitationSource,
    EvidenceChunk,
    RerankerError,
    RetrievalCandidates,
    TextEmbeddingsInferenceReranker,
)


def _candidates() -> RetrievalCandidates:
    version_id, document_id = uuid4(), uuid4()
    evidence = []
    for index, text in enumerate(("مطلب اول", "مطلب دوم"), start=1):
        source_id = uuid4()
        evidence.append(
            EvidenceChunk(
                chunk_id=uuid4(),
                document_id=document_id,
                document_version_id=version_id,
                document_title="راهنما",
                chunk_ordinal=index,
                normalized_text=text,
                token_count=4,
                cosine_similarity=0.9 - index / 10,
                citation_sources=(
                    CitationSource(
                        source_paragraph_id=source_id,
                        document_id=document_id,
                        document_version_id=version_id,
                        document_title="راهنما",
                        page_number=1,
                        paragraph_number=index,
                        kind=ParagraphKind.PROSE,
                        original_text=text,
                        normalized_text=text,
                    ),
                ),
                section_path=({"id": str(uuid4()), "title": f"زیربخش {index}", "level": 1},),
            )
        )
    return RetrievalCandidates(
        normalized_query="پرسش فارسی",
        embedding_model="test-embedding",
        embedding_dimensions=1024,
        final_top_k=1,
        candidates=tuple(evidence),
    )


@pytest.mark.asyncio
async def test_tei_cross_encoder_reorders_candidates_and_keeps_original_evidence() -> None:
    requests: list[httpx.Request] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=[{"index": 1, "score": 0.92}, {"index": 0, "score": 0.13}],
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    reranker = TextEmbeddingsInferenceReranker(
        base_url="http://reranker.local",
        model="BAAI/bge-reranker-v2-m3",
        api_key="test-reranker-key",
        client=client,
    )
    candidates = _candidates()

    result = await reranker.rerank(
        question="پرسش اصلی",
        candidates=candidates,
        top_k=1,
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].evidence.chunk_id == candidates.candidates[1].chunk_id
    assert result.candidates[0].score == pytest.approx(0.92)
    assert (
        result.candidates[0].evidence.citation_sources == candidates.candidates[1].citation_sources
    )
    assert requests[0].url.path == "/rerank"
    assert requests[0].headers["authorization"] == "Bearer test-reranker-key"
    body = json.loads(requests[0].content)
    assert body["query"] == "پرسش اصلی"
    assert body["texts"][0].startswith("زیربخش 1\n")
    assert body["truncate"] is True

    await client.aclose()


@pytest.mark.asyncio
async def test_tei_cross_encoder_rejects_duplicate_and_non_finite_rankings() -> None:
    async def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"index": 0, "score": 1.0}, {"index": 0, "score": float("nan")}],
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    reranker = TextEmbeddingsInferenceReranker(
        base_url="http://reranker.local",
        model="BAAI/bge-reranker-v2-m3",
        client=client,
    )
    with pytest.raises(RerankerError):
        await reranker.rerank(question="پرسش", candidates=_candidates(), top_k=2)
    await client.aclose()
