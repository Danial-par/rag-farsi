"""Cross-encoder reranking through a Hugging Face TEI endpoint."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import httpx
from langfuse import get_client, observe

from rag_assistant.core.config import Settings
from rag_assistant.retrieval.models import (
    EvidenceChunk,
    RankedCandidate,
    RankedCandidates,
    RetrievalCandidates,
)


class CandidateReranker(Protocol):
    async def rerank(
        self,
        *,
        question: str,
        candidates: RetrievalCandidates,
        top_k: int,
    ) -> RankedCandidates: ...


class RerankerError(RuntimeError):
    """The configured cross-encoder could not return a valid candidate ordering."""


class TextEmbeddingsInferenceReranker:
    """Score vector candidates with TEI's multilingual cross-encoder endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 30,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    @classmethod
    def from_settings(cls, settings: Settings) -> TextEmbeddingsInferenceReranker:
        return cls(
            base_url=str(settings.reranker_base_url),
            model=settings.reranker_model,
            timeout_seconds=settings.reranker_timeout_seconds,
            api_key=(
                settings.reranker_api_key.get_secret_value()
                if settings.reranker_api_key is not None
                else None
            ),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @observe(name="rag.retrieval.cross_encoder_rerank", capture_input=False, capture_output=False)
    async def rerank(
        self,
        *,
        question: str,
        candidates: RetrievalCandidates,
        top_k: int,
    ) -> RankedCandidates:
        if not candidates.candidates:
            return RankedCandidates(candidates=())

        passages = [_passage_text(candidate) for candidate in candidates.candidates]
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            response = await self._client.post(
                f"{self._base_url}/rerank",
                json={
                    "query": question,
                    "texts": passages,
                    "truncate": True,
                    "truncation_direction": "right",
                    "raw_scores": True,
                },
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RerankerError("The cross-encoder request failed") from exc

        ranked = _parse_rankings(payload, candidates.candidates)
        selected = tuple(ranked[:top_k])
        get_client().update_current_span(
            metadata={
                "reranker_model": self._model,
                "candidate_count": len(passages),
                "selected_count": len(selected),
            }
        )
        return RankedCandidates(candidates=selected)


def _passage_text(candidate: EvidenceChunk) -> str:
    headings = [
        str(item["title"])
        for item in candidate.section_path
        if isinstance(item, dict) and isinstance(item.get("title"), str)
    ]
    return "\n".join((*headings, candidate.normalized_text))


def _parse_rankings(
    payload: object,
    candidates: Sequence[EvidenceChunk],
) -> list[RankedCandidate]:
    if not isinstance(payload, list) or len(payload) != len(candidates):
        raise RerankerError("The cross-encoder returned an incomplete ranking")
    ranks: list[tuple[int, float]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise RerankerError("The cross-encoder returned an invalid ranking item")
        index = item.get("index")
        score = item.get("score")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(candidates)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise RerankerError("The cross-encoder returned an invalid ranking score")
        ranks.append((index, float(score)))
    if len({index for index, _ in ranks}) != len(candidates):
        raise RerankerError("The cross-encoder ranking must contain each candidate once")
    ranks.sort(key=lambda item: (-item[1], item[0]))
    return [RankedCandidate(evidence=candidates[index], score=score) for index, score in ranks]
