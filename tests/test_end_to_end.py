from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from rag_assistant.assistant import (
    AnswerCandidate,
    EvidenceAnswerValidator,
    NoopNodeObserver,
    RAGDependencies,
    SqlAlchemyWorkflowPersistence,
    create_rag_workflow,
)
from rag_assistant.core.config import Settings
from rag_assistant.db.models import AnswerOutcome
from rag_assistant.indexing import DocumentIndexingService
from rag_assistant.indexing.storage import LocalObjectStore
from rag_assistant.indexing.worker import DocumentIngestionWorker
from rag_assistant.retrieval import (
    ExpandedRetrievalContext,
    RankedCandidates,
    RetrievalCandidates,
    RetrievalDecision,
    RetrievalRequest,
    RetrievalResult,
    RetrievalService,
)

PASSWORD = "correct horse battery staple"
SAMPLE_PDF = Path(__file__).parent / "fixtures" / "documents" / "sample_doc.pdf"


class DeterministicEmbeddingClient:
    """Use identical vectors for indexing and query retrieval without network access."""

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int,
    ) -> list[list[float]]:
        del model
        return [[1.0, *([0.0] * (dimensions - 1))] for _ in texts]


class SmallTokenCounter:
    def count(self, text: str) -> int:
        return max(1, len(text.split()))


class SessionRetriever:
    def __init__(self, application: FastAPI, embeddings: DeterministicEmbeddingClient) -> None:
        self.application = application
        self.embeddings = embeddings

    async def retrieve_candidates(self, request: RetrievalRequest) -> RetrievalCandidates:
        async with self.application.state.session_factory() as session:
            return await RetrievalService(
                session=session,
                embedding_client=self.embeddings,
                settings=self.application.state.settings,
            ).retrieve_candidates(request)

    async def rerank_candidates(self, candidates: RetrievalCandidates) -> RankedCandidates:
        async with self.application.state.session_factory() as session:
            service = RetrievalService(
                session=session,
                embedding_client=self.embeddings,
                settings=self.application.state.settings,
            )
            return await service.rerank_candidates(candidates)

    async def assess_context(
        self,
        *,
        question: str,
        candidates: RetrievalCandidates,
        ranked: RankedCandidates,
    ) -> RetrievalDecision:
        async with self.application.state.session_factory() as session:
            service = RetrievalService(
                session=session,
                embedding_client=self.embeddings,
                settings=self.application.state.settings,
            )
            return await service.assess_context(
                question=question,
                candidates=candidates,
                ranked=ranked,
            )

    async def validate_scope(
        self,
        *,
        request: RetrievalRequest,
        ranked: RankedCandidates,
        decision: RetrievalDecision,
    ) -> None:
        async with self.application.state.session_factory() as session:
            await RetrievalService(
                session=session,
                embedding_client=self.embeddings,
                settings=self.application.state.settings,
            ).validate_scope(request=request, ranked=ranked, decision=decision)

    async def expand_context(
        self,
        *,
        request: RetrievalRequest,
        candidates: RetrievalCandidates,
        ranked: RankedCandidates,
        decision: RetrievalDecision,
    ) -> ExpandedRetrievalContext:
        async with self.application.state.session_factory() as session:
            return await RetrievalService(
                session=session,
                embedding_client=self.embeddings,
                settings=self.application.state.settings,
            ).expand_context(
                request=request,
                candidates=candidates,
                ranked=ranked,
                decision=decision,
            )


class GroundedGenerator:
    async def generate(
        self,
        *,
        question: str,
        history: tuple[Any, ...],
        evidence: RetrievalResult,
    ) -> AnswerCandidate:
        del history
        if "ناموجود" in question:
            return AnswerCandidate(
                content="شواهد کافی برای پاسخ یافت نشد.",
                outcome=AnswerOutcome.ABSTAINED,
            )
        chunk_id = evidence.evidence[0].chunk_id
        return AnswerCandidate(
            content="RAG اطلاعات مرتبط را بازیابی می‌کند و پاسخ را بر پایه آن می‌سازد. [1]",
            outcome=AnswerOutcome.FULLY_ANSWERED,
            cited_chunk_ids=(chunk_id,),
        )


async def test_registered_upload_worker_grounded_chat_abstention_and_owner_isolation(
    database_client: AsyncClient,
    application: FastAPI,
    settings: Settings,
    tmp_path: Path,
) -> None:
    registration = await database_client.post(
        "/api/v1/auth/register",
        json={"email": "end-to-end@example.com", "password": PASSWORD},
    )
    assert registration.status_code == 201
    store = LocalObjectStore(tmp_path)
    application.state.object_store = store
    upload = await database_client.post(
        "/api/v1/documents?title=Persian%20RAG%20guide",
        content=SAMPLE_PDF.read_bytes(),
        headers={"Content-Type": "application/pdf"},
    )
    assert upload.status_code == 202
    document_id = upload.json()["id"]

    embeddings = DeterministicEmbeddingClient()
    indexing = DocumentIndexingService(
        session_factory=application.state.session_factory,
        object_store=store,
        embedding_client=embeddings,
        settings=settings,
        token_counter_factory=lambda _model: SmallTokenCounter(),
    )
    worker = DocumentIngestionWorker(
        session_factory=application.state.session_factory,
        object_store=store,
        indexing_service=indexing,
        settings=settings,
    )
    assert await worker.run_once() is True
    indexed_document = await database_client.get(f"/api/v1/documents/{document_id}")
    assert indexed_document.status_code == 200
    assert indexed_document.json()["active_version_id"] is not None
    version = indexed_document.json()["versions"][0]
    assert version["status"] == "succeeded"

    conversation = await database_client.post("/api/v1/conversations", json={})
    conversation_id = conversation.json()["id"]
    selected = await database_client.put(
        f"/api/v1/conversations/{conversation_id}/documents",
        json={"document_ids": [document_id]},
    )
    assert selected.status_code == 200

    application.state.rag_workflow = create_rag_workflow(
        RAGDependencies(
            retriever=SessionRetriever(application, embeddings),
            generator=GroundedGenerator(),
            validator=EvidenceAnswerValidator(),
            persistence=SqlAlchemyWorkflowPersistence(application.state.session_factory),
            observer=NoopNodeObserver(),
        )
    )
    grounded = await database_client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "مراحل اصلی RAG چیست؟"},
        headers={"Idempotency-Key": "end-to-end-grounded"},
    )
    assert grounded.status_code == 200
    answer = grounded.json()
    assert answer["answer_outcome"] == "fully_answered"
    assert len(answer["citations"]) == 1
    citation = answer["citations"][0]
    assert citation["chunk_id"]
    assert citation["chunk_text"]
    assert citation["page_start"] >= 1
    assert citation["page_end"] >= citation["page_start"]

    abstained = await database_client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "سند درباره موضوع ناموجود چه می‌گوید؟"},
        headers={"Idempotency-Key": "end-to-end-abstain"},
    )
    assert abstained.status_code == 200
    assert abstained.json()["answer_outcome"] == "abstained"
    assert abstained.json()["citations"] == []

    async with AsyncClient(
        transport=ASGITransport(app=application), base_url="http://test"
    ) as other_client:
        other_registration = await other_client.post(
            "/api/v1/auth/register",
            json={"email": "end-to-end-other@example.com", "password": PASSWORD},
        )
        assert other_registration.status_code == 201
        hidden_conversation = await other_client.get(f"/api/v1/conversations/{conversation_id}")
        assert hidden_conversation.status_code == 404
        other_conversation = await other_client.post("/api/v1/conversations", json={})
        isolated = await other_client.post(
            f"/api/v1/conversations/{other_conversation.json()['id']}/messages",
            json={"content": "مراحل اصلی RAG چیست؟"},
            headers={"Idempotency-Key": "end-to-end-isolation"},
        )
        assert isolated.status_code == 200
        assert isolated.json()["answer_outcome"] == "abstained"
        assert isolated.json()["citations"] == []
