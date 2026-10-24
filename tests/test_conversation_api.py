from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from rag_assistant.api.auth import CSRF_COOKIE_NAME
from rag_assistant.assistant import (
    AnswerCandidate,
    EvidenceAnswerValidator,
    NoopNodeObserver,
    RAGDependencies,
    RAGGraphInput,
    RAGGraphOutput,
    RAGWorkflow,
    SqlAlchemyWorkflowPersistence,
    create_rag_workflow,
)
from rag_assistant.assistant.models import ConversationHistoryMessage
from rag_assistant.db.base import utc_now
from rag_assistant.db.models import AnswerOutcome, Message, MessageStatus
from rag_assistant.db.repositories import (
    DocumentRepository,
    NewDocumentVersion,
)
from rag_assistant.retrieval import RetrievalRequest, RetrievalResult

PASSWORD = "correct horse battery staple"


async def _register(client: AsyncClient, email: str) -> dict[str, Any]:
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": PASSWORD},
    )
    assert response.status_code == 201
    return cast(dict[str, Any], response.json())


def _csrf_header(client: AsyncClient) -> dict[str, str]:
    return {"X-CSRF-Token": client.cookies[CSRF_COOKIE_NAME]}


async def _active_document(session: AsyncSession, owner_id: UUID) -> UUID:
    repository = DocumentRepository(session)
    document = await repository.create_owned(owner_id=owner_id, title="راهنما")
    version = await repository.create_owned_version(
        owner_id=owner_id,
        document_id=document.id,
        values=NewDocumentVersion(
            object_key="documents/guide.pdf",
            content_hash="a" * 64,
            file_size_bytes=1024,
            embedding_model="openai/text-embedding-3-large",
            embedding_dimensions=1024,
        ),
    )
    await repository.mark_processing(version.id)
    await repository.mark_succeeded(version.id, extraction_metadata={})
    await repository.activate_owned_version(
        owner_id=owner_id,
        document_id=document.id,
        version_id=version.id,
    )
    return document.id


class EmptyRetriever:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        del request
        if self.failure is not None:
            raise self.failure
        return RetrievalResult(
            normalized_query="",
            embedding_model="openai/text-embedding-3-large",
            embedding_dimensions=1024,
            evidence=(),
        )


class UnusedGenerator:
    async def generate(
        self,
        *,
        question: str,
        history: tuple[ConversationHistoryMessage, ...],
        evidence: RetrievalResult,
    ) -> AnswerCandidate:
        del question, history, evidence
        raise AssertionError("The generator should not run for empty evidence")


class StaticWorkflow:
    def __init__(self, output: RAGGraphOutput) -> None:
        self.output = output

    async def ainvoke(self, graph_input: RAGGraphInput) -> RAGGraphOutput:
        del graph_input
        return self.output


def _workflow(application: FastAPI, *, retrieval_failure: Exception | None = None) -> RAGWorkflow:
    return create_rag_workflow(
        RAGDependencies(
            retriever=EmptyRetriever(retrieval_failure),
            generator=UnusedGenerator(),
            validator=EvidenceAnswerValidator(),
            persistence=SqlAlchemyWorkflowPersistence(application.state.session_factory),
            observer=NoopNodeObserver(),
        )
    )


async def test_conversation_api_requires_authentication_and_csrf(
    database_client: AsyncClient,
) -> None:
    denied = await database_client.get("/api/v1/conversations")
    assert denied.status_code == 401

    await _register(database_client, "chat-auth@example.com")
    missing_csrf = await database_client.post("/api/v1/conversations", json={})
    assert missing_csrf.status_code == 403

    created = await database_client.post(
        "/api/v1/conversations", json={"title": "پرسش‌های من"}, headers=_csrf_header(database_client)
    )
    assert created.status_code == 201
    assert created.json()["title"] == "پرسش‌های من"


async def test_conversation_read_is_owner_scoped_and_lists_with_cursor(
    database_client: AsyncClient,
) -> None:
    await _register(database_client, "chat-owner@example.com")
    first = await database_client.post(
        "/api/v1/conversations", json={}, headers=_csrf_header(database_client)
    )
    second = await database_client.post(
        "/api/v1/conversations", json={}, headers=_csrf_header(database_client)
    )
    assert first.status_code == second.status_code == 201
    first_id = first.json()["id"]

    listed = await database_client.get("/api/v1/conversations?limit=1")
    assert listed.status_code == 200
    assert listed.json()["next_cursor"]

    await _register(database_client, "chat-other@example.com")
    hidden = await database_client.get(f"/api/v1/conversations/{first_id}")
    assert hidden.status_code == 404


async def test_conversation_rename_persists_and_is_owner_scoped(
    database_client: AsyncClient,
) -> None:
    await _register(database_client, "chat-rename-owner@example.com")
    created = await database_client.post(
        "/api/v1/conversations", json={}, headers=_csrf_header(database_client)
    )
    conversation_id = created.json()["id"]
    missing_csrf = await database_client.patch(
        f"/api/v1/conversations/{conversation_id}", json={"title": "تغییر نام"}
    )
    assert missing_csrf.status_code == 403

    renamed = await database_client.patch(
        f"/api/v1/conversations/{conversation_id}",
        json={"title": "  پژوهش‌های من  "},
        headers=_csrf_header(database_client),
    )

    assert renamed.status_code == 200
    assert renamed.json()["title"] == "پژوهش‌های من"
    assert renamed.json()["updated_at"] >= created.json()["updated_at"]
    persisted = await database_client.get(f"/api/v1/conversations/{conversation_id}")
    assert persisted.status_code == 200
    assert persisted.json()["title"] == "پژوهش‌های من"

    await _register(database_client, "chat-rename-other@example.com")
    forbidden = await database_client.patch(
        f"/api/v1/conversations/{conversation_id}",
        json={"title": "نام دیگر"},
        headers=_csrf_header(database_client),
    )
    assert forbidden.status_code == 404


async def test_conversation_rename_rejects_blank_or_oversized_title(
    database_client: AsyncClient,
) -> None:
    await _register(database_client, "chat-rename-validation@example.com")
    created = await database_client.post(
        "/api/v1/conversations", json={}, headers=_csrf_header(database_client)
    )
    conversation_id = created.json()["id"]
    headers = _csrf_header(database_client)

    blank = await database_client.patch(
        f"/api/v1/conversations/{conversation_id}", json={"title": "  "}, headers=headers
    )
    oversized = await database_client.patch(
        f"/api/v1/conversations/{conversation_id}", json={"title": "x" * 501}, headers=headers
    )

    assert blank.status_code == oversized.status_code == 422


async def test_document_selection_replaces_explicit_ids_and_checks_access(
    database_client: AsyncClient,
    database_connection: AsyncConnection,
) -> None:
    registered = await _register(database_client, "chat-docs@example.com")
    user_id = UUID(registered["id"])
    async with AsyncSession(bind=database_connection, expire_on_commit=False) as session:
        document_id = await _active_document(session, user_id)
        await session.commit()

    conversation = await database_client.post(
        "/api/v1/conversations", json={}, headers=_csrf_header(database_client)
    )
    selected = await database_client.put(
        f"/api/v1/conversations/{conversation.json()['id']}/documents",
        json={"document_ids": [str(document_id)]},
        headers=_csrf_header(database_client),
    )
    assert selected.status_code == 200
    assert [item["id"] for item in selected.json()] == [str(document_id)]

    replaced = await database_client.put(
        f"/api/v1/conversations/{conversation.json()['id']}/documents",
        json={"document_ids": []},
        headers=_csrf_header(database_client),
    )
    assert replaced.status_code == 200
    assert replaced.json() == []

    inaccessible = await database_client.put(
        f"/api/v1/conversations/{conversation.json()['id']}/documents",
        json={"document_ids": [str(UUID(int=1))]},
        headers=_csrf_header(database_client),
    )
    assert inaccessible.status_code == 404


async def test_conversation_includes_active_shared_default_document(
    database_client: AsyncClient,
    database_connection: AsyncConnection,
) -> None:
    registered = await _register(database_client, "chat-default@example.com")
    async with AsyncSession(bind=database_connection, expire_on_commit=False) as session:
        repository = DocumentRepository(session)
        document = await repository.create_shared(title="پیش‌فرض عمومی", default_for_all_users=True)
        version = await repository.create_shared_version(
            document_id=document.id,
            values=NewDocumentVersion(
                object_key="shared/default.pdf",
                content_hash="c" * 64,
                file_size_bytes=2048,
                embedding_model="openai/text-embedding-3-large",
                embedding_dimensions=1024,
            ),
        )
        await repository.mark_processing(version.id)
        await repository.mark_succeeded(version.id, extraction_metadata={})
        await repository.activate_shared_version(document_id=document.id, version_id=version.id)
        await session.commit()

    response = await database_client.post(
        "/api/v1/conversations", json={}, headers=_csrf_header(database_client)
    )
    assert response.status_code == 201
    assert response.json()["selected_documents"] == [
        {"id": str(document.id), "title": "پیش‌فرض عمومی"}
    ]
    assert registered["id"]


async def test_chat_serializes_partial_answer_and_whole_paragraph_citation(
    database_client: AsyncClient,
    application: FastAPI,
) -> None:
    await _register(database_client, "chat-citation@example.com")
    conversation = await database_client.post(
        "/api/v1/conversations", json={}, headers=_csrf_header(database_client)
    )
    citation = {
        "source_paragraph_id": "10000000-0000-0000-0000-000000000001",
        "document_id": "20000000-0000-0000-0000-000000000002",
        "document_version_id": "30000000-0000-0000-0000-000000000003",
        "document_title": "راهنمای فارسی",
        "page_number": 3,
        "paragraph_number": 2,
        "original_text": "متن اصلی پاراگراف",
    }
    application.state.rag_workflow = StaticWorkflow(
        RAGGraphOutput(
            assistant_message_id=UUID("40000000-0000-0000-0000-000000000004"),
            status=MessageStatus.COMPLETED,
            outcome=AnswerOutcome.PARTIALLY_ANSWERED,
            content="پاسخ مستند [1]\n\nبخش پشتیبانی‌نشده: مقدار دقیق",
            citation_snapshot=(citation,),
            created_at=utc_now(),
        )
    )
    response = await database_client.post(
        f"/api/v1/conversations/{conversation.json()['id']}/messages",
        json={"content": "پرسش فارسی"},
        headers={**_csrf_header(database_client), "Idempotency-Key": "citation-1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["execution_status"] == "completed"
    assert body["answer_outcome"] == "partially_answered"
    assert body["citations"][0] == {
        **citation,
        "chunk_id": None,
        "chunk_ordinal": None,
        "chunk_text": None,
        "page_start": None,
        "page_end": None,
        "start_offset": None,
        "end_offset": None,
    }


async def test_chat_persists_abstention_history_and_replays_idempotently(
    database_client: AsyncClient,
    application: FastAPI,
    database_connection: AsyncConnection,
) -> None:
    await _register(database_client, "chat-retry@example.com")
    conversation = await database_client.post(
        "/api/v1/conversations", json={}, headers=_csrf_header(database_client)
    )
    conversation_id = conversation.json()["id"]
    application.state.rag_workflow = _workflow(application)
    headers = {**_csrf_header(database_client), "Idempotency-Key": "question-1"}
    payload = {"content": "این سند چه می‌گوید؟"}

    answer = await database_client.post(
        f"/api/v1/conversations/{conversation_id}/messages", json=payload, headers=headers
    )
    history_after_first = await database_client.get(
        f"/api/v1/conversations/{conversation_id}/messages"
    )
    async with AsyncSession(bind=database_connection) as session:
        after_first = list(await session.scalars(select(Message)))
    assert len(history_after_first.json()["items"]) == 2, history_after_first.json()
    assert len(after_first) == 2, [
        (message.conversation_id, message.role, message.idempotency_key) for message in after_first
    ]
    retry = await database_client.post(
        f"/api/v1/conversations/{conversation_id}/messages", json=payload, headers=headers
    )
    conflicting_retry = await database_client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "پرسش دیگری"},
        headers=headers,
    )
    assert answer.status_code == retry.status_code == 200
    assert conflicting_retry.status_code == 409
    async with AsyncSession(bind=database_connection) as session:
        stored = list(
            await session.scalars(select(Message).where(Message.idempotency_key == "question-1"))
        )
    assert len(stored) == 2, [(message.role, message.idempotency_key) for message in stored]
    assert answer.json()["id"] == retry.json()["id"]
    assert answer.json()["execution_status"] == "completed"
    assert answer.json()["answer_outcome"] == "abstained"
    assert answer.json()["citations"] == []

    history = await database_client.get(f"/api/v1/conversations/{conversation_id}/messages?limit=1")
    assert history.status_code == 200
    assert history.json()["next_cursor"]
    next_page = await database_client.get(
        f"/api/v1/conversations/{conversation_id}/messages?limit=1&cursor={history.json()['next_cursor']}"
    )
    assert next_page.status_code == 200
    assert history.json()["items"][0]["id"] != next_page.json()["items"][0]["id"]


async def test_chat_failure_returns_persisted_assistant_message_id(
    database_client: AsyncClient,
    application: FastAPI,
) -> None:
    await _register(database_client, "chat-failure@example.com")
    conversation = await database_client.post(
        "/api/v1/conversations", json={}, headers=_csrf_header(database_client)
    )
    application.state.rag_workflow = _workflow(
        application, retrieval_failure=RuntimeError("private provider details")
    )
    response = await database_client.post(
        f"/api/v1/conversations/{conversation.json()['id']}/messages",
        json={"content": "پرسش"},
        headers={**_csrf_header(database_client), "Idempotency-Key": "failed-once"},
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["execution_status"] == "failed"
    assert detail["id"]
    assert "private provider details" not in str(detail)
