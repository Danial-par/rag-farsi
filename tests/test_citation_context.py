from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from rag_assistant.api.auth import CSRF_COOKIE_NAME
from rag_assistant.db.base import utc_now
from rag_assistant.db.models import (
    AnswerOutcome,
    Document,
    Message,
    MessageRole,
    MessageStatus,
    ParagraphKind,
)
from rag_assistant.db.repositories import (
    ConversationRepository,
    DocumentRepository,
    NewDocumentVersion,
    NewSourceParagraph,
)
from rag_assistant.db.repositories.citation_context import CONTEXT_NEIGHBORS


@dataclass
class CitedSection:
    owner_id: UUID
    document_id: UUID
    version_id: UUID
    conversation_id: UUID
    message_id: UUID
    source_id: UUID
    uncited_id: UUID

    @property
    def url(self) -> str:
        return (
            f"/api/v1/conversations/{self.conversation_id}/messages/{self.message_id}"
            f"/citations/{self.source_id}/context"
        )


def _block(page: int, number: int, kind: ParagraphKind, text: str) -> NewSourceParagraph:
    return NewSourceParagraph(
        page_number=page,
        paragraph_number=number,
        kind=kind,
        original_text=text,
        normalized_text=text,
    )


async def _cited_section(
    client: AsyncClient, app: FastAPI, *, shared: bool = False, long_section: bool = False
) -> CitedSection:
    registration = await client.post(
        "/api/v1/auth/register",
        json={"email": "citation-context@example.com", "password": "correct horse battery staple"},
    )
    assert registration.status_code == 201
    owner_id = UUID(registration.json()["id"])
    async with app.state.session_factory() as session, session.begin():
        documents = DocumentRepository(session)
        if shared:
            document = await documents.create_shared(
                title="راهنمای فارسی", default_for_all_users=False
            )
        else:
            document = await documents.create_owned(owner_id=owner_id, title="راهنمای فارسی")
        values = NewDocumentVersion(
            object_key="documents/citation-context.pdf",
            content_hash="a" * 64,
            file_size_bytes=100,
            embedding_model="openai/text-embedding-3-large",
            embedding_dimensions=1024,
        )
        if shared:
            version = await documents.create_shared_version(document_id=document.id, values=values)
        else:
            version = await documents.create_owned_version(
                owner_id=owner_id, document_id=document.id, values=values
            )
        await documents.mark_processing(version.id)
        if long_section:
            blocks = [_block(1, 1, ParagraphKind.HEADING, "فهرست طولانی")]
            blocks += [_block(1, i, ParagraphKind.LIST_ITEM, f"• مورد {i}") for i in range(2, 52)]
            cited_index = 25
        else:
            blocks = [
                _block(1, 1, ParagraphKind.HEADING, "بخش قبلی"),
                _block(1, 2, ParagraphKind.PROSE, "خارج از بخش"),
                _block(1, 3, ParagraphKind.HEADING, "مدل‌های رتبه‌بندی"),
                _block(1, 4, ParagraphKind.PROSE, "مدل‌های پیشنهادی عبارتند از:"),
                _block(1, 5, ParagraphKind.LIST_ITEM, "• Cohere"),
                _block(2, 1, ParagraphKind.LIST_ITEM, "• Cross-Encoder"),
                _block(2, 2, ParagraphKind.HEADING, "بخش بعدی"),
                _block(2, 3, ParagraphKind.PROSE, "خارج از بخش بعدی"),
            ]
            cited_index = 4
        paragraphs = await documents.add_source_paragraphs(version_id=version.id, paragraphs=blocks)
        await documents.mark_succeeded(version.id, extraction_metadata={})
        if shared:
            await documents.activate_shared_version(document_id=document.id, version_id=version.id)
        else:
            await documents.activate_owned_version(
                owner_id=owner_id, document_id=document.id, version_id=version.id
            )
        conversations = ConversationRepository(session)
        conversation = await conversations.create(owner_id=owner_id)
        await conversations.add_document(
            owner_id=owner_id, conversation_id=conversation.id, document_id=document.id
        )
        message = await conversations.create_pending_assistant_message(
            owner_id=owner_id, conversation_id=conversation.id
        )
        await conversations.complete_assistant_message(
            owner_id=owner_id,
            message_id=message.id,
            content="مدل پیشنهادی [1]",
            outcome=AnswerOutcome.FULLY_ANSWERED,
            cited_paragraph_ids=[paragraphs[cited_index].id],
        )
        return CitedSection(
            owner_id,
            document.id,
            version.id,
            conversation.id,
            message.id,
            paragraphs[cited_index].id,
            paragraphs[0].id,
        )


@pytest.mark.parametrize("shared", [False, True])
async def test_context_shows_section_across_pages_and_preserves_snapshot(
    database_client: AsyncClient, application: FastAPI, shared: bool
) -> None:
    citation = await _cited_section(database_client, application, shared=shared)
    history_url = f"/api/v1/conversations/{citation.conversation_id}/messages"
    before = (await database_client.get(history_url)).json()

    response = await database_client.get(citation.url)

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    context = response.json()
    assert context["heading"] == "مدل‌های رتبه‌بندی"
    assert [(p["page_number"], p["paragraph_number"]) for p in context["paragraphs"]] == [
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 1),
    ]
    assert [p["original_text"] for p in context["paragraphs"] if p["is_cited"]] == ["• Cohere"]
    assert context["truncated"] is False
    assert (await database_client.get(history_url)).json() == before


async def test_context_denies_uncited_sources_wrong_messages_and_foreign_conversations(
    database_client: AsyncClient, application: FastAPI
) -> None:
    citation = await _cited_section(database_client, application)
    for url in (
        citation.url.replace(str(citation.source_id), str(citation.uncited_id)),
        citation.url.replace(str(citation.message_id), str(uuid4())),
        citation.url.replace(str(citation.conversation_id), str(uuid4())),
    ):
        assert (await database_client.get(url)).status_code == 404
    await database_client.post(
        "/api/v1/auth/logout", headers={"X-CSRF-Token": database_client.cookies[CSRF_COOKIE_NAME]}
    )
    await database_client.post(
        "/api/v1/auth/register",
        json={"email": "another-reader@example.com", "password": "correct horse battery staple"},
    )
    assert (await database_client.get(citation.url)).status_code == 404


async def test_context_uses_historical_version_after_selection_and_active_version_change(
    database_client: AsyncClient, application: FastAPI
) -> None:
    citation = await _cited_section(database_client, application)
    async with application.state.session_factory() as session, session.begin():
        documents = DocumentRepository(session)
        newer = await documents.create_owned_version(
            owner_id=citation.owner_id,
            document_id=citation.document_id,
            values=NewDocumentVersion(
                object_key="documents/new-context.pdf",
                content_hash="b" * 64,
                file_size_bytes=100,
                embedding_model="openai/text-embedding-3-large",
                embedding_dimensions=1024,
            ),
        )
        await documents.mark_processing(newer.id)
        await documents.add_source_paragraphs(
            version_id=newer.id,
            paragraphs=[_block(1, 1, ParagraphKind.PROSE, "متن نسخه جدید")],
        )
        await documents.mark_succeeded(newer.id, extraction_metadata={})
        await documents.activate_owned_version(
            owner_id=citation.owner_id, document_id=citation.document_id, version_id=newer.id
        )
        await ConversationRepository(session).replace_documents(
            owner_id=citation.owner_id, conversation_id=citation.conversation_id, document_ids=[]
        )
    response = await database_client.get(citation.url)
    assert response.status_code == 200
    assert "متن نسخه جدید" not in response.text
    assert any(
        p["source_paragraph_id"] == str(citation.source_id) for p in response.json()["paragraphs"]
    )


async def test_context_fails_closed_for_deleted_document_and_mismatched_snapshot(
    database_client: AsyncClient, application: FastAPI
) -> None:
    citation = await _cited_section(database_client, application)
    async with application.state.session_factory() as session, session.begin():
        message = await session.get(Message, citation.message_id)
        assert message is not None
        # Simulate a malformed legacy snapshot without modifying a terminal message.
        malformed = Message(
            conversation_id=citation.conversation_id,
            role=MessageRole.ASSISTANT,
            status=MessageStatus.COMPLETED,
            answer_outcome=AnswerOutcome.FULLY_ANSWERED,
            content="پاسخ [1]",
            citation_snapshot=[{**message.citation_snapshot[0], "original_text": "متن ناسازگار"}],
        )
        session.add(malformed)
        await session.flush()
        malformed_id = malformed.id
    malformed_url = citation.url.replace(str(citation.message_id), str(malformed_id))
    assert (await database_client.get(malformed_url)).status_code == 404
    async with application.state.session_factory() as session, session.begin():
        document = await session.get(Document, citation.document_id)
        assert document is not None
        document.deleted_at = utc_now()
    assert (await database_client.get(citation.url)).status_code == 404


async def test_long_sections_return_bounded_neighbors_with_the_cited_paragraph(
    database_client: AsyncClient, application: FastAPI
) -> None:
    citation = await _cited_section(database_client, application, long_section=True)
    response = await database_client.get(citation.url)
    assert response.status_code == 200
    context = response.json()
    assert context["truncated"] is True
    assert len(context["paragraphs"]) == CONTEXT_NEIGHBORS * 2 + 1
    assert context["paragraphs"][CONTEXT_NEIGHBORS]["source_paragraph_id"] == str(
        citation.source_id
    )
    assert sum(p["is_cited"] for p in context["paragraphs"]) == 1
