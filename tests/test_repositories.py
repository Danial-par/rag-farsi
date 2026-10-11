from __future__ import annotations

from datetime import timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from rag_assistant.db.base import utc_now
from rag_assistant.db.models import AnswerOutcome, IngestionStatus, ParagraphKind, User
from rag_assistant.db.repositories import (
    AuthSessionRepository,
    ConversationRepository,
    DocumentRepository,
    InvalidPersistenceStateError,
    NewDocumentVersion,
    NewSourceParagraph,
    RecordNotFoundError,
    UserRepository,
)


def _new_version(object_key: str, content_hash_character: str = "a") -> NewDocumentVersion:
    return NewDocumentVersion(
        object_key=object_key,
        content_hash=content_hash_character * 64,
        file_size_bytes=4096,
        embedding_model="openai/text-embedding-3-large",
        embedding_dimensions=1024,
    )


async def _create_user(session: AsyncSession, email: str) -> User:
    return await UserRepository(session).create(
        email=email,
        password_hash="test-password-hash",
    )


async def _succeed_version(repository: DocumentRepository, version_id: UUID) -> None:
    await repository.mark_processing(version_id)
    await repository.mark_succeeded(version_id, extraction_metadata={})


async def test_document_queries_enforce_ownership_and_include_shared_documents(
    db_session: AsyncSession,
) -> None:
    owner = await _create_user(db_session, "owner@example.com")
    other = await _create_user(db_session, "other@example.com")
    repository = DocumentRepository(db_session)
    private = await repository.create_owned(owner_id=owner.id, title="خصوصی")
    shared = await repository.create_shared(title="نمونه همگانی", default_for_all_users=True)

    assert await repository.get_owned(owner_id=other.id, document_id=private.id) is None
    accessible = await repository.list_accessible(user_id=other.id)
    assert [document.id for document in accessible] == [shared.id]


async def test_only_successful_owned_version_can_be_activated(db_session: AsyncSession) -> None:
    owner = await _create_user(db_session, "activation@example.com")
    repository = DocumentRepository(db_session)
    document = await repository.create_owned(owner_id=owner.id, title="نسخه‌ها")
    version = await repository.create_owned_version(
        owner_id=owner.id,
        document_id=document.id,
        values=_new_version("documents/activation.pdf"),
    )

    try:
        await repository.activate_owned_version(
            owner_id=owner.id,
            document_id=document.id,
            version_id=version.id,
        )
    except InvalidPersistenceStateError:
        pass
    else:
        raise AssertionError("Pending version was activated")

    await _succeed_version(repository, version.id)
    await repository.activate_owned_version(
        owner_id=owner.id,
        document_id=document.id,
        version_id=version.id,
    )

    assert document.active_version_id == version.id
    assert version.activated_at is not None
    try:
        await repository.hard_delete_never_activated_version(
            owner_id=owner.id,
            document_id=document.id,
            version_id=version.id,
        )
    except InvalidPersistenceStateError:
        pass
    else:
        raise AssertionError("Historically active version was deleted")


async def test_shared_default_document_is_added_to_existing_and_new_conversations(
    db_session: AsyncSession,
) -> None:
    user = await _create_user(db_session, "shared@example.com")
    conversations = ConversationRepository(db_session)
    existing = await conversations.create(owner_id=user.id)

    documents = DocumentRepository(db_session)
    shared = await documents.create_shared(title="راهنمای نمونه", default_for_all_users=True)
    version = await documents.create_shared_version(
        document_id=shared.id,
        values=_new_version("shared/sample.pdf", "b"),
    )
    await _succeed_version(documents, version.id)
    await documents.activate_shared_version(document_id=shared.id, version_id=version.id)

    new = await conversations.create(owner_id=user.id)
    existing_documents = await conversations.list_active_documents(
        owner_id=user.id, conversation_id=existing.id
    )
    new_documents = await conversations.list_active_documents(
        owner_id=user.id, conversation_id=new.id
    )

    assert [document.id for document in existing_documents] == [shared.id]
    assert [document.id for document in new_documents] == [shared.id]


async def test_auth_sessions_are_revocable_and_fail_closed_for_other_users(
    db_session: AsyncSession,
) -> None:
    user = await _create_user(db_session, "session@example.com")
    repository = AuthSessionRepository(db_session)
    auth_session = await repository.create(
        user_id=user.id,
        expires_at=utc_now() + timedelta(hours=1),
        refresh_token_hash="a" * 64,
        csrf_token_hash="b" * 64,
        client_metadata={"user_agent": "tests"},
    )

    assert await repository.get_active(auth_session.id) is auth_session
    await repository.revoke(auth_session.id, reason="logout")
    assert await repository.get_active(auth_session.id) is None


async def test_completed_messages_store_validated_immutable_citation_snapshots(
    db_session: AsyncSession,
) -> None:
    user = await _create_user(db_session, "citations@example.com")
    documents = DocumentRepository(db_session)
    document = await documents.create_owned(owner_id=user.id, title="منبع فارسی")
    version = await documents.create_owned_version(
        owner_id=user.id,
        document_id=document.id,
        values=_new_version("documents/citations.pdf", "c"),
    )
    await documents.mark_processing(version.id)
    (paragraph,) = await documents.add_source_paragraphs(
        version_id=version.id,
        paragraphs=[
            NewSourceParagraph(
                page_number=4,
                paragraph_number=2,
                kind=ParagraphKind.PROSE,
                original_text="این متن اصلی و بدون نرمال‌سازی است.",
                normalized_text="این متن اصلی و بدون نرمال سازی است.",
            )
        ],
    )
    await documents.mark_succeeded(version.id, extraction_metadata={})
    await documents.activate_owned_version(
        owner_id=user.id, document_id=document.id, version_id=version.id
    )

    conversations = ConversationRepository(db_session)
    conversation = await conversations.create(owner_id=user.id)
    await conversations.add_document(
        owner_id=user.id,
        conversation_id=conversation.id,
        document_id=document.id,
    )
    pending = await conversations.create_pending_assistant_message(
        owner_id=user.id, conversation_id=conversation.id
    )
    completed = await conversations.complete_assistant_message(
        owner_id=user.id,
        message_id=pending.id,
        content="بخش پشتیبانی‌شده پاسخ.",
        outcome=AnswerOutcome.PARTIALLY_ANSWERED,
        cited_paragraph_ids=[paragraph.id],
        model_usage={"input_tokens": 10, "output_tokens": 4},
    )

    assert completed.status.value == "completed"
    assert completed.citation_snapshot == [
        {
            "source_paragraph_id": str(paragraph.id),
            "document_id": str(document.id),
            "document_version_id": str(version.id),
            "document_title": "منبع فارسی",
            "page_number": 4,
            "paragraph_number": 2,
            "original_text": "این متن اصلی و بدون نرمال‌سازی است.",
        }
    ]


async def test_citations_outside_conversation_scope_are_rejected(
    db_session: AsyncSession,
) -> None:
    owner = await _create_user(db_session, "chat-owner@example.com")
    other = await _create_user(db_session, "source-owner@example.com")
    documents = DocumentRepository(db_session)
    document = await documents.create_owned(owner_id=other.id, title="غیرمجاز")
    version = await documents.create_owned_version(
        owner_id=other.id,
        document_id=document.id,
        values=_new_version("documents/unauthorized.pdf", "d"),
    )
    await documents.mark_processing(version.id)
    (paragraph,) = await documents.add_source_paragraphs(
        version_id=version.id,
        paragraphs=[
            NewSourceParagraph(
                page_number=1,
                paragraph_number=1,
                kind=ParagraphKind.PROSE,
                original_text="محرمانه",
                normalized_text="محرمانه",
            )
        ],
    )

    conversations = ConversationRepository(db_session)
    conversation = await conversations.create(owner_id=owner.id)
    pending = await conversations.create_pending_assistant_message(
        owner_id=owner.id, conversation_id=conversation.id
    )
    try:
        await conversations.complete_assistant_message(
            owner_id=owner.id,
            message_id=pending.id,
            content="نباید ذخیره شود",
            outcome=AnswerOutcome.FULLY_ANSWERED,
            cited_paragraph_ids=[paragraph.id],
        )
    except InvalidPersistenceStateError:
        pass
    else:
        raise AssertionError("Out-of-scope citation was accepted")

    try:
        await conversations.add_document(
            owner_id=owner.id,
            conversation_id=conversation.id,
            document_id=document.id,
        )
    except RecordNotFoundError:
        pass
    else:
        raise AssertionError("Another user's document entered the conversation scope")


async def test_failed_versions_remain_auditable(db_session: AsyncSession) -> None:
    user = await _create_user(db_session, "failure@example.com")
    documents = DocumentRepository(db_session)
    document = await documents.create_owned(owner_id=user.id, title="شکست")
    version = await documents.create_owned_version(
        owner_id=user.id,
        document_id=document.id,
        values=_new_version("documents/failure.pdf", "e"),
    )
    await documents.mark_failed(
        version.id,
        failure_code="pdf_invalid",
        failure_message="PDF could not be parsed",
        extraction_metadata={"extractor": "test"},
    )

    assert version.status is IngestionStatus.FAILED
    assert version.failed_at is not None
    assert version.failure_code == "pdf_invalid"
