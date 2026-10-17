from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast
from uuid import UUID

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from rag_assistant.core.config import Settings
from rag_assistant.db.models import (
    ChunkSourceParagraph,
    ConversationDocument,
    Document,
    DocumentVersion,
    ParagraphKind,
    User,
)
from rag_assistant.db.repositories import (
    ChunkParagraphReference,
    ConversationRepository,
    DocumentRepository,
    NewChunk,
    NewDocumentVersion,
    NewSourceParagraph,
    UserRepository,
)
from rag_assistant.retrieval import (
    IncompatibleEmbeddingConfigurationError,
    InvalidRetrievalQueryError,
    RetrievalIntegrityError,
    RetrievalRequest,
    RetrievalScopeError,
    RetrievalService,
)


def _vector(first: float, second: float = 0.0) -> list[float]:
    return [first, second, *([0.0] * 1022)]


class QueryEmbeddingClient:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or _vector(1.0)
        self.calls: list[tuple[list[str], str, int]] = []

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int,
    ) -> list[list[float]]:
        self.calls.append((list(texts), model, dimensions))
        return [list(self.vector) for _ in texts]


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    document: Document
    version: DocumentVersion
    paragraph_ids: tuple[UUID, ...]


async def _user(session: AsyncSession, email: str) -> User:
    return await UserRepository(session).create(
        email=email,
        password_hash="test-password-hash",
    )


async def _indexed_document(
    session: AsyncSession,
    *,
    key: str,
    title: str,
    vectors: Sequence[list[float]],
    owner_id: UUID | None = None,
    shared_default: bool = False,
    embedding_model: str = "openai/text-embedding-3-large",
    activate: bool = True,
) -> IndexedDocument:
    repository = DocumentRepository(session)
    if shared_default:
        document = await repository.create_shared(title=title, default_for_all_users=True)
        version = await repository.create_shared_version(
            document_id=document.id,
            values=NewDocumentVersion(
                object_key=key,
                content_hash="a" * 64,
                file_size_bytes=1024,
                embedding_model=embedding_model,
                embedding_dimensions=1024,
            ),
        )
    else:
        assert owner_id is not None
        document = await repository.create_owned(owner_id=owner_id, title=title)
        version = await repository.create_owned_version(
            owner_id=owner_id,
            document_id=document.id,
            values=NewDocumentVersion(
                object_key=key,
                content_hash="b" * 64,
                file_size_bytes=1024,
                embedding_model=embedding_model,
                embedding_dimensions=1024,
            ),
        )

    await repository.mark_processing(version.id)
    paragraphs = await repository.add_source_paragraphs(
        version_id=version.id,
        paragraphs=[
            NewSourceParagraph(
                page_number=index,
                paragraph_number=1,
                kind=ParagraphKind.PROSE,
                original_text=f"متن اصلی {title}، صفحه {index}",
                normalized_text=f"متن اصلی {title} صفحه {index}",
            )
            for index in range(1, len(vectors) + 1)
        ],
    )
    for ordinal, (paragraph, vector) in enumerate(zip(paragraphs, vectors, strict=True), start=1):
        await repository.add_chunk(
            version_id=version.id,
            values=NewChunk(
                ordinal=ordinal,
                normalized_text=paragraph.normalized_text,
                token_count=5,
                embedding=vector,
                paragraphs=[ChunkParagraphReference(source_paragraph_id=paragraph.id)],
            ),
        )
    await repository.mark_succeeded(version.id, extraction_metadata={})
    if activate:
        if shared_default:
            await repository.activate_shared_version(
                document_id=document.id,
                version_id=version.id,
            )
        else:
            assert owner_id is not None
            await repository.activate_owned_version(
                owner_id=owner_id,
                document_id=document.id,
                version_id=version.id,
            )
    return IndexedDocument(
        document=document,
        version=version,
        paragraph_ids=tuple(paragraph.id for paragraph in paragraphs),
    )


def _service(
    session: AsyncSession,
    settings: Settings,
    client: QueryEmbeddingClient,
) -> RetrievalService:
    return RetrievalService(session=session, embedding_client=client, settings=settings)


async def test_retrieval_normalizes_ranks_thresholds_and_resolves_citations(
    db_session: AsyncSession,
    settings: Settings,
) -> None:
    user = await _user(db_session, "retrieval@example.com")
    indexed = await _indexed_document(
        db_session,
        key="documents/ranked.pdf",
        title="راهنما",
        owner_id=user.id,
        vectors=[_vector(1.0), _vector(0.8, 0.6), _vector(0.8, 0.6)],
    )
    conversations = ConversationRepository(db_session)
    conversation = await conversations.create(owner_id=user.id)
    await conversations.add_document(
        owner_id=user.id,
        conversation_id=conversation.id,
        document_id=indexed.document.id,
    )
    embedding_client = QueryEmbeddingClient()
    service = _service(db_session, settings, embedding_client)

    result = await service.retrieve(
        RetrievalRequest(
            user_id=user.id,
            conversation_id=conversation.id,
            question="  كِتاب ۱۲  ",
        )
    )

    assert result.normalized_query == "کتاب 12"
    assert embedding_client.calls == [(["کتاب 12"], "openai/text-embedding-3-large", 1024)]
    assert [evidence.chunk_ordinal for evidence in result.evidence] == [1, 2, 3]
    assert result.evidence[0].cosine_similarity == pytest.approx(1.0)
    assert result.evidence[1].cosine_similarity == pytest.approx(0.8)
    citation = result.evidence[0].citation_sources[0]
    assert citation.source_paragraph_id == indexed.paragraph_ids[0]
    assert citation.document_id == indexed.document.id
    assert citation.document_version_id == indexed.version.id
    assert citation.document_title == "راهنما"
    assert (citation.page_number, citation.paragraph_number) == (1, 1)
    assert citation.original_text == "متن اصلی راهنما، صفحه 1"

    thresholded = await service.retrieve(
        RetrievalRequest(
            user_id=user.id,
            conversation_id=conversation.id,
            question="کتاب",
            minimum_similarity=0.81,
        )
    )
    assert [evidence.chunk_ordinal for evidence in thresholded.evidence] == [1]


async def test_retrieval_enforces_selection_defaults_and_cross_user_isolation(
    db_session: AsyncSession,
    settings: Settings,
) -> None:
    owner = await _user(db_session, "scope-owner@example.com")
    other = await _user(db_session, "scope-other@example.com")
    selected = await _indexed_document(
        db_session,
        key="documents/selected.pdf",
        title="انتخاب‌شده",
        owner_id=owner.id,
        vectors=[_vector(1.0)],
    )
    unselected = await _indexed_document(
        db_session,
        key="documents/unselected.pdf",
        title="انتخاب‌نشده",
        owner_id=owner.id,
        vectors=[_vector(1.0)],
    )
    foreign = await _indexed_document(
        db_session,
        key="documents/foreign.pdf",
        title="خصوصی کاربر دیگر",
        owner_id=other.id,
        vectors=[_vector(1.0)],
    )
    shared = await _indexed_document(
        db_session,
        key="documents/default.pdf",
        title="همگانی پیش‌فرض",
        shared_default=True,
        vectors=[_vector(1.0)],
    )
    conversations = ConversationRepository(db_session)
    conversation = await conversations.create(owner_id=owner.id)
    await conversations.add_document(
        owner_id=owner.id,
        conversation_id=conversation.id,
        document_id=selected.document.id,
    )
    # A stale or corrupt selection row must not grant access to another user's document.
    db_session.add(
        ConversationDocument(
            conversation_id=conversation.id,
            document_id=foreign.document.id,
        )
    )
    await db_session.flush()
    embedding_client = QueryEmbeddingClient()
    service = _service(db_session, settings, embedding_client)

    result = await service.retrieve(
        RetrievalRequest(
            user_id=owner.id,
            conversation_id=conversation.id,
            question="پرسش",
        )
    )

    returned_ids = {evidence.document_id for evidence in result.evidence}
    assert returned_ids == {selected.document.id, shared.document.id}
    assert unselected.document.id not in returned_ids
    assert foreign.document.id not in returned_ids

    unauthorized_client = QueryEmbeddingClient()
    with pytest.raises(RetrievalScopeError):
        await _service(db_session, settings, unauthorized_client).retrieve(
            RetrievalRequest(
                user_id=other.id,
                conversation_id=conversation.id,
                question="پرسش",
            )
        )
    assert unauthorized_client.calls == []


async def test_retrieval_ignores_deleted_and_inactive_versions_without_embedding(
    db_session: AsyncSession,
    settings: Settings,
) -> None:
    user = await _user(db_session, "unavailable@example.com")
    deleted = await _indexed_document(
        db_session,
        key="documents/deleted.pdf",
        title="حذف‌شده",
        owner_id=user.id,
        vectors=[_vector(1.0)],
    )
    inactive = await _indexed_document(
        db_session,
        key="documents/inactive.pdf",
        title="غیرفعال",
        owner_id=user.id,
        vectors=[_vector(1.0)],
        activate=False,
    )
    conversations = ConversationRepository(db_session)
    conversation = await conversations.create(owner_id=user.id)
    for document in (deleted.document, inactive.document):
        await conversations.add_document(
            owner_id=user.id,
            conversation_id=conversation.id,
            document_id=document.id,
        )
    await DocumentRepository(db_session).soft_delete_owned(
        owner_id=user.id,
        document_id=deleted.document.id,
    )
    embedding_client = QueryEmbeddingClient()

    result = await _service(db_session, settings, embedding_client).retrieve(
        RetrievalRequest(
            user_id=user.id,
            conversation_id=conversation.id,
            question="پرسش معتبر",
        )
    )

    assert result.evidence == ()
    assert embedding_client.calls == []


async def test_retrieval_fails_entire_scope_for_incompatible_active_index(
    db_session: AsyncSession,
    settings: Settings,
) -> None:
    user = await _user(db_session, "incompatible@example.com")
    compatible = await _indexed_document(
        db_session,
        key="documents/compatible.pdf",
        title="سازگار",
        owner_id=user.id,
        vectors=[_vector(1.0)],
    )
    incompatible = await _indexed_document(
        db_session,
        key="documents/incompatible.pdf",
        title="ناسازگار",
        owner_id=user.id,
        vectors=[_vector(1.0)],
        embedding_model="openai/different-embedding-model",
    )
    conversations = ConversationRepository(db_session)
    conversation = await conversations.create(owner_id=user.id)
    for document in (compatible.document, incompatible.document):
        await conversations.add_document(
            owner_id=user.id,
            conversation_id=conversation.id,
            document_id=document.id,
        )
    embedding_client = QueryEmbeddingClient()

    with pytest.raises(IncompatibleEmbeddingConfigurationError):
        await _service(db_session, settings, embedding_client).retrieve(
            RetrievalRequest(
                user_id=user.id,
                conversation_id=conversation.id,
                question="پرسش",
            )
        )
    assert embedding_client.calls == []


async def test_retrieval_rejects_normalization_only_query(
    settings: Settings,
) -> None:
    embedding_client = QueryEmbeddingClient()
    service = _service(cast(AsyncSession, object()), settings, embedding_client)

    with pytest.raises(InvalidRetrievalQueryError):
        await service.retrieve(
            RetrievalRequest(
                user_id=UUID(int=1),
                conversation_id=UUID(int=2),
                question=" \u064e\u064f\u200f\u0640 ",
            )
        )
    assert embedding_client.calls == []


async def test_retrieval_fails_closed_when_chunk_provenance_is_missing(
    db_session: AsyncSession,
    settings: Settings,
) -> None:
    user = await _user(db_session, "provenance@example.com")
    indexed = await _indexed_document(
        db_session,
        key="documents/provenance.pdf",
        title="منبع",
        owner_id=user.id,
        vectors=[_vector(1.0)],
    )
    conversations = ConversationRepository(db_session)
    conversation = await conversations.create(owner_id=user.id)
    await conversations.add_document(
        owner_id=user.id,
        conversation_id=conversation.id,
        document_id=indexed.document.id,
    )
    await db_session.execute(
        delete(ChunkSourceParagraph).where(
            ChunkSourceParagraph.source_paragraph_id == indexed.paragraph_ids[0]
        )
    )

    with pytest.raises(RetrievalIntegrityError):
        await _service(db_session, settings, QueryEmbeddingClient()).retrieve(
            RetrievalRequest(
                user_id=user.id,
                conversation_id=conversation.id,
                question="پرسش",
            )
        )
