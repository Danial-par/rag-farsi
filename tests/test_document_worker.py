from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession, async_sessionmaker

from rag_assistant.core.config import Settings
from rag_assistant.db.base import utc_now
from rag_assistant.db.models import Document, User
from rag_assistant.db.repositories import DocumentRepository, NewDocumentVersion, UserRepository
from rag_assistant.indexing.storage import LocalObjectStore, ObjectStoreError
from rag_assistant.indexing.worker import DocumentIngestionWorker


async def test_skip_locked_worker_claims_do_not_duplicate_work(
    database_engine: AsyncEngine,
) -> None:
    factory = async_sessionmaker(database_engine, expire_on_commit=False)
    owner_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    async with factory() as setup, setup.begin():
        user = User(
            id=owner_id,
            email=f"{owner_id}@example.com",
            normalized_email=f"{owner_id}@example.com",
            password_hash="test-password-hash",
        )
        setup.add(user)
        document = await DocumentRepository(setup).create_owned(
            owner_id=owner_id, title="Worker claim", document_id=document_id
        )
        await DocumentRepository(setup).create_owned_version(
            owner_id=owner_id,
            document_id=document.id,
            values=NewDocumentVersion(
                version_id=version_id,
                object_key=f"documents/{owner_id}/{document_id}/{version_id}.pdf",
                content_hash="a" * 64,
                file_size_bytes=10,
                embedding_model="openai/text-embedding-3-large",
                embedding_dimensions=1024,
            ),
        )

    try:
        async with factory() as first:
            await first.begin()
            claim = await DocumentRepository(first).claim_next_for_indexing(
                lease_duration=timedelta(minutes=5)
            )
            assert claim is not None and claim.version_id == version_id
            async with factory() as second, second.begin():
                second_claim = await DocumentRepository(second).claim_next_for_indexing(
                    lease_duration=timedelta(minutes=5)
                )
                assert second_claim is None
            await first.rollback()
    finally:
        async with factory() as cleanup, cleanup.begin():
            await cleanup.execute(delete(Document).where(Document.id == document_id))
            await cleanup.execute(delete(User).where(User.id == owner_id))


async def test_orphan_reconciliation_deletes_only_old_unreferenced_objects(
    database_connection: AsyncConnection,
    settings: Settings,
    tmp_path: Path,
) -> None:
    store = LocalObjectStore(tmp_path)
    persisted_key = "documents/user/doc/persisted.pdf"
    orphan_key = "documents/user/doc/orphan.pdf"
    await store.put(persisted_key, b"persisted", content_type="application/pdf")
    await store.put(orphan_key, b"orphan", content_type="application/pdf")
    expired_at = (utc_now() - timedelta(seconds=120)).timestamp()
    os.utime(tmp_path / persisted_key, (expired_at, expired_at))
    os.utime(tmp_path / orphan_key, (expired_at, expired_at))

    async with AsyncSession(bind=database_connection) as session, session.begin():
        user = await UserRepository(session).create(
            email="orphan-owner@example.com",
            password_hash="test-password-hash",
        )
        document = await DocumentRepository(session).create_owned(
            owner_id=user.id, title="Orphan cleanup"
        )
        await DocumentRepository(session).create_owned_version(
            owner_id=user.id,
            document_id=document.id,
            values=NewDocumentVersion(
                object_key=persisted_key,
                content_hash="b" * 64,
                file_size_bytes=9,
                embedding_model="openai/text-embedding-3-large",
                embedding_dimensions=1024,
            ),
        )

    session_factory = async_sessionmaker(
        bind=database_connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    class UnusedIndexingService:
        async def index_claim(self, claim: Any) -> None:
            del claim

    worker = DocumentIngestionWorker(
        session_factory=session_factory,
        object_store=store,
        indexing_service=UnusedIndexingService(),  # type: ignore[arg-type]
        settings=settings.model_copy(update={"orphan_object_grace_seconds": 60}),
    )
    removed = await worker.reconcile_orphan_objects()

    assert removed == 1
    assert await store.get(persisted_key) == b"persisted"
    with pytest.raises(ObjectStoreError):
        await store.get(orphan_key)
