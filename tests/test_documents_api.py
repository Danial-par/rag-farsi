from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import cast
from uuid import UUID

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from rag_assistant.db.models import Document, DocumentVersion
from rag_assistant.db.repositories import DocumentRepository
from rag_assistant.indexing.storage import LocalObjectStore, ObjectStoreError

PASSWORD = "correct horse battery staple"


async def _register(client: AsyncClient, email: str) -> dict[str, str]:
    response = await client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": PASSWORD},
    )
    assert response.status_code == 201
    return cast(dict[str, str], response.json())


async def test_document_upload_hashes_source_and_returns_pending_version(
    database_client: AsyncClient,
    application: FastAPI,
    database_connection: AsyncConnection,
    tmp_path: Path,
) -> None:
    owner = await _register(database_client, "document-owner@example.com")
    store = LocalObjectStore(tmp_path)
    application.state.object_store = store
    payload = b"%PDF-1.7\nfixture payload"

    response = await database_client.post(
        "/api/v1/documents?title=Persian%20source",
        content=payload,
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 202
    document = response.json()
    version = document["versions"][0]
    assert document["title"] == "Persian source"
    assert version["status"] == "pending"
    assert version["file_size_bytes"] == len(payload)
    assert version["media_type"] == "application/pdf"
    async with AsyncSession(bind=database_connection) as session:
        row = await session.get(DocumentVersion, UUID(version["id"]))
        assert row is not None
        assert row.content_hash == sha256(payload).hexdigest()
        assert row.object_key.endswith(f"/{document['id']}/{version['id']}.pdf")
        assert await store.get(row.object_key) == payload
        assert row.object_key.startswith(f"documents/{owner['id']}/")


async def test_available_documents_includes_shared_defaults(
    database_client: AsyncClient,
    database_connection: AsyncConnection,
) -> None:
    await _register(database_client, "shared-browser@example.com")
    async with AsyncSession(bind=database_connection) as session, session.begin():
        await DocumentRepository(session).create_shared(
            title="Shared sample",
            default_for_all_users=True,
            metadata={"seed_key": "test-shared"},
        )

    response = await database_client.get("/api/v1/documents/available")

    assert response.status_code == 200
    assert [document["title"] for document in response.json()["items"]] == ["Shared sample"]


async def test_upload_rejects_media_type_empty_body_and_non_pdf(
    database_client: AsyncClient,
) -> None:
    await _register(database_client, "validation-owner@example.com")
    wrong_type = await database_client.post(
        "/api/v1/documents?title=bad",
        content=b"%PDF-1.7",
        headers={"Content-Type": "application/octet-stream"},
    )
    empty = await database_client.post(
        "/api/v1/documents?title=bad",
        content=b"",
        headers={"Content-Type": "application/pdf"},
    )
    invalid = await database_client.post(
        "/api/v1/documents?title=bad",
        content=b"not a PDF",
        headers={"Content-Type": "application/pdf"},
    )
    assert wrong_type.status_code == 415
    assert empty.status_code == invalid.status_code == 422


async def test_document_endpoints_are_owner_scoped_and_soft_delete(
    database_client: AsyncClient,
    application: FastAPI,
    database_connection: AsyncConnection,
    tmp_path: Path,
) -> None:
    await _register(database_client, "first-owner@example.com")
    application.state.object_store = LocalObjectStore(tmp_path)
    payload = b"%PDF-1.7\nfixture"
    uploaded = await database_client.post(
        "/api/v1/documents?title=Owned",
        content=payload,
        headers={"Content-Type": "application/pdf"},
    )
    assert uploaded.status_code == 202
    document_id = UUID(uploaded.json()["id"])
    version_id = UUID(uploaded.json()["versions"][0]["id"])

    async with AsyncClient(
        transport=ASGITransport(app=application), base_url="http://test"
    ) as other_client:
        await _register(other_client, "second-owner@example.com")
        assert (await other_client.get(f"/api/v1/documents/{document_id}")).status_code == 404
        assert (
            await other_client.delete(f"/api/v1/documents/{document_id}")
        ).status_code == 404
        foreign_upload = await other_client.post(
            f"/api/v1/documents?document_id={document_id}",
            content=payload,
            headers={"Content-Type": "application/pdf"},
        )
        assert foreign_upload.status_code == 404
        assert len((await application.state.object_store.list("documents/")).objects) == 1

    async with AsyncSession(bind=database_connection) as session, session.begin():
        version = await session.get(DocumentVersion, version_id)
        assert version is not None
        await DocumentRepository(session).mark_failed(
            version_id,
            failure_code="source_unavailable",
            failure_message="private provider detail must not be returned",
        )

    retry = await database_client.post(
        f"/api/v1/documents/{document_id}/versions/{version_id}/retry"
    )
    assert retry.status_code == 202
    assert retry.json()["status"] == "pending"
    assert retry.json()["failure_message"] is None

    deleted = await database_client.delete(f"/api/v1/documents/{document_id}")
    assert deleted.status_code == 204
    assert (await database_client.get(f"/api/v1/documents/{document_id}")).status_code == 404
    async with AsyncSession(bind=database_connection) as session:
        row = await session.scalar(select(Document).where(Document.id == document_id))
        assert row is not None and row.deleted_at is not None


async def test_upload_enforces_configured_size_limit(
    database_client: AsyncClient,
    application: FastAPI,
) -> None:
    await _register(database_client, "size-owner@example.com")
    application.state.settings = application.state.settings.model_copy(
        update={"max_pdf_upload_bytes": 8}
    )
    response = await database_client.post(
        "/api/v1/documents?title=Large",
        content=b"%PDF-1.7 too large",
        headers={"Content-Type": "application/pdf"},
    )
    assert response.status_code == 413


async def test_upload_reports_storage_failure_without_creating_a_document(
    database_client: AsyncClient,
    application: FastAPI,
) -> None:
    await _register(database_client, "storage-failure@example.com")

    class FailingStore:
        async def put(self, object_key: str, payload: bytes, *, content_type: str) -> None:
            del object_key, payload, content_type
            raise ObjectStoreError("private storage failure detail")

    application.state.object_store = FailingStore()
    response = await database_client.post(
        "/api/v1/documents?title=Unavailable",
        content=b"%PDF-1.7\nfixture",
        headers={"Content-Type": "application/pdf"},
    )

    assert response.status_code == 503
    assert "private storage failure detail" not in response.text
