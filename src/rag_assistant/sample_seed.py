"""Idempotently create or requeue the shared local sample PDF for indexing."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import select, text

from rag_assistant.core.config import Settings
from rag_assistant.db.models import Document, DocumentVersion, IngestionStatus
from rag_assistant.db.repositories import DocumentRepository, NewDocumentVersion
from rag_assistant.db.session import AsyncSessionFactory
from rag_assistant.indexing.storage import ObjectStore, ObjectStoreError

SAMPLE_SEED_KEY = "sample_doc_pdf_v1"
SAMPLE_RELATIVE_PATH = Path("tests/fixtures/documents/sample_doc.pdf")
SAMPLE_PDF_PATH = Path(__file__).resolve().parents[2] / SAMPLE_RELATIVE_PATH


@dataclass(frozen=True, slots=True)
class SampleSeedResult:
    document_id: UUID
    version_id: UUID
    status: IngestionStatus
    already_active: bool


async def seed_sample_document(
    *,
    settings: Settings,
    session_factory: AsyncSessionFactory,
    object_store: ObjectStore,
    fixture_path: Path | None = None,
) -> SampleSeedResult:
    """Store the committed sample and add one shared default pending version."""
    if settings.app_env != "local":
        raise ValueError("Sample seeding is available only when APP_ENV=local")
    if settings.object_store_backend != "local":
        raise ValueError("Sample seeding requires the local object store")

    path = fixture_path or SAMPLE_PDF_PATH
    payload = await asyncio.to_thread(path.read_bytes)
    if not payload or len(payload) > settings.max_pdf_upload_bytes:
        raise ValueError("Sample PDF is empty or exceeds the configured upload limit")
    digest = hashlib.sha256(payload).hexdigest()
    object_key = f"documents/shared/sample-doc/{digest}.pdf"

    async with session_factory() as session, session.begin():
        # Serialize repeated local invocations before looking up the shared default.
        await session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:seed_key))"),
            {"seed_key": SAMPLE_SEED_KEY},
        )
        document = await session.scalar(
            select(Document)
            .where(
                Document.is_shared.is_(True),
                Document.document_metadata["seed_key"].astext == SAMPLE_SEED_KEY,
                Document.deleted_at.is_(None),
            )
            .with_for_update()
        )
        if document is None:
            document = await DocumentRepository(session).create_shared(
                title="Sample document",
                default_for_all_users=True,
                metadata={
                    "seed_key": SAMPLE_SEED_KEY,
                    "source_fixture": SAMPLE_RELATIVE_PATH.as_posix(),
                },
            )
        elif not document.is_default:
            raise ValueError("The seeded sample document is not configured as a shared default")

        matching_versions = list(
            await session.scalars(
                select(DocumentVersion)
                .where(
                    DocumentVersion.document_id == document.id,
                    DocumentVersion.content_hash == digest,
                )
                .order_by(DocumentVersion.version_number.desc())
                .with_for_update()
            )
        )
        active = next(
            (
                version
                for version in matching_versions
                if document.active_version_id == version.id
                and version.status is IngestionStatus.SUCCEEDED
            ),
            None,
        )
        if active is not None:
            return SampleSeedResult(document.id, active.id, active.status, True)

        version = next(
            (
                item
                for item in matching_versions
                if item.status in {IngestionStatus.PENDING, IngestionStatus.PROCESSING}
            ),
            None,
        )
        if version is None:
            failed = next(
                (
                    item
                    for item in matching_versions
                    if item.status is IngestionStatus.FAILED and item.activated_at is None
                ),
                None,
            )
            if failed is not None:
                version = await DocumentRepository(session).request_retry_shared(
                    document_id=document.id, version_id=failed.id
                )

        if version is None:
            await _ensure_sample_object(object_store, object_key, payload)
            version = await DocumentRepository(session).create_shared_version(
                document_id=document.id,
                values=NewDocumentVersion(
                    object_key=object_key,
                    content_hash=digest,
                    file_size_bytes=len(payload),
                    embedding_model=settings.embedding_model,
                    embedding_dimensions=settings.embedding_dimensions,
                    ingestion_config={
                        "chunk_target_tokens": settings.chunk_target_tokens,
                        "chunk_max_tokens": settings.chunk_max_tokens,
                        "chunk_paragraph_overlap": settings.chunk_paragraph_overlap,
                    },
                ),
            )
        else:
            await _ensure_sample_object(object_store, version.object_key, payload)
        return SampleSeedResult(document.id, version.id, version.status, False)


async def _ensure_sample_object(object_store: ObjectStore, object_key: str, payload: bytes) -> None:
    try:
        existing = await object_store.get(object_key)
    except ObjectStoreError:
        await object_store.put(object_key, payload, content_type="application/pdf")
        return
    if existing != payload:
        raise ValueError("Stored sample object does not match its content hash")
