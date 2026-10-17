from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Annotated, cast
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_assistant.api.auth import get_current_user
from rag_assistant.core.config import Settings
from rag_assistant.db.models import Document, DocumentVersion, IngestionStatus, User
from rag_assistant.db.repositories import (
    DocumentRepository,
    InvalidPersistenceStateError,
    NewDocumentVersion,
    RecordNotFoundError,
)
from rag_assistant.db.session import get_db_session
from rag_assistant.indexing.storage import ObjectStoreError, ObjectWriter

router = APIRouter(prefix="/documents", tags=["documents"])
logger = structlog.stdlib.get_logger("rag_assistant.api.documents")
MAX_PAGE_SIZE = 100

SAFE_FAILURE_MESSAGES = {
    "pdf_encrypted": "The PDF is encrypted and could not be processed.",
    "pdf_malformed": "The PDF is malformed and could not be processed.",
    "pdf_empty": "The PDF contains no extractable text.",
    "pdf_ocr_required": "The PDF requires OCR, which is not supported.",
    "pdf_unsupported_layout": "The PDF layout is not supported.",
    "source_integrity_failed": "The stored source failed an integrity check.",
    "source_unavailable": "The stored source is temporarily unavailable.",
    "embedding_retries_exhausted": "Indexing could not reach the embedding service.",
    "index_persistence_failed": "The document index could not be saved.",
}


class DocumentVersionResponse(BaseModel):
    id: UUID
    version_number: int
    status: IngestionStatus
    file_size_bytes: int
    media_type: str
    failure_code: str | None
    failure_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    failed_at: datetime | None
    activated_at: datetime | None

    @classmethod
    def from_version(cls, version: DocumentVersion) -> DocumentVersionResponse:
        code = version.failure_code
        return cls(
            id=version.id,
            version_number=version.version_number,
            status=version.status,
            file_size_bytes=version.file_size_bytes,
            media_type=version.media_type,
            failure_code=code,
            failure_message=(
                SAFE_FAILURE_MESSAGES.get(code, "Document ingestion failed.") if code else None
            ),
            created_at=version.created_at,
            started_at=version.started_at,
            completed_at=version.completed_at,
            failed_at=version.failed_at,
            activated_at=version.activated_at,
        )


class DocumentResponse(BaseModel):
    id: UUID
    title: str
    active_version_id: UUID | None
    created_at: datetime
    updated_at: datetime
    versions: list[DocumentVersionResponse]

    @classmethod
    def from_document(cls, document: Document) -> DocumentResponse:
        return cls(
            id=document.id,
            title=document.title,
            active_version_id=document.active_version_id,
            created_at=document.created_at,
            updated_at=document.updated_at,
            versions=[
                DocumentVersionResponse.from_version(version)
                for version in sorted(document.versions, key=lambda item: item.version_number)
            ],
        )


class DocumentPageResponse(BaseModel):
    items: list[DocumentResponse]


class DocumentUploadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=500)
    document_id: UUID | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Title cannot be blank")
        return value


def _settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def _object_store(request: Request) -> ObjectWriter:
    return cast(ObjectWriter, request.app.state.object_store)


async def _read_pdf_body(request: Request, *, maximum_bytes: int) -> bytes:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/pdf":
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/pdf"
        )
    raw_length = request.headers.get("content-length")
    if raw_length is not None:
        try:
            declared_length = int(raw_length)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid Content-Length") from exc
        if declared_length > maximum_bytes:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "PDF exceeds the upload limit")
    received = bytearray()
    async for part in request.stream():
        if len(received) + len(part) > maximum_bytes:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "PDF exceeds the upload limit")
        received.extend(part)
    payload = bytes(received)
    if not payload:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "PDF must not be empty")
    if b"%PDF-" not in payload[:1024]:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "File does not have a PDF signature"
        )
    return payload


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    request: Request,
    response: Response,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    store: Annotated[ObjectWriter, Depends(_object_store)],
    settings: Annotated[Settings, Depends(_settings)],
    upload: Annotated[DocumentUploadRequest, Depends()],
) -> DocumentResponse:
    """Store a bounded PDF and atomically create its immutable version record."""
    response.headers["Cache-Control"] = "no-store"
    if upload.document_id is None and (upload.title is None or not upload.title.strip()):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "A title is required for a new document"
        )
    payload = await _read_pdf_body(request, maximum_bytes=settings.max_pdf_upload_bytes)
    is_new_document = upload.document_id is None
    document_id = upload.document_id or uuid4()
    version_id = uuid4()
    object_key = f"documents/{user.id}/{document_id}/{version_id}.pdf"
    digest = hashlib.sha256(payload).hexdigest()

    await session.commit()
    try:
        await store.put(object_key, payload, content_type="application/pdf")
    except ObjectStoreError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Document storage is unavailable"
        ) from exc
    try:
        async with session.begin():
            repository = DocumentRepository(session)
            if is_new_document:
                await repository.create_owned(
                    owner_id=user.id,
                    title=upload.title.strip() if upload.title is not None else "",
                    document_id=document_id,
                )
            else:
                existing_document = await repository.get_owned(
                    owner_id=user.id, document_id=document_id
                )
                if existing_document is None:
                    raise RecordNotFoundError("Owned document not found")
            await repository.create_owned_version(
                owner_id=user.id,
                document_id=document_id,
                values=NewDocumentVersion(
                    version_id=version_id,
                    object_key=object_key,
                    content_hash=digest,
                    file_size_bytes=len(payload),
                    embedding_model=settings.embedding_model,
                    embedding_dimensions=settings.embedding_dimensions,
                ),
            )
            document = await repository.get_owned(owner_id=user.id, document_id=document_id)
            if document is None:
                raise RecordNotFoundError("Owned document not found")
            result = DocumentResponse.from_document(document)
    except (RecordNotFoundError, SQLAlchemyError, InvalidPersistenceStateError) as exc:
        try:
            await store.delete(object_key)
        except ObjectStoreError:
            await logger.aerror("upload_compensation_failed", document_version_id=str(version_id))
        if isinstance(exc, RecordNotFoundError):
            raise _not_found() from exc
        await logger.aerror(
            "document_upload_persistence_failed",
            document_id=str(document_id),
            document_version_id=str(version_id),
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Document could not be saved"
        ) from exc
    return result


@router.get("", response_model=DocumentPageResponse)
async def list_documents(
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    response: Response,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = MAX_PAGE_SIZE,
) -> DocumentPageResponse:
    response.headers["Cache-Control"] = "no-store"
    documents = await DocumentRepository(session).list_owned(owner_id=user.id, limit=limit)
    return DocumentPageResponse(items=[DocumentResponse.from_document(item) for item in documents])


@router.get("/{document_id}", response_model=DocumentResponse)
async def read_document(
    document_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    response: Response,
) -> DocumentResponse:
    response.headers["Cache-Control"] = "no-store"
    document = await DocumentRepository(session).get_owned(
        owner_id=user.id, document_id=document_id
    )
    if document is None:
        raise _not_found()
    return DocumentResponse.from_document(document)


@router.post(
    "/{document_id}/versions/{version_id}/retry",
    response_model=DocumentVersionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_document_version(
    document_id: UUID,
    version_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    response: Response,
) -> DocumentVersionResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        version = await DocumentRepository(session).request_retry_owned(
            owner_id=user.id, document_id=document_id, version_id=version_id
        )
        await session.commit()
    except RecordNotFoundError as exc:
        raise _not_found() from exc
    except InvalidPersistenceStateError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "Document version cannot be retried") from exc
    return DocumentVersionResponse.from_version(version)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Response:
    try:
        await DocumentRepository(session).soft_delete_owned(
            owner_id=user.id, document_id=document_id
        )
        await session.commit()
    except RecordNotFoundError as exc:
        raise _not_found() from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT, headers={"Cache-Control": "no-store"})


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Document not found",
        headers={"Cache-Control": "no-store"},
    )
