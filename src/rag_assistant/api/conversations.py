from __future__ import annotations

import base64
import binascii
from datetime import UTC, datetime
from typing import Annotated, TypeVar, cast
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from rag_assistant.api.auth import get_current_user
from rag_assistant.assistant.errors import ConversationAuthorizationError
from rag_assistant.assistant.graph import RAGWorkflow
from rag_assistant.assistant.models import RAGGraphInput
from rag_assistant.db.models import (
    AnswerOutcome,
    Conversation,
    Document,
    Message,
    MessageRole,
    MessageStatus,
    ParagraphKind,
    User,
)
from rag_assistant.db.repositories import ConversationRepository, RecordNotFoundError
from rag_assistant.db.repositories.citation_context import read_citation_context
from rag_assistant.db.repositories.conversations import IdempotencyConflictError
from rag_assistant.db.session import get_db_session

router = APIRouter(prefix="/conversations", tags=["conversations"])
logger = structlog.stdlib.get_logger(__name__)
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
ItemT = TypeVar("ItemT")


class ConversationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=500)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Title cannot be blank")
        return value


class ConversationTitleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        title = value.strip()
        if not title:
            raise ValueError("Title cannot be blank")
        return title


class DocumentSelectionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_ids: list[UUID] = Field(max_length=500)


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=10_000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank")
        return value


class DocumentSummary(BaseModel):
    id: UUID
    title: str

    @classmethod
    def from_document(cls, document: Document) -> DocumentSummary:
        return cls(id=document.id, title=document.title)


class ConversationSummary(BaseModel):
    id: UUID
    title: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> ConversationSummary:
        return cls(
            id=conversation.id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )


class ConversationDetail(ConversationSummary):
    selected_documents: list[DocumentSummary]


class CitationResponse(BaseModel):
    source_paragraph_id: UUID
    document_id: UUID
    document_version_id: UUID
    document_title: str
    page_number: int = Field(ge=1)
    paragraph_number: int = Field(ge=1)
    original_text: str
    start_offset: int | None = None
    end_offset: int | None = None


class MessageResponse(BaseModel):
    id: UUID
    role: MessageRole
    execution_status: MessageStatus
    answer_outcome: AnswerOutcome | None
    content: str | None
    citations: list[CitationResponse]
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime

    @classmethod
    def from_message(cls, message: Message) -> MessageResponse:
        return cls(
            id=message.id,
            role=message.role,
            execution_status=message.status,
            answer_outcome=message.answer_outcome,
            content=message.content,
            citations=[CitationResponse.model_validate(item) for item in message.citation_snapshot],
            error_code=message.error_code,
            error_message=message.error_message,
            created_at=message.created_at,
        )


class CitationContextParagraph(BaseModel):
    source_paragraph_id: UUID
    page_number: int
    paragraph_number: int
    kind: ParagraphKind
    original_text: str
    is_cited: bool


class CitationContextResponse(BaseModel):
    heading: str | None
    paragraphs: list[CitationContextParagraph]
    truncated: bool


class PageResponse[ItemT](BaseModel):
    items: list[ItemT]
    next_cursor: str | None


def get_rag_workflow(request: Request) -> RAGWorkflow:
    workflow = getattr(request.app.state, "rag_workflow", None)
    if workflow is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant service is unavailable",
            headers={"Cache-Control": "no-store"},
        )
    return cast(RAGWorkflow, workflow)


@router.post("", response_model=ConversationDetail, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    response: Response,
) -> ConversationDetail:
    repository = ConversationRepository(session)
    conversation = await repository.create(owner_id=user.id, title=payload.title)
    documents = await repository.list_active_documents(
        owner_id=user.id, conversation_id=conversation.id
    )
    result = ConversationDetail(
        **ConversationSummary.from_conversation(conversation).model_dump(),
        selected_documents=[DocumentSummary.from_document(doc) for doc in documents],
    )
    await session.commit()
    response.headers["Cache-Control"] = "no-store"
    return result


@router.get("", response_model=PageResponse[ConversationSummary])
async def list_conversations(
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    response: Response,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    cursor: Annotated[str | None, Query(max_length=256)] = None,
) -> PageResponse[ConversationSummary]:
    response.headers["Cache-Control"] = "no-store"
    before = _decode_cursor(cursor) if cursor else None
    rows = await ConversationRepository(session).list_owned(
        owner_id=user.id, limit=limit + 1, before=before
    )
    has_more = len(rows) > limit
    page = rows[:limit]
    return PageResponse[ConversationSummary](
        items=[ConversationSummary.from_conversation(item) for item in page],
        next_cursor=_encode_cursor(page[-1].updated_at, page[-1].id) if has_more and page else None,
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def read_conversation(
    conversation_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    response: Response,
) -> ConversationDetail:
    response.headers["Cache-Control"] = "no-store"
    repository = ConversationRepository(session)
    conversation = await repository.get_owned(owner_id=user.id, conversation_id=conversation_id)
    if conversation is None:
        raise _not_found()
    documents = await repository.list_active_documents(
        owner_id=user.id, conversation_id=conversation_id
    )
    return ConversationDetail(
        **ConversationSummary.from_conversation(conversation).model_dump(),
        selected_documents=[DocumentSummary.from_document(doc) for doc in documents],
    )


@router.patch("/{conversation_id}", response_model=ConversationSummary)
async def rename_conversation(
    conversation_id: UUID,
    payload: ConversationTitleUpdate,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    response: Response,
) -> ConversationSummary:
    response.headers["Cache-Control"] = "no-store"
    try:
        conversation = await ConversationRepository(session).rename(
            owner_id=user.id,
            conversation_id=conversation_id,
            title=payload.title,
        )
    except RecordNotFoundError as exc:
        raise _not_found() from exc
    await session.commit()
    return ConversationSummary.from_conversation(conversation)


@router.put("/{conversation_id}/documents", response_model=list[DocumentSummary])
async def replace_conversation_documents(
    conversation_id: UUID,
    payload: DocumentSelectionUpdate,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    response: Response,
) -> list[DocumentSummary]:
    response.headers["Cache-Control"] = "no-store"
    try:
        documents = await ConversationRepository(session).replace_documents(
            owner_id=user.id,
            conversation_id=conversation_id,
            document_ids=payload.document_ids,
        )
        await session.commit()
    except RecordNotFoundError as exc:
        raise _not_found() from exc
    return [DocumentSummary.from_document(document) for document in documents]


@router.get("/{conversation_id}/messages", response_model=PageResponse[MessageResponse])
async def read_message_history(
    conversation_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    response: Response,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    cursor: Annotated[str | None, Query(max_length=256)] = None,
) -> PageResponse[MessageResponse]:
    response.headers["Cache-Control"] = "no-store"
    before = _decode_cursor(cursor) if cursor else None
    rows = await ConversationRepository(session).list_messages(
        owner_id=user.id,
        conversation_id=conversation_id,
        limit=limit + 1,
        before=before,
    )
    if rows is None:
        raise _not_found()
    has_more = len(rows) > limit
    page = rows[:limit]
    return PageResponse[MessageResponse](
        items=[MessageResponse.from_message(item) for item in page],
        next_cursor=_encode_cursor(page[-1].created_at, page[-1].id) if has_more and page else None,
    )


@router.get(
    "/{conversation_id}/messages/{message_id}/citations/{source_paragraph_id}/context",
    response_model=CitationContextResponse,
)
async def get_citation_context(
    conversation_id: UUID,
    message_id: UUID,
    source_paragraph_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    response: Response,
) -> CitationContextResponse:
    response.headers["Cache-Control"] = "no-store"
    context = await read_citation_context(
        session,
        owner_id=user.id,
        conversation_id=conversation_id,
        message_id=message_id,
        source_paragraph_id=source_paragraph_id,
    )
    if context is None:
        raise HTTPException(
            status_code=404,
            detail="Citation context not found",
            headers={"Cache-Control": "no-store"},
        )
    return CitationContextResponse(
        heading=context.heading,
        paragraphs=[
            CitationContextParagraph(
                source_paragraph_id=paragraph.id,
                page_number=paragraph.page_number,
                paragraph_number=paragraph.paragraph_number,
                kind=paragraph.kind,
                original_text=paragraph.original_text,
                is_cited=paragraph.id == source_paragraph_id,
            )
            for paragraph in context.paragraphs
        ],
        truncated=context.truncated,
    )


@router.post("/{conversation_id}/messages", response_model=MessageResponse)
async def post_question(
    conversation_id: UUID,
    payload: QuestionRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
    workflow: Annotated[RAGWorkflow, Depends(get_rag_workflow)],
    http_response: Response,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=128)],
) -> MessageResponse:
    http_response.headers["Cache-Control"] = "no-store"
    # Release the authentication read transaction before the graph opens its own sessions.
    await session.commit()
    try:
        output = await workflow.ainvoke(
            RAGGraphInput(
                user_id=user.id,
                conversation_id=conversation_id,
                question=payload.content,
                idempotency_key=idempotency_key,
            )
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Idempotency key was already used for another question",
            headers={"Cache-Control": "no-store"},
        ) from exc
    except ConversationAuthorizationError as exc:
        raise _not_found() from exc
    except Exception as exc:
        await logger.awarning(
            "chat_workflow_unhandled_failure",
            user_id=str(user.id),
            conversation_id=str(conversation_id),
            error_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant execution failed",
            headers={"Cache-Control": "no-store"},
        ) from exc

    response = MessageResponse(
        id=output.assistant_message_id,
        role=MessageRole.ASSISTANT,
        execution_status=output.status,
        answer_outcome=output.outcome,
        content=output.content,
        citations=[CitationResponse.model_validate(item) for item in output.citation_snapshot],
        error_code=output.error_code,
        error_message=output.error_message,
        created_at=output.created_at,
    )
    if output.status is MessageStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(mode="json"),
            headers={"Cache-Control": "no-store"},
        )
    if output.status is MessageStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=response.model_dump(mode="json"),
            headers={"Cache-Control": "no-store"},
        )
    return response


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Conversation not found",
        headers={"Cache-Control": "no-store"},
    )


def _encode_cursor(created_at: datetime, item_id: UUID) -> str:
    raw = f"{created_at.astimezone(UTC).isoformat()}|{item_id}"
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _decode_cursor(value: str) -> tuple[datetime, UUID]:
    try:
        padded = value + "=" * (-len(value) % 4)
        timestamp, raw_id = base64.urlsafe_b64decode(padded.encode()).decode().split("|", 1)
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None:
            raise ValueError
        return parsed.astimezone(UTC), UUID(raw_id)
    except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
        raise HTTPException(status_code=422, detail="Invalid pagination cursor") from exc
