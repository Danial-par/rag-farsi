from __future__ import annotations

from typing import Annotated, Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from rag_assistant.db.session import AsyncSessionFactory, get_session_factory

router = APIRouter(prefix="/health", tags=["health"])
logger = structlog.stdlib.get_logger(__name__)


class HealthResponse(BaseModel):
    status: Literal["ok", "ready"]


async def require_database_ready(
    session_factory: Annotated[AsyncSessionFactory, Depends(get_session_factory)],
) -> None:
    """Raise a service-unavailable response when PostgreSQL cannot answer."""
    try:
        async with session_factory() as session:
            await session.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        await logger.awarning("readiness_check_failed", error_type=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        ) from exc


@router.get("/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Report whether the application process is serving requests."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse)
async def readiness(_: Annotated[None, Depends(require_database_ready)]) -> HealthResponse:
    """Report whether required infrastructure is available."""
    return HealthResponse(status="ready")
