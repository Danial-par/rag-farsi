from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_assistant.db.base import utc_now
from rag_assistant.db.models import AuthSession, User
from rag_assistant.db.repositories.errors import (
    InvalidPersistenceStateError,
    RecordNotFoundError,
)


class AuthSessionRepository:
    """Revocable server-side session records referenced by JWT session claims."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        user_id: UUID,
        expires_at: datetime,
        refresh_token_hash: str,
        csrf_token_hash: str,
        client_metadata: dict[str, Any] | None = None,
    ) -> AuthSession:
        user_exists = await self._session.scalar(
            select(User.id).where(User.id == user_id, User.is_active.is_(True))
        )
        if user_exists is None:
            raise RecordNotFoundError("Active user not found")
        if expires_at <= utc_now():
            raise InvalidPersistenceStateError("Session expiry must be in the future")

        auth_session = AuthSession(
            user_id=user_id,
            expires_at=expires_at,
            refresh_token_hash=refresh_token_hash,
            csrf_token_hash=csrf_token_hash,
            client_metadata=client_metadata or {},
        )
        self._session.add(auth_session)
        await self._session.flush()
        return auth_session

    async def get_by_id(self, session_id: UUID) -> AuthSession | None:
        return await self._session.get(AuthSession, session_id)

    async def get_for_update(self, session_id: UUID) -> AuthSession | None:
        """Lock a session while rotating or revoking its refresh-token family."""
        return await self._session.scalar(
            select(AuthSession).where(AuthSession.id == session_id).with_for_update()
        )

    async def get_active(
        self, session_id: UUID, *, at: datetime | None = None
    ) -> AuthSession | None:
        checked_at = at or utc_now()
        return await self._session.scalar(
            select(AuthSession)
            .join(User, User.id == AuthSession.user_id)
            .where(
                AuthSession.id == session_id,
                AuthSession.revoked_at.is_(None),
                AuthSession.expires_at > checked_at,
                User.is_active.is_(True),
            )
        )

    async def revoke(self, session_id: UUID, *, reason: str | None = None) -> AuthSession:
        auth_session = await self.get_for_update(session_id)
        if auth_session is None:
            raise RecordNotFoundError("Authentication session not found")
        if auth_session.revoked_at is None:
            auth_session.revoked_at = utc_now()
            auth_session.revocation_reason = reason
            await self._session.flush()
        return auth_session
