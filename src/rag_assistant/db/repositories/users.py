from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_assistant.db.models import User
from rag_assistant.db.repositories.errors import PersistenceConflictError


def normalize_email(email: str) -> str:
    """Return the canonical value used for case-insensitive email lookup."""
    return email.strip().casefold()


class UserRepository:
    """Persistence operations for the user aggregate; callers own transactions."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, user_id: UUID) -> User | None:
        return await self._session.get(User, user_id)

    async def get_by_normalized_email(self, email: str) -> User | None:
        return await self._session.scalar(
            select(User).where(User.normalized_email == normalize_email(email))
        )

    async def create(self, *, email: str, password_hash: str) -> User:
        normalized_email = normalize_email(email)
        user = User(
            email=email.strip(),
            normalized_email=normalized_email,
            password_hash=password_hash,
        )
        self._session.add(user)
        try:
            await self._session.flush()
        except IntegrityError as exc:
            raise PersistenceConflictError(
                "A user with that normalized email already exists"
            ) from exc
        return user
