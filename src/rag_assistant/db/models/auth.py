from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_assistant.db.base import (
    Base,
    CreatedAtMixin,
    UpdatedAtMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from rag_assistant.db.models.user import User


class AuthSession(UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, Base):
    __tablename__ = "auth_sessions"
    __table_args__ = (
        CheckConstraint("expires_at > created_at", name="expiry_after_creation"),
        CheckConstraint(
            "revoked_at IS NULL OR revoked_at >= created_at", name="revocation_after_creation"
        ),
        CheckConstraint(
            "jsonb_typeof(client_metadata) = 'object'", name="client_metadata_is_object"
        ),
        CheckConstraint("length(refresh_token_hash) = 64", name="refresh_token_hash_length"),
        CheckConstraint("length(csrf_token_hash) = 64", name="csrf_token_hash_length"),
        CheckConstraint(
            "previous_refresh_token_hash IS NULL OR length(previous_refresh_token_hash) = 64",
            name="previous_refresh_token_hash_length",
        ),
        CheckConstraint(
            "(previous_refresh_token_hash IS NULL) = (previous_refresh_token_valid_until IS NULL)",
            name="previous_refresh_token_grace_complete",
        ),
        Index("ix_auth_sessions_user_id_expires_at", "user_id", "expires_at"),
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    refresh_token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    previous_refresh_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    previous_refresh_token_valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    csrf_token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revocation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    client_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    user: Mapped[User] = relationship(back_populates="auth_sessions")
