"""Add refresh rotation and CSRF state to authentication sessions.

Revision ID: 20261011_0003
Revises: 20261011_0002
Create Date: 2026-10-11
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261011_0003"
down_revision: str | None = "20261011_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "auth_sessions", sa.Column("refresh_token_hash", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "auth_sessions",
        sa.Column("previous_refresh_token_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "auth_sessions",
        sa.Column("previous_refresh_token_valid_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "auth_sessions", sa.Column("csrf_token_hash", sa.String(length=64), nullable=True)
    )

    # No tokens were issued by earlier revisions. Fail closed if seed data nevertheless exists.
    op.execute(
        """
        UPDATE auth_sessions
        SET refresh_token_hash = repeat('0', 64),
            csrf_token_hash = repeat('0', 64),
            revoked_at = COALESCE(revoked_at, CURRENT_TIMESTAMP),
            revocation_reason = COALESCE(revocation_reason, 'authentication_upgrade')
        """
    )
    op.alter_column("auth_sessions", "refresh_token_hash", nullable=False)
    op.alter_column("auth_sessions", "csrf_token_hash", nullable=False)
    op.create_check_constraint(
        "refresh_token_hash_length",
        "auth_sessions",
        "length(refresh_token_hash) = 64",
    )
    op.create_check_constraint(
        "csrf_token_hash_length",
        "auth_sessions",
        "length(csrf_token_hash) = 64",
    )
    op.create_check_constraint(
        "previous_refresh_token_hash_length",
        "auth_sessions",
        "previous_refresh_token_hash IS NULL OR length(previous_refresh_token_hash) = 64",
    )
    op.create_check_constraint(
        "previous_refresh_token_grace_complete",
        "auth_sessions",
        "(previous_refresh_token_hash IS NULL) = (previous_refresh_token_valid_until IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_auth_sessions_previous_refresh_token_grace_complete"),
        "auth_sessions",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_auth_sessions_previous_refresh_token_hash_length"),
        "auth_sessions",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_auth_sessions_csrf_token_hash_length"), "auth_sessions", type_="check"
    )
    op.drop_constraint(
        op.f("ck_auth_sessions_refresh_token_hash_length"), "auth_sessions", type_="check"
    )
    op.drop_column("auth_sessions", "csrf_token_hash")
    op.drop_column("auth_sessions", "previous_refresh_token_valid_until")
    op.drop_column("auth_sessions", "previous_refresh_token_hash")
    op.drop_column("auth_sessions", "refresh_token_hash")
