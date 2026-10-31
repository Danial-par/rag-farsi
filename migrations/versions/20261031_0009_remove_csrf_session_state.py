"""Remove double-submit CSRF session state.

Revision ID: 20261031_0009
Revises: 20261017_0008
Create Date: 2026-10-31
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261031_0009"
down_revision: str | None = "20261017_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(
        op.f("ck_auth_sessions_csrf_token_hash_length"),
        "auth_sessions",
        type_="check",
    )
    op.drop_column("auth_sessions", "csrf_token_hash")


def downgrade() -> None:
    op.add_column(
        "auth_sessions",
        sa.Column("csrf_token_hash", sa.String(length=64), nullable=True),
    )
    op.execute("UPDATE auth_sessions SET csrf_token_hash = repeat('0', 64)")
    op.alter_column("auth_sessions", "csrf_token_hash", nullable=False)
    op.create_check_constraint(
        "csrf_token_hash_length",
        "auth_sessions",
        "length(csrf_token_hash) = 64",
    )
