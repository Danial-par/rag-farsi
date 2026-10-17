"""Add message idempotency keys for chat request retries.

Revision ID: 20261017_0006
Revises: 20261017_0005
Create Date: 2026-10-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261017_0006"
down_revision: str | None = "20261017_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("idempotency_key", sa.String(length=128), nullable=True))
    op.create_index(
        "uq_messages_user_idempotency_key",
        "messages",
        ["conversation_id", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text("role = 'user' AND idempotency_key IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_messages_user_idempotency_key", table_name="messages")
    op.drop_column("messages", "idempotency_key")
