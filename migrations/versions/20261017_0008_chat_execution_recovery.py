"""Allow pending assistant execution heartbeat updates.

Revision ID: 20261017_0008
Revises: 20261017_0007
Create Date: 2026-10-17
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20261017_0008"
down_revision: str | None = "20261017_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_message_update() RETURNS trigger AS $$
        BEGIN
            IF OLD.status IN ('completed', 'failed') THEN
                RAISE EXCEPTION 'terminal messages are immutable' USING ERRCODE = 'check_violation';
            END IF;
            IF OLD.status = 'pending' AND OLD.role = 'assistant'
                AND NEW.status = 'pending'
                AND (to_jsonb(NEW) - 'updated_at') = (to_jsonb(OLD) - 'updated_at') THEN
                RETURN NEW;
            END IF;
            IF OLD.role IS DISTINCT FROM NEW.role
                OR OLD.conversation_id IS DISTINCT FROM NEW.conversation_id
                OR OLD.created_at IS DISTINCT FROM NEW.created_at
                OR NEW.status NOT IN ('completed', 'failed') THEN
                RAISE EXCEPTION 'invalid message transition' USING ERRCODE = 'check_violation';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )


def downgrade() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_message_update() RETURNS trigger AS $$
        BEGIN
            IF OLD.status IN ('completed', 'failed') THEN
                RAISE EXCEPTION 'terminal messages are immutable' USING ERRCODE = 'check_violation';
            END IF;
            IF OLD.role IS DISTINCT FROM NEW.role
                OR OLD.conversation_id IS DISTINCT FROM NEW.conversation_id
                OR OLD.created_at IS DISTINCT FROM NEW.created_at
                OR NEW.status NOT IN ('completed', 'failed') THEN
                RAISE EXCEPTION 'invalid message transition' USING ERRCODE = 'check_violation';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
