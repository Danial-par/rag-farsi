"""Allow extraction metadata to be recorded during processing.

Revision ID: 20261011_0004
Revises: 20261011_0003
Create Date: 2026-10-11
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20261011_0004"
down_revision: str | None = "20261011_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_FUNCTION_PREFIX = """
CREATE OR REPLACE FUNCTION enforce_document_version_update() RETURNS trigger AS $$
BEGIN
    IF NEW.document_id IS DISTINCT FROM OLD.document_id
        OR NEW.version_number IS DISTINCT FROM OLD.version_number
        OR NEW.object_key IS DISTINCT FROM OLD.object_key
        OR NEW.content_hash IS DISTINCT FROM OLD.content_hash
        OR NEW.file_size_bytes IS DISTINCT FROM OLD.file_size_bytes
        OR NEW.media_type IS DISTINCT FROM OLD.media_type
        OR NEW.embedding_model IS DISTINCT FROM OLD.embedding_model
        OR NEW.embedding_dimensions IS DISTINCT FROM OLD.embedding_dimensions
        OR NEW.ingestion_config IS DISTINCT FROM OLD.ingestion_config
        OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
        RAISE EXCEPTION 'immutable document version identity cannot be changed'
            USING ERRCODE = 'check_violation';
    END IF;

    IF OLD.status = 'succeeded'
        AND NEW.status = 'succeeded'
        AND OLD.activated_at IS NULL
        AND NEW.activated_at IS NOT NULL
        AND (to_jsonb(NEW) - ARRAY['activated_at', 'updated_at'])
            = (to_jsonb(OLD) - ARRAY['activated_at', 'updated_at']) THEN
        RETURN NEW;
    END IF;
"""

_FUNCTION_SUFFIX = """
    IF OLD.status = 'pending' AND NEW.status IN ('processing', 'failed') THEN
        RETURN NEW;
    END IF;
    IF OLD.status = 'processing' AND NEW.status IN ('succeeded', 'failed') THEN
        RETURN NEW;
    END IF;

    RAISE EXCEPTION 'invalid or terminal document version transition: % -> %',
        OLD.status, NEW.status USING ERRCODE = 'check_violation';
END;
$$ LANGUAGE plpgsql
"""


def upgrade() -> None:
    op.execute(
        _FUNCTION_PREFIX
        + """
    IF OLD.status = 'processing'
        AND NEW.status = 'processing'
        AND (to_jsonb(NEW) - ARRAY['extraction_metadata', 'updated_at'])
            = (to_jsonb(OLD) - ARRAY['extraction_metadata', 'updated_at']) THEN
        RETURN NEW;
    END IF;
"""
        + _FUNCTION_SUFFIX
    )


def downgrade() -> None:
    op.execute(_FUNCTION_PREFIX + _FUNCTION_SUFFIX)
