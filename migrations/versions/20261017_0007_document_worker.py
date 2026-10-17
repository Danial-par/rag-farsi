"""Support durable document uploads and worker retries.

Revision ID: 20261017_0007
Revises: 20261017_0006
Create Date: 2026-10-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261017_0007"
down_revision: str | None = "20261017_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_index(
        "ix_document_versions_worker_queue",
        "document_versions",
        ["created_at", "id"],
        postgresql_where=sa.text("status IN ('pending', 'processing') AND activated_at IS NULL"),
    )
    op.execute(
        """
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
            IF OLD.status = 'pending' AND NEW.status IN ('processing', 'failed') THEN
                RETURN NEW;
            END IF;
            IF OLD.status = 'failed' AND NEW.status = 'pending'
                AND OLD.activated_at IS NULL
                AND NEW.started_at IS NULL AND NEW.failed_at IS NULL
                AND NEW.failure_code IS NULL AND NEW.failure_message IS NULL
                AND NEW.processing_attempt_id IS NULL
                AND NEW.processing_lease_expires_at IS NULL THEN
                RETURN NEW;
            END IF;
            IF OLD.status = 'failed' AND NEW.status = 'processing'
                AND OLD.activated_at IS NULL THEN
                RETURN NEW;
            END IF;
            IF OLD.status = 'processing' AND NEW.status = 'processing'
                AND (to_jsonb(NEW) - ARRAY[
                    'started_at', 'processing_attempt_id', 'processing_lease_expires_at',
                    'extraction_metadata', 'updated_at'
                ]) = (to_jsonb(OLD) - ARRAY[
                    'started_at', 'processing_attempt_id', 'processing_lease_expires_at',
                    'extraction_metadata', 'updated_at'
                ]) THEN
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
    )


def downgrade() -> None:
    op.drop_index("ix_document_versions_worker_queue", table_name="document_versions")
    op.execute(
        """
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
            IF OLD.status = 'succeeded' AND NEW.status = 'succeeded'
                AND OLD.activated_at IS NULL AND NEW.activated_at IS NOT NULL
                AND (to_jsonb(NEW) - ARRAY['activated_at', 'updated_at'])
                    = (to_jsonb(OLD) - ARRAY['activated_at', 'updated_at']) THEN
                RETURN NEW;
            END IF;
            IF OLD.status = 'pending' AND NEW.status IN ('processing', 'failed') THEN
                RETURN NEW;
            END IF;
            IF OLD.status = 'failed' AND NEW.status = 'processing'
                AND OLD.activated_at IS NULL THEN
                RETURN NEW;
            END IF;
            IF OLD.status = 'processing' AND NEW.status = 'processing'
                AND (to_jsonb(NEW) - ARRAY[
                    'started_at', 'processing_attempt_id', 'processing_lease_expires_at',
                    'extraction_metadata', 'updated_at'
                ]) = (to_jsonb(OLD) - ARRAY[
                    'started_at', 'processing_attempt_id', 'processing_lease_expires_at',
                    'extraction_metadata', 'updated_at'
                ]) THEN
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
    )
