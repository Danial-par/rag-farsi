"""Add safe retriable indexing leases.

Revision ID: 20261017_0005
Revises: 20261011_0004
Create Date: 2026-10-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261017_0005"
down_revision: str | None = "20261011_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_IDENTITY_GUARD = """
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
"""


def upgrade() -> None:
    op.add_column(
        "document_versions",
        sa.Column("processing_attempt_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "document_versions",
        sa.Column("processing_lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        """
        UPDATE document_versions
        SET processing_attempt_id = md5(id::text || clock_timestamp()::text)::uuid,
            processing_lease_expires_at = GREATEST(now(), started_at + interval '1 second')
        WHERE status = 'processing'
        """
    )
    op.drop_constraint(
        "ck_document_versions_ingestion_lifecycle_valid",
        "document_versions",
        type_="check",
    )
    op.create_check_constraint(
        "ingestion_lifecycle_valid",
        "document_versions",
        "(status = 'pending' AND started_at IS NULL AND completed_at IS NULL "
        "AND failed_at IS NULL AND failure_code IS NULL AND failure_message IS NULL "
        "AND processing_attempt_id IS NULL AND processing_lease_expires_at IS NULL) OR "
        "(status = 'processing' AND started_at IS NOT NULL AND completed_at IS NULL "
        "AND failed_at IS NULL AND failure_code IS NULL AND failure_message IS NULL "
        "AND processing_attempt_id IS NOT NULL AND processing_lease_expires_at > started_at) OR "
        "(status = 'succeeded' AND started_at IS NOT NULL AND completed_at IS NOT NULL "
        "AND failed_at IS NULL AND failure_code IS NULL AND failure_message IS NULL "
        "AND processing_attempt_id IS NULL AND processing_lease_expires_at IS NULL) OR "
        "(status = 'failed' AND completed_at IS NULL AND failed_at IS NOT NULL "
        "AND failure_code IS NOT NULL AND failure_message IS NOT NULL "
        "AND processing_attempt_id IS NULL AND processing_lease_expires_at IS NULL)",
    )
    op.create_index(
        "ix_document_versions_processing_lease",
        "document_versions",
        ["status", "processing_lease_expires_at"],
    )
    op.drop_constraint(
        "ck_chunk_source_paragraphs_offsets_ordered",
        "chunk_source_paragraphs",
        type_="check",
    )
    op.create_check_constraint(
        "offsets_complete_and_ordered",
        "chunk_source_paragraphs",
        "(start_offset IS NULL AND end_offset IS NULL) OR "
        "(start_offset >= 0 AND end_offset > start_offset)",
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_document_version_update() RETURNS trigger AS $$
        BEGIN
        """
        + _IDENTITY_GUARD
        + """
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
    op.execute(
        """
        UPDATE document_versions
        SET status = 'failed',
            failed_at = now(),
            failure_code = 'migration_downgrade',
            failure_message = 'Processing attempt stopped during schema downgrade',
            processing_attempt_id = NULL,
            processing_lease_expires_at = NULL
        WHERE status = 'processing'
        """
    )
    op.drop_index("ix_document_versions_processing_lease", table_name="document_versions")
    op.drop_constraint(
        "ingestion_lifecycle_valid",
        "document_versions",
        type_="check",
    )
    op.create_check_constraint(
        "ingestion_lifecycle_valid",
        "document_versions",
        "(status = 'pending' AND started_at IS NULL AND completed_at IS NULL "
        "AND failed_at IS NULL AND failure_code IS NULL AND failure_message IS NULL) OR "
        "(status = 'processing' AND started_at IS NOT NULL AND completed_at IS NULL "
        "AND failed_at IS NULL AND failure_code IS NULL AND failure_message IS NULL) OR "
        "(status = 'succeeded' AND started_at IS NOT NULL AND completed_at IS NOT NULL "
        "AND failed_at IS NULL AND failure_code IS NULL AND failure_message IS NULL) OR "
        "(status = 'failed' AND completed_at IS NULL AND failed_at IS NOT NULL "
        "AND failure_code IS NOT NULL AND failure_message IS NOT NULL)",
    )
    op.drop_constraint(
        "offsets_complete_and_ordered",
        "chunk_source_paragraphs",
        type_="check",
    )
    op.create_check_constraint(
        "offsets_ordered",
        "chunk_source_paragraphs",
        "end_offset IS NULL OR end_offset > start_offset",
    )
    op.drop_column("document_versions", "processing_lease_expires_at")
    op.drop_column("document_versions", "processing_attempt_id")
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_document_version_update() RETURNS trigger AS $$
        BEGIN
        """
        + _IDENTITY_GUARD
        + """
            IF OLD.status = 'succeeded'
                AND NEW.status = 'succeeded'
                AND OLD.activated_at IS NULL
                AND NEW.activated_at IS NOT NULL
                AND (to_jsonb(NEW) - ARRAY['activated_at', 'updated_at'])
                    = (to_jsonb(OLD) - ARRAY['activated_at', 'updated_at']) THEN
                RETURN NEW;
            END IF;
            IF OLD.status = 'processing' AND NEW.status = 'processing'
                AND (to_jsonb(NEW) - ARRAY['extraction_metadata', 'updated_at'])
                    = (to_jsonb(OLD) - ARRAY['extraction_metadata', 'updated_at']) THEN
                RETURN NEW;
            END IF;
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
    )
