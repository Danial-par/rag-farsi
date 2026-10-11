"""Create the core persistence schema.

Revision ID: 20261011_0002
Revises: 20261011_0001
Create Date: 2026-10-11
"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261011_0002"
down_revision: str | None = "20261011_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("normalized_email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=512), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("length(btrim(email)) > 0", name="ck_users_email_not_blank"),
        sa.CheckConstraint(
            "length(btrim(normalized_email)) > 0",
            name="ck_users_normalized_email_not_blank",
        ),
        sa.CheckConstraint("length(password_hash) > 0", name="ck_users_password_hash_not_blank"),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("normalized_email", name="uq_users_normalized_email"),
    )

    op.create_table(
        "auth_sessions",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revocation_reason", sa.Text(), nullable=True),
        sa.Column("client_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "jsonb_typeof(client_metadata) = 'object'",
            name="ck_auth_sessions_client_metadata_is_object",
        ),
        sa.CheckConstraint(
            "expires_at > created_at", name="ck_auth_sessions_expiry_after_creation"
        ),
        sa.CheckConstraint(
            "revoked_at IS NULL OR revoked_at >= created_at",
            name="ck_auth_sessions_revocation_after_creation",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_auth_sessions_user_id_users", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_auth_sessions"),
    )
    op.create_index(
        "ix_auth_sessions_user_id_expires_at",
        "auth_sessions",
        ["user_id", "expires_at"],
    )

    op.create_table(
        "documents",
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("is_shared", sa.Boolean(), nullable=False),
        sa.Column("is_default", sa.Boolean(), nullable=False),
        sa.Column("active_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("document_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "NOT is_default OR is_shared", name="ck_documents_default_document_is_shared"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(document_metadata) = 'object'",
            name="ck_documents_document_metadata_is_object",
        ),
        sa.CheckConstraint(
            "(is_shared AND owner_id IS NULL) OR (NOT is_shared AND owner_id IS NOT NULL)",
            name="ck_documents_ownership_matches_sharing",
        ),
        sa.CheckConstraint("length(btrim(title)) > 0", name="ck_documents_title_not_blank"),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="fk_documents_owner_id_users", ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
    )
    op.create_index("ix_documents_owner_id_deleted_at", "documents", ["owner_id", "deleted_at"])
    op.create_index(
        "ix_documents_shared_default_active",
        "documents",
        ["is_shared", "is_default", "active_version_id"],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    op.create_table(
        "document_versions",
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("object_key", sa.String(length=1024), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("media_type", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("ingestion_config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("extraction_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_code", sa.String(length=100), nullable=True),
        sa.Column("failure_message", sa.Text(), nullable=True),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "activated_at IS NULL OR status = 'succeeded'",
            name="ck_document_versions_activation_requires_success",
        ),
        sa.CheckConstraint(
            "completed_at IS NULL OR completed_at >= started_at",
            name="ck_document_versions_completion_after_start",
        ),
        sa.CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'",
            name="ck_document_versions_content_hash_sha256",
        ),
        sa.CheckConstraint(
            "embedding_dimensions > 0",
            name="ck_document_versions_embedding_dimensions_positive",
        ),
        sa.CheckConstraint(
            "length(btrim(embedding_model)) > 0",
            name="ck_document_versions_embedding_model_not_blank",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(extraction_metadata) = 'object'",
            name="ck_document_versions_extraction_metadata_is_object",
        ),
        sa.CheckConstraint(
            "failed_at IS NULL OR started_at IS NULL OR failed_at >= started_at",
            name="ck_document_versions_failure_after_start",
        ),
        sa.CheckConstraint("file_size_bytes > 0", name="ck_document_versions_file_size_positive"),
        sa.CheckConstraint(
            "jsonb_typeof(ingestion_config) = 'object'",
            name="ck_document_versions_ingestion_config_is_object",
        ),
        sa.CheckConstraint(
            "(status = 'pending' AND started_at IS NULL AND completed_at IS NULL "
            "AND failed_at IS NULL AND failure_code IS NULL AND failure_message IS NULL) OR "
            "(status = 'processing' AND started_at IS NOT NULL AND completed_at IS NULL "
            "AND failed_at IS NULL AND failure_code IS NULL AND failure_message IS NULL) OR "
            "(status = 'succeeded' AND started_at IS NOT NULL AND completed_at IS NOT NULL "
            "AND failed_at IS NULL AND failure_code IS NULL AND failure_message IS NULL) OR "
            "(status = 'failed' AND completed_at IS NULL AND failed_at IS NOT NULL "
            "AND failure_code IS NOT NULL AND failure_message IS NOT NULL)",
            name="ck_document_versions_ingestion_lifecycle_valid",
        ),
        sa.CheckConstraint(
            "length(btrim(media_type)) > 0", name="ck_document_versions_media_type_not_blank"
        ),
        sa.CheckConstraint(
            "length(btrim(object_key)) > 0", name="ck_document_versions_object_key_not_blank"
        ),
        sa.CheckConstraint(
            "version_number >= 1", name="ck_document_versions_version_number_positive"
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_versions_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_versions"),
        sa.UniqueConstraint(
            "document_id",
            "version_number",
            name="uq_document_versions_document_id_version_number",
        ),
        sa.UniqueConstraint("id", "document_id", name="uq_document_versions_id_document_id"),
        sa.UniqueConstraint(
            "id",
            "embedding_model",
            "embedding_dimensions",
            name="uq_document_versions_id_embedding_configuration",
        ),
        sa.UniqueConstraint("object_key", name="uq_document_versions_object_key"),
    )
    op.create_index("ix_document_versions_content_hash", "document_versions", ["content_hash"])
    op.create_index(
        "ix_document_versions_document_id_status",
        "document_versions",
        ["document_id", "status"],
    )
    op.create_index(
        "ix_document_versions_embedding_configuration",
        "document_versions",
        ["embedding_model", "embedding_dimensions"],
    )
    op.create_foreign_key(
        "fk_documents_active_version_belongs_to_document",
        "documents",
        "document_versions",
        ["active_version_id", "id"],
        ["id", "document_id"],
        deferrable=True,
        initially="DEFERRED",
    )

    op.create_table(
        "source_paragraphs",
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("paragraph_number", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("original_text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("paragraph_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(normalized_text) > 0",
            name="ck_source_paragraphs_normalized_text_not_empty",
        ),
        sa.CheckConstraint(
            "length(original_text) > 0", name="ck_source_paragraphs_original_text_not_empty"
        ),
        sa.CheckConstraint("page_number >= 1", name="ck_source_paragraphs_page_number_one_based"),
        sa.CheckConstraint(
            "paragraph_number >= 1",
            name="ck_source_paragraphs_paragraph_number_one_based",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(paragraph_metadata) = 'object'",
            name="ck_source_paragraphs_paragraph_metadata_is_object",
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id"],
            ["document_versions.id"],
            name="fk_source_paragraphs_document_version_id_document_versions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_source_paragraphs"),
        sa.UniqueConstraint(
            "document_version_id",
            "page_number",
            "paragraph_number",
            name="uq_source_paragraphs_citation_identity",
        ),
        sa.UniqueConstraint(
            "id",
            "document_version_id",
            name="uq_source_paragraphs_id_document_version_id",
        ),
    )
    op.create_index(
        "ix_source_paragraphs_document_version_id_page_number",
        "source_paragraphs",
        ["document_version_id", "page_number"],
    )

    op.create_table(
        "chunks",
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=1024), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("chunk_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("provenance_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "embedding_dimensions = 1024", name="ck_chunks_embedding_dimensions_supported"
        ),
        sa.CheckConstraint(
            "vector_dims(embedding) = 1024", name="ck_chunks_embedding_vector_dimensions"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(chunk_metadata) = 'object'",
            name="ck_chunks_chunk_metadata_is_object",
        ),
        sa.CheckConstraint(
            "length(normalized_text) > 0", name="ck_chunks_normalized_text_not_empty"
        ),
        sa.CheckConstraint("ordinal >= 1", name="ck_chunks_ordinal_positive"),
        sa.CheckConstraint(
            "jsonb_typeof(provenance_snapshot) = 'array'",
            name="ck_chunks_provenance_snapshot_is_array",
        ),
        sa.CheckConstraint("token_count > 0", name="ck_chunks_token_count_positive"),
        sa.ForeignKeyConstraint(
            ["document_version_id", "embedding_model", "embedding_dimensions"],
            [
                "document_versions.id",
                "document_versions.embedding_model",
                "document_versions.embedding_dimensions",
            ],
            name="fk_chunks_embedding_configuration_document_versions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_chunks"),
        sa.UniqueConstraint(
            "document_version_id", "ordinal", name="uq_chunks_document_version_id_ordinal"
        ),
        sa.UniqueConstraint("id", "document_version_id", name="uq_chunks_id_document_version_id"),
    )
    op.create_index(
        "ix_chunks_document_version_id_embedding_configuration",
        "chunks",
        ["document_version_id", "embedding_model", "embedding_dimensions"],
    )
    op.create_index(
        "ix_chunks_embedding_hnsw_cosine",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "chunk_source_paragraphs",
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_paragraph_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=True),
        sa.Column("end_offset", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "end_offset IS NULL OR end_offset > start_offset",
            name="ck_chunk_source_paragraphs_offsets_ordered",
        ),
        sa.CheckConstraint("ordinal >= 1", name="ck_chunk_source_paragraphs_ordinal_positive"),
        sa.ForeignKeyConstraint(
            ["chunk_id", "document_version_id"],
            ["chunks.id", "chunks.document_version_id"],
            name="fk_chunk_source_paragraphs_chunk_version",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_paragraph_id", "document_version_id"],
            ["source_paragraphs.id", "source_paragraphs.document_version_id"],
            name="fk_chunk_source_paragraphs_paragraph_version",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "chunk_id", "source_paragraph_id", name="pk_chunk_source_paragraphs"
        ),
        sa.UniqueConstraint(
            "chunk_id", "ordinal", name="uq_chunk_source_paragraphs_chunk_id_ordinal"
        ),
    )

    op.create_table(
        "conversations",
        sa.Column("owner_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("conversation_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "jsonb_typeof(conversation_metadata) = 'object'",
            name="ck_conversations_conversation_metadata_is_object",
        ),
        sa.CheckConstraint(
            "title IS NULL OR length(btrim(title)) > 0",
            name="ck_conversations_title_not_blank",
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name="fk_conversations_owner_id_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_conversations"),
    )
    op.create_index(
        "ix_conversations_owner_id_updated_at",
        "conversations",
        ["owner_id", "updated_at"],
    )

    op.create_table(
        "conversation_documents",
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_conversation_documents_conversation_id_conversations",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_conversation_documents_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("conversation_id", "document_id", name="pk_conversation_documents"),
    )

    op.create_table(
        "messages",
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("answer_outcome", sa.String(length=24), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("citation_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("model_usage", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("provider_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "jsonb_typeof(citation_snapshot) = 'array'",
            name="ck_messages_citation_snapshot_is_array",
        ),
        sa.CheckConstraint(
            "(role = 'user' AND status = 'completed' AND content IS NOT NULL "
            "AND length(content) > 0 AND answer_outcome IS NULL AND error_code IS NULL "
            "AND error_message IS NULL AND jsonb_array_length(citation_snapshot) = 0) OR "
            "(role = 'assistant' AND status = 'pending' AND content IS NULL "
            "AND answer_outcome IS NULL AND error_code IS NULL AND error_message IS NULL "
            "AND jsonb_array_length(citation_snapshot) = 0) OR "
            "(role = 'assistant' AND status = 'failed' AND content IS NULL "
            "AND answer_outcome IS NULL AND error_code IS NOT NULL AND error_message IS NOT NULL "
            "AND jsonb_array_length(citation_snapshot) = 0) OR "
            "(role = 'assistant' AND status = 'completed' AND content IS NOT NULL "
            "AND length(content) > 0 AND answer_outcome IS NOT NULL AND error_code IS NULL "
            "AND error_message IS NULL AND ((answer_outcome = 'abstained' "
            "AND jsonb_array_length(citation_snapshot) = 0) OR "
            "(answer_outcome IN ('fully_answered', 'partially_answered') "
            "AND jsonb_array_length(citation_snapshot) > 0)))",
            name="ck_messages_message_state_valid",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(model_usage) = 'object'", name="ck_messages_model_usage_is_object"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(provider_metadata) = 'object'",
            name="ck_messages_provider_metadata_is_object",
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name="fk_messages_conversation_id_conversations",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_messages"),
    )
    op.create_index(
        "ix_messages_conversation_id_created_at",
        "messages",
        ["conversation_id", "created_at"],
    )

    op.execute(
        """
        CREATE FUNCTION reject_immutable_row_update() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'immutable ingestion records cannot be updated'
                USING ERRCODE = 'check_violation';
        END;
        $$ LANGUAGE plpgsql
        """
    )
    for table_name in ("source_paragraphs", "chunks", "chunk_source_paragraphs"):
        op.execute(
            f"""
            CREATE TRIGGER trg_{table_name}_reject_update
            BEFORE UPDATE ON {table_name}
            FOR EACH ROW EXECUTE FUNCTION reject_immutable_row_update()
            """
        )

    op.execute(
        """
        CREATE FUNCTION enforce_document_version_update() RETURNS trigger AS $$
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
            IF OLD.status = 'processing' AND NEW.status IN ('succeeded', 'failed') THEN
                RETURN NEW;
            END IF;

            RAISE EXCEPTION 'invalid or terminal document version transition: % -> %',
                OLD.status, NEW.status USING ERRCODE = 'check_violation';
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_document_versions_enforce_update
        BEFORE UPDATE ON document_versions
        FOR EACH ROW EXECUTE FUNCTION enforce_document_version_update()
        """
    )
    op.execute(
        """
        CREATE FUNCTION enforce_active_document_version() RETURNS trigger AS $$
        BEGIN
            IF NEW.active_version_id IS NOT NULL
                AND NEW.active_version_id IS DISTINCT FROM OLD.active_version_id
                AND NOT EXISTS (
                    SELECT 1 FROM document_versions
                    WHERE id = NEW.active_version_id
                      AND document_id = NEW.id
                      AND status = 'succeeded'
                      AND activated_at IS NOT NULL
                ) THEN
                RAISE EXCEPTION 'active version must be a successful version of this document'
                    USING ERRCODE = 'check_violation';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_documents_enforce_active_version
        BEFORE UPDATE OF active_version_id ON documents
        FOR EACH ROW EXECUTE FUNCTION enforce_active_document_version()
        """
    )
    op.execute(
        """
        CREATE FUNCTION enforce_message_update() RETURNS trigger AS $$
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
    op.execute(
        """
        CREATE TRIGGER trg_messages_enforce_update
        BEFORE UPDATE ON messages
        FOR EACH ROW EXECUTE FUNCTION enforce_message_update()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_messages_enforce_update ON messages")
    op.execute("DROP FUNCTION IF EXISTS enforce_message_update")
    op.execute("DROP TRIGGER IF EXISTS trg_documents_enforce_active_version ON documents")
    op.execute("DROP FUNCTION IF EXISTS enforce_active_document_version")
    op.execute("DROP TRIGGER IF EXISTS trg_document_versions_enforce_update ON document_versions")
    op.execute("DROP FUNCTION IF EXISTS enforce_document_version_update")
    for table_name in ("source_paragraphs", "chunks", "chunk_source_paragraphs"):
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table_name}_reject_update ON {table_name}")
    op.execute("DROP FUNCTION IF EXISTS reject_immutable_row_update")

    op.drop_index("ix_messages_conversation_id_created_at", table_name="messages")
    op.drop_table("messages")
    op.drop_table("conversation_documents")
    op.drop_index("ix_conversations_owner_id_updated_at", table_name="conversations")
    op.drop_table("conversations")
    op.drop_table("chunk_source_paragraphs")
    op.drop_index("ix_chunks_embedding_hnsw_cosine", table_name="chunks", postgresql_using="hnsw")
    op.drop_index("ix_chunks_document_version_id_embedding_configuration", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index(
        "ix_source_paragraphs_document_version_id_page_number",
        table_name="source_paragraphs",
    )
    op.drop_table("source_paragraphs")
    op.drop_constraint(
        "fk_documents_active_version_belongs_to_document",
        "documents",
        type_="foreignkey",
    )
    op.drop_index("ix_document_versions_embedding_configuration", table_name="document_versions")
    op.drop_index("ix_document_versions_document_id_status", table_name="document_versions")
    op.drop_index("ix_document_versions_content_hash", table_name="document_versions")
    op.drop_table("document_versions")
    op.drop_index("ix_documents_shared_default_active", table_name="documents")
    op.drop_index("ix_documents_owner_id_deleted_at", table_name="documents")
    op.drop_table("documents")
    op.drop_index("ix_auth_sessions_user_id_expires_at", table_name="auth_sessions")
    op.drop_table("auth_sessions")
    op.drop_table("users")
