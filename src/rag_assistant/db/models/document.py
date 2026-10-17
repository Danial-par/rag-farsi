from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_assistant.db.base import (
    Base,
    CreatedAtMixin,
    UpdatedAtMixin,
    UUIDPrimaryKeyMixin,
)
from rag_assistant.db.models.enums import IngestionStatus, ParagraphKind

if TYPE_CHECKING:
    from rag_assistant.db.models.conversation import ConversationDocument
    from rag_assistant.db.models.user import User

EMBEDDING_DIMENSIONS = 1024


def _enum_values(enum_class: type[IngestionStatus] | type[ParagraphKind]) -> list[str]:
    return [member.value for member in enum_class]


class Document(UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("length(btrim(title)) > 0", name="title_not_blank"),
        CheckConstraint(
            "(is_shared AND owner_id IS NULL) OR (NOT is_shared AND owner_id IS NOT NULL)",
            name="ownership_matches_sharing",
        ),
        CheckConstraint("NOT is_default OR is_shared", name="default_document_is_shared"),
        CheckConstraint(
            "jsonb_typeof(document_metadata) = 'object'", name="document_metadata_is_object"
        ),
        ForeignKeyConstraint(
            ["active_version_id", "id"],
            ["document_versions.id", "document_versions.document_id"],
            name="fk_documents_active_version_belongs_to_document",
            deferrable=True,
            initially="DEFERRED",
            use_alter=True,
        ),
        Index("ix_documents_owner_id_deleted_at", "owner_id", "deleted_at"),
        Index(
            "ix_documents_shared_default_active",
            "is_shared",
            "is_default",
            "active_version_id",
            postgresql_where=text("deleted_at IS NULL"),
        ),
    )

    owner_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    is_shared: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    active_version_id: Mapped[UUID | None] = mapped_column(nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    document_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    owner: Mapped[User | None] = relationship(back_populates="documents")
    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        foreign_keys="DocumentVersion.document_id",
    )
    active_version: Mapped[DocumentVersion | None] = relationship(
        primaryjoin="Document.active_version_id == DocumentVersion.id",
        foreign_keys=[active_version_id],
        viewonly=True,
    )
    conversation_links: Mapped[list[ConversationDocument]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentVersion(UUIDPrimaryKeyMixin, CreatedAtMixin, UpdatedAtMixin, Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "version_number"),
        UniqueConstraint("id", "document_id", name="uq_document_versions_id_document_id"),
        UniqueConstraint(
            "id",
            "embedding_model",
            "embedding_dimensions",
            name="uq_document_versions_id_embedding_configuration",
        ),
        CheckConstraint("version_number >= 1", name="version_number_positive"),
        CheckConstraint("file_size_bytes > 0", name="file_size_positive"),
        CheckConstraint("content_hash ~ '^[0-9a-f]{64}$'", name="content_hash_sha256"),
        CheckConstraint("length(btrim(object_key)) > 0", name="object_key_not_blank"),
        CheckConstraint("length(btrim(media_type)) > 0", name="media_type_not_blank"),
        CheckConstraint("length(btrim(embedding_model)) > 0", name="embedding_model_not_blank"),
        CheckConstraint("embedding_dimensions > 0", name="embedding_dimensions_positive"),
        CheckConstraint(
            "jsonb_typeof(ingestion_config) = 'object'", name="ingestion_config_is_object"
        ),
        CheckConstraint(
            "jsonb_typeof(extraction_metadata) = 'object'",
            name="extraction_metadata_is_object",
        ),
        CheckConstraint(
            "(status = 'pending' AND started_at IS NULL AND completed_at IS NULL "
            "AND failed_at IS NULL AND failure_code IS NULL AND failure_message IS NULL "
            "AND processing_attempt_id IS NULL AND processing_lease_expires_at IS NULL) OR "
            "(status = 'processing' AND started_at IS NOT NULL AND completed_at IS NULL "
            "AND failed_at IS NULL AND failure_code IS NULL AND failure_message IS NULL "
            "AND processing_attempt_id IS NOT NULL "
            "AND processing_lease_expires_at > started_at) OR "
            "(status = 'succeeded' AND started_at IS NOT NULL AND completed_at IS NOT NULL "
            "AND failed_at IS NULL AND failure_code IS NULL AND failure_message IS NULL "
            "AND processing_attempt_id IS NULL AND processing_lease_expires_at IS NULL) OR "
            "(status = 'failed' AND completed_at IS NULL AND failed_at IS NOT NULL "
            "AND failure_code IS NOT NULL AND failure_message IS NOT NULL "
            "AND processing_attempt_id IS NULL AND processing_lease_expires_at IS NULL)",
            name="ingestion_lifecycle_valid",
        ),
        CheckConstraint(
            "completed_at IS NULL OR completed_at >= started_at", name="completion_after_start"
        ),
        CheckConstraint(
            "failed_at IS NULL OR started_at IS NULL OR failed_at >= started_at",
            name="failure_after_start",
        ),
        CheckConstraint(
            "activated_at IS NULL OR status = 'succeeded'", name="activation_requires_success"
        ),
        Index("ix_document_versions_document_id_status", "document_id", "status"),
        Index("ix_document_versions_content_hash", "content_hash"),
        Index("ix_document_versions_processing_lease", "status", "processing_lease_expires_at"),
        Index(
            "ix_document_versions_embedding_configuration",
            "embedding_model",
            "embedding_dimensions",
        ),
    )

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False, unique=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    media_type: Mapped[str] = mapped_column(String(255), nullable=False, default="application/pdf")
    status: Mapped[IngestionStatus] = mapped_column(
        Enum(
            IngestionStatus,
            values_callable=_enum_values,
            native_enum=False,
            create_constraint=False,
            length=16,
        ),
        nullable=False,
        default=IngestionStatus.PENDING,
    )
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    ingestion_config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    extraction_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    processing_attempt_id: Mapped[UUID | None] = mapped_column(nullable=True)
    processing_lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    failure_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    failure_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    document: Mapped[Document] = relationship(back_populates="versions", foreign_keys=[document_id])
    source_paragraphs: Mapped[list[SourceParagraph]] = relationship(
        back_populates="document_version", cascade="all, delete-orphan"
    )
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document_version", cascade="all, delete-orphan"
    )


class SourceParagraph(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "source_paragraphs"
    __table_args__ = (
        UniqueConstraint(
            "document_version_id",
            "page_number",
            "paragraph_number",
            name="uq_source_paragraphs_citation_identity",
        ),
        UniqueConstraint(
            "id", "document_version_id", name="uq_source_paragraphs_id_document_version_id"
        ),
        CheckConstraint("page_number >= 1", name="page_number_one_based"),
        CheckConstraint("paragraph_number >= 1", name="paragraph_number_one_based"),
        CheckConstraint("length(original_text) > 0", name="original_text_not_empty"),
        CheckConstraint("length(normalized_text) > 0", name="normalized_text_not_empty"),
        CheckConstraint(
            "jsonb_typeof(paragraph_metadata) = 'object'", name="paragraph_metadata_is_object"
        ),
        Index(
            "ix_source_paragraphs_document_version_id_page_number",
            "document_version_id",
            "page_number",
        ),
    )

    document_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    paragraph_number: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[ParagraphKind] = mapped_column(
        Enum(
            ParagraphKind,
            values_callable=_enum_values,
            native_enum=False,
            create_constraint=False,
            length=16,
        ),
        nullable=False,
    )
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    paragraph_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    document_version: Mapped[DocumentVersion] = relationship(back_populates="source_paragraphs")
    chunk_links: Mapped[list[ChunkSourceParagraph]] = relationship(
        back_populates="source_paragraph",
        cascade="all, delete-orphan",
        overlaps="chunk,paragraph_links",
    )


class Chunk(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_version_id", "ordinal"),
        UniqueConstraint("id", "document_version_id", name="uq_chunks_id_document_version_id"),
        ForeignKeyConstraint(
            ["document_version_id", "embedding_model", "embedding_dimensions"],
            [
                "document_versions.id",
                "document_versions.embedding_model",
                "document_versions.embedding_dimensions",
            ],
            name="fk_chunks_embedding_configuration_document_versions",
            ondelete="CASCADE",
        ),
        CheckConstraint("ordinal >= 1", name="ordinal_positive"),
        CheckConstraint("length(normalized_text) > 0", name="normalized_text_not_empty"),
        CheckConstraint("token_count > 0", name="token_count_positive"),
        CheckConstraint(
            f"embedding_dimensions = {EMBEDDING_DIMENSIONS}",
            name="embedding_dimensions_supported",
        ),
        CheckConstraint(
            f"vector_dims(embedding) = {EMBEDDING_DIMENSIONS}", name="embedding_vector_dimensions"
        ),
        CheckConstraint("jsonb_typeof(chunk_metadata) = 'object'", name="chunk_metadata_is_object"),
        CheckConstraint(
            "jsonb_typeof(provenance_snapshot) = 'array'",
            name="provenance_snapshot_is_array",
        ),
        Index(
            "ix_chunks_document_version_id_embedding_configuration",
            "document_version_id",
            "embedding_model",
            "embedding_dimensions",
        ),
        Index(
            "ix_chunks_embedding_hnsw_cosine",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    document_version_id: Mapped[UUID] = mapped_column(nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSIONS), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    provenance_snapshot: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )

    document_version: Mapped[DocumentVersion] = relationship(
        back_populates="chunks",
        foreign_keys=[document_version_id],
        primaryjoin="Chunk.document_version_id == DocumentVersion.id",
    )
    paragraph_links: Mapped[list[ChunkSourceParagraph]] = relationship(
        back_populates="chunk",
        cascade="all, delete-orphan",
        order_by="ChunkSourceParagraph.ordinal",
        overlaps="chunk_links,source_paragraph",
    )


class ChunkSourceParagraph(CreatedAtMixin, Base):
    __tablename__ = "chunk_source_paragraphs"
    __table_args__ = (
        ForeignKeyConstraint(
            ["chunk_id", "document_version_id"],
            ["chunks.id", "chunks.document_version_id"],
            name="fk_chunk_source_paragraphs_chunk_version",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["source_paragraph_id", "document_version_id"],
            ["source_paragraphs.id", "source_paragraphs.document_version_id"],
            name="fk_chunk_source_paragraphs_paragraph_version",
            ondelete="CASCADE",
        ),
        UniqueConstraint("chunk_id", "ordinal"),
        CheckConstraint("ordinal >= 1", name="ordinal_positive"),
        CheckConstraint(
            "(start_offset IS NULL AND end_offset IS NULL) OR "
            "(start_offset >= 0 AND end_offset > start_offset)",
            name="offsets_complete_and_ordered",
        ),
    )

    chunk_id: Mapped[UUID] = mapped_column(primary_key=True)
    source_paragraph_id: Mapped[UUID] = mapped_column(primary_key=True)
    document_version_id: Mapped[UUID] = mapped_column(nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    start_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)

    chunk: Mapped[Chunk] = relationship(
        back_populates="paragraph_links", overlaps="chunk_links,source_paragraph"
    )
    source_paragraph: Mapped[SourceParagraph] = relationship(
        back_populates="chunk_links", overlaps="chunk,paragraph_links"
    )
