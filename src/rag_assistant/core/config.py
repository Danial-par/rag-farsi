from __future__ import annotations

from functools import lru_cache
from ipaddress import ip_network
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, Field, PostgresDsn, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated process configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    app_name: str = "Persian RAG Assistant"
    app_env: Literal["local", "test", "production"] = "local"
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"

    database_url: PostgresDsn
    database_pool_size: int = Field(default=5, ge=1, le=100)
    database_max_overflow: int = Field(default=10, ge=0, le=100)

    jwt_secret: SecretStr
    jwt_issuer: str = Field(default="rag-assistant", min_length=1, max_length=255)
    jwt_audience: str = Field(default="rag-assistant-api", min_length=1, max_length=255)
    access_token_ttl_seconds: int = Field(default=900, ge=60, le=3600)
    refresh_token_ttl_days: int = Field(default=30, ge=1, le=365)
    refresh_reuse_grace_seconds: int = Field(default=10, ge=0, le=60)
    allowed_origins: list[str] = Field(default_factory=list)
    trusted_proxy_hosts: list[str] = Field(default_factory=list)
    assistant_execution_timeout_seconds: int = Field(default=600, ge=60, le=3600)
    assistant_execution_recovery_poll_seconds: int = Field(default=30, ge=5, le=300)

    openrouter_base_url: AnyHttpUrl = AnyHttpUrl("https://openrouter.ai/api/v1")
    openrouter_api_key: SecretStr | None = None
    chat_model: str = "openai/gpt-6-luna"
    chat_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] = "none"
    chat_connect_timeout_seconds: float = Field(default=10, gt=0, le=120)
    chat_read_timeout_seconds: float = Field(default=60, gt=0, le=600)
    chat_max_retries: int = Field(default=2, ge=0, le=5)
    structure_annotation_enabled: bool = True
    retrieval_planner_enabled: bool = True
    retrieval_candidate_count: int = Field(default=30, ge=6, le=100)
    retrieval_max_expanded_sections: int = Field(default=2, ge=0, le=8)
    retrieval_section_max_characters: int = Field(default=24_000, ge=2_000, le=100_000)
    retrieval_outline_max_sections: int = Field(default=200, ge=1, le=1000)
    reranker_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:8081")
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_api_key: SecretStr | None = None
    reranker_timeout_seconds: float = Field(default=60, gt=0, le=600)
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: AnyHttpUrl = AnyHttpUrl("https://cloud.langfuse.com")
    embedding_model: str = "openai/text-embedding-3-large"
    embedding_dimensions: int = Field(default=1024, ge=1, le=4096)
    embedding_batch_size: int = Field(default=64, ge=1, le=512)
    embedding_concurrency: int = Field(default=2, ge=1, le=32)
    embedding_connect_timeout_seconds: float = Field(default=10, gt=0, le=120)
    embedding_read_timeout_seconds: float = Field(default=60, gt=0, le=600)
    embedding_max_retries: int = Field(default=3, ge=0, le=10)

    chunk_target_tokens: int = Field(default=400, ge=32, le=8192)
    chunk_max_tokens: int = Field(default=600, ge=32, le=8192)
    chunk_paragraph_overlap: int = Field(default=1, ge=0, le=1)
    ingestion_lease_seconds: int = Field(default=300, ge=30, le=3600)
    ingestion_poll_seconds: float = Field(default=2, gt=0, le=60)
    orphan_object_grace_seconds: int = Field(default=86400, ge=60, le=604800)
    orphan_reconcile_interval_seconds: int = Field(default=3600, ge=60, le=86400)
    max_pdf_upload_bytes: int = Field(default=25 * 1024 * 1024, ge=1, le=25 * 1024 * 1024)

    object_store_backend: Literal["local", "s3"] = "local"
    local_object_store_root: Path = Path(".local-data/objects")
    s3_bucket: str | None = None
    s3_endpoint_url: AnyHttpUrl | None = None
    s3_region: str | None = None
    s3_access_key_id: SecretStr | None = None
    s3_secret_access_key: SecretStr | None = None

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("s3_endpoint_url", mode="before")
    @classmethod
    def blank_optional_s3_endpoint_is_unset(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("reranker_api_key", mode="before")
    @classmethod
    def blank_optional_reranker_key_is_unset(cls, value: object) -> object:
        return None if value == "" else value

    @model_validator(mode="after")
    def require_psycopg_driver(self) -> Self:
        if self.database_url.scheme != "postgresql+psycopg":
            raise ValueError("DATABASE_URL must use the postgresql+psycopg scheme")
        if len(self.jwt_secret.get_secret_value().encode("utf-8")) < 32:
            raise ValueError("JWT_SECRET must contain at least 32 bytes")
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError("CHUNK_TARGET_TOKENS cannot exceed CHUNK_MAX_TOKENS")
        if (self.langfuse_public_key is None) != (self.langfuse_secret_key is None):
            raise ValueError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set together")
        if (self.s3_access_key_id is None) != (self.s3_secret_access_key is None):
            raise ValueError("S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY must be set together")
        if self.app_env == "production" and self.object_store_backend != "s3":
            raise ValueError("Production requires OBJECT_STORE_BACKEND=s3")
        if self.app_env == "production" and not self.s3_bucket:
            raise ValueError("S3_BUCKET is required in production")
        if self.app_env == "production":
            if self.openrouter_api_key is None:
                raise ValueError("OPENROUTER_API_KEY is required in production")
            if not self.allowed_origins:
                raise ValueError("ALLOWED_ORIGINS must contain at least one production origin")
        for origin in self.allowed_origins:
            parsed = urlsplit(origin)
            try:
                port = parsed.port
            except ValueError as exc:
                raise ValueError("ALLOWED_ORIGINS entries must be exact HTTP(S) origins") from exc
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path
                or (port is not None and not 0 < port < 65536)
                or parsed.query
                or parsed.fragment
                or origin == "*"
            ):
                raise ValueError("ALLOWED_ORIGINS entries must be exact HTTP(S) origins")
            if self.app_env == "production" and parsed.scheme != "https":
                raise ValueError("Production ALLOWED_ORIGINS must use HTTPS")
        reranker_host = urlsplit(str(self.reranker_base_url)).hostname
        if self.app_env == "production" and reranker_host in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("RERANKER_BASE_URL must point to the production reranker service")
        for proxy in self.trusted_proxy_hosts:
            try:
                ip_network(proxy, strict=False)
            except ValueError as exc:
                raise ValueError(
                    "TRUSTED_PROXY_HOSTS entries must be IP addresses or CIDRs"
                ) from exc
        return self

    @property
    def sqlalchemy_database_url(self) -> str:
        """Return the validated database URL in SQLAlchemy's string form."""
        return self.database_url.unicode_string()

    @property
    def secure_cookies(self) -> bool:
        """Require HTTPS cookies in production while allowing local HTTP development."""
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache the immutable process configuration."""
    return Settings()
