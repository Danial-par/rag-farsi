from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

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

    openrouter_base_url: AnyHttpUrl = AnyHttpUrl("https://openrouter.ai/api/v1")
    openrouter_api_key: SecretStr | None = None
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

    object_store_backend: Literal["local", "s3"] = "local"
    local_object_store_root: Path = Path(".")
    s3_bucket: str | None = None
    s3_endpoint_url: AnyHttpUrl | None = None
    s3_region: str | None = None
    s3_access_key_id: SecretStr | None = None
    s3_secret_access_key: SecretStr | None = None

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def require_psycopg_driver(self) -> Self:
        if self.database_url.scheme != "postgresql+psycopg":
            raise ValueError("DATABASE_URL must use the postgresql+psycopg scheme")
        if len(self.jwt_secret.get_secret_value().encode("utf-8")) < 32:
            raise ValueError("JWT_SECRET must contain at least 32 bytes")
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError("CHUNK_TARGET_TOKENS cannot exceed CHUNK_MAX_TOKENS")
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
