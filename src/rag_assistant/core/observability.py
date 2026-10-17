"""Initialization and shutdown for configured Langfuse tracing."""

from __future__ import annotations

from langfuse import Langfuse, get_client

from rag_assistant.core.config import Settings


def configure_langfuse(settings: Settings) -> bool:
    """Register the Langfuse project loaded by application settings.

    Langfuse reads process environment variables directly and does not consume the
    ``.env`` file that Pydantic Settings reads. Pass the effective settings into the
    SDK explicitly so local ``.env`` configuration works without mutating process
    environment state.
    """
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return False
    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=str(settings.langfuse_base_url),
    )
    return True


def shutdown_langfuse(settings: Settings) -> None:
    """Flush the configured project without implicitly creating a disabled client."""
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return
    client = get_client(public_key=settings.langfuse_public_key)
    client.flush()
    client.shutdown()
