"""Async ChatOpenAI client configured for the OpenRouter-compatible gateway."""

from __future__ import annotations

import httpx
from langchain_openai import ChatOpenAI

from rag_assistant.core.config import Settings


def create_chat_model(settings: Settings) -> ChatOpenAI:
    """Create the configured chat client without exposing its secret key."""
    if settings.openrouter_api_key is None:
        raise ValueError("OPENROUTER_API_KEY is required for model calls")
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=str(settings.openrouter_base_url),
        timeout=httpx.Timeout(
            connect=settings.chat_connect_timeout_seconds,
            read=settings.chat_read_timeout_seconds,
            write=settings.chat_read_timeout_seconds,
            pool=settings.chat_connect_timeout_seconds,
        ),
        max_retries=settings.chat_max_retries,
        reasoning_effort=settings.chat_reasoning_effort,
    )
