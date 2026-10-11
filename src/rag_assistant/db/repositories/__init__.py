"""Explicit async persistence boundaries for application aggregates."""

from rag_assistant.db.repositories.auth_sessions import AuthSessionRepository
from rag_assistant.db.repositories.conversations import ConversationRepository
from rag_assistant.db.repositories.documents import (
    ChunkParagraphReference,
    DocumentRepository,
    NewChunk,
    NewDocumentVersion,
    NewSourceParagraph,
)
from rag_assistant.db.repositories.errors import (
    InvalidPersistenceStateError,
    PersistenceConflictError,
    RecordNotFoundError,
    RepositoryError,
)
from rag_assistant.db.repositories.users import UserRepository, normalize_email

__all__ = [
    "AuthSessionRepository",
    "ChunkParagraphReference",
    "ConversationRepository",
    "DocumentRepository",
    "InvalidPersistenceStateError",
    "NewChunk",
    "NewDocumentVersion",
    "NewSourceParagraph",
    "PersistenceConflictError",
    "RecordNotFoundError",
    "RepositoryError",
    "UserRepository",
    "normalize_email",
]
