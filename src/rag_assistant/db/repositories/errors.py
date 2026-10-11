from __future__ import annotations


class RepositoryError(RuntimeError):
    """Base class for expected persistence-boundary failures."""


class RecordNotFoundError(RepositoryError):
    """An aggregate required by a command was not visible in its ownership scope."""


class PersistenceConflictError(RepositoryError):
    """A uniqueness or concurrent-state conflict prevented persistence."""


class InvalidPersistenceStateError(RepositoryError):
    """An aggregate cannot perform the requested lifecycle transition."""
