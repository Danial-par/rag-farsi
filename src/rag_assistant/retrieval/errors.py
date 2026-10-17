class RetrievalError(RuntimeError):
    """Base class for retrieval failures safe to classify at application boundaries."""


class InvalidRetrievalQueryError(RetrievalError):
    """The query has no searchable content after deterministic normalization."""


class RetrievalScopeError(RetrievalError):
    """The owned, non-deleted conversation scope could not be established."""


class IncompatibleEmbeddingConfigurationError(RetrievalError):
    """A selected active document is incompatible with the query embedding."""


class RetrievalIntegrityError(RetrievalError):
    """Retrieved evidence does not resolve to complete authoritative provenance."""
