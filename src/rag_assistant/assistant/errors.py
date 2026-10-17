"""Expected workflow-boundary failures with safe public messages."""


class WorkflowError(RuntimeError):
    """Base class for expected assistant-workflow failures."""


class ConversationAuthorizationError(WorkflowError):
    """The caller could not establish access to the requested conversation."""


class WorkflowExecutionError(WorkflowError):
    """The graph ended before it could persist an assistant outcome."""
