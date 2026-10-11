from enum import StrEnum


class IngestionStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ParagraphKind(StrEnum):
    PROSE = "prose"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    TABLE_ROW = "table_row"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class MessageStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class AnswerOutcome(StrEnum):
    FULLY_ANSWERED = "fully_answered"
    PARTIALLY_ANSWERED = "partially_answered"
    ABSTAINED = "abstained"
