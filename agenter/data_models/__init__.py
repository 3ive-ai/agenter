"""Data models and exceptions for Agenter.

This module provides the curated public API surface for all data models.
Types are organized into focused submodules but re-exported here for convenience.

Submodules:
    - types: Request/response types, status enums, events
    - events: Typed event data classes
    - tools: Tool execution result types and error codes
    - messages: Backend message types (Pydantic models)
    - usage: Token usage and cost tracking
    - exceptions: SDK exception hierarchy
"""

from .budget import BudgetLimitType
from .events import (
    EventData,
    IterationCompleted,
    IterationStarted,
    MessageReceived,
    RequestRefused,
    SessionEnded,
    SessionStarted,
    TaskCompleted,
    TaskErrored,
    TaskFailed,
    ValidationCompleted,
    ValidationStarted,
)
from .exceptions import (
    AgenterError,
    BackendError,
    BudgetExceededError,
    ConfigurationError,
    PathSecurityError,
    ToolExecutionError,
    ValidationError,
)
from .messages import (
    BackendMessage,
    PromptMessage,
    RefusalMessage,
    TextMessage,
    ToolCallMessage,
    TurnError,
)
from .tools import (
    ToolError,
    ToolErrorCode,
    ToolResult,
)
from .types import (
    Budget,
    CodingEvent,
    CodingEventType,
    CodingRequest,
    CodingResult,
    CodingStatus,
    ContentModifiedFiles,
    ModifiedFiles,
    PathsModifiedFiles,
    ValidationResult,
    Verbosity,
)
from .usage import (
    Usage,
    UsageDelta,
)

__all__ = [
    # Public exports
    "AgenterError",
    "BackendError",
    "BackendMessage",
    "Budget",
    "BudgetExceededError",
    "BudgetLimitType",
    "CodingEvent",
    "CodingEventType",
    "CodingRequest",
    "CodingResult",
    "CodingStatus",
    "ConfigurationError",
    "ContentModifiedFiles",
    "EventData",
    "IterationCompleted",
    "IterationStarted",
    "MessageReceived",
    "ModifiedFiles",
    "PathSecurityError",
    "PathsModifiedFiles",
    "PromptMessage",
    "RefusalMessage",
    "RequestRefused",
    "SessionEnded",
    "SessionStarted",
    "TaskCompleted",
    "TaskErrored",
    "TaskFailed",
    "TextMessage",
    "ToolCallMessage",
    "ToolError",
    "ToolErrorCode",
    "ToolExecutionError",
    "ToolResult",
    "TurnError",
    "Usage",
    "UsageDelta",
    "ValidationCompleted",
    "ValidationError",
    "ValidationResult",
    "ValidationStarted",
    "Verbosity",
]
