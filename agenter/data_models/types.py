"""Core types for Agenter.

This module contains the primary request/response types, status enums,
and event types used throughout the SDK.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path  # noqa: TC003 - required at runtime by Pydantic

import structlog
from pydantic import BaseModel, Field

from ..config import DEFAULT_MAX_ITERATIONS
from .events import EventData  # noqa: TC001 - required at runtime by Pydantic
from .messages import BackendMessage  # noqa: TC001 - required at runtime by Pydantic

logger = structlog.get_logger(__name__)


class CodingStatus(str, Enum):
    """Status of a coding session.

    Status semantics:
        COMPLETED: Task succeeded within all budget constraints.
        COMPLETED_WITH_LIMIT_EXCEEDED: Task succeeded (validation passed) but
            exceeded one or more budget limits. The work is valid but consumed
            more resources than configured.
        BUDGET_EXCEEDED: Task was stopped because budget limits were reached
            before the task could complete successfully.
        REFUSED: LLM explicitly refused the request due to safety, policy,
            or capability limitations. Use result.refusal_reason for details.
        FAILED: True failure (exception, unrecoverable error, not budget-related).
    """

    COMPLETED = "completed"
    COMPLETED_WITH_LIMIT_EXCEEDED = "completed_with_limit_exceeded"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    REFUSED = "refused"


class Verbosity(str, Enum):
    """Verbosity level for console output."""

    QUIET = "quiet"  # Errors only
    NORMAL = "normal"  # Progress indicators (default)
    VERBOSE = "verbose"  # Full details


class CodingEventType(str, Enum):
    """Types of events emitted during coding session.

    Event lifecycle:
        SESSION_START → ITERATION_START → BACKEND_MESSAGE* →
        VALIDATION_START → VALIDATION_RESULT* → ITERATION_END →
        (repeat iterations) → COMPLETED/FAILED/REFUSED → SESSION_END
    """

    SESSION_START = "session_start"  # Session initialized with config
    ITERATION_START = "iteration_start"  # Beginning iteration N
    BACKEND_MESSAGE = "backend_message"  # Message from backend (text, tool_use, etc.)
    VALIDATION_START = "validation_start"  # About to run validators
    VALIDATION_RESULT = "validation_result"  # Result from a single validator
    ITERATION_END = "iteration_end"  # Iteration complete with metrics
    SESSION_END = "session_end"  # Session finished (always emitted)
    COMPLETED = "completed"  # Task completed successfully
    FAILED = "failed"  # Task failed (budget exceeded, max iterations, etc.)
    REFUSED = "refused"  # LLM explicitly refused the request


class ContentModifiedFiles(BaseModel):
    """Files modified with full content available.

    Used by backends that track file contents (e.g., AnthropicSDKBackend).

    Example:
        changes = ContentModifiedFiles(files={"main.py": "print('hi')"})
        text = changes.content("main.py")  # Returns "print('hi')"
    """

    files: dict[str, str] = Field(default_factory=dict)

    @property
    def paths_only(self) -> bool:
        return False

    def __len__(self) -> int:
        return len(self.files)

    def __contains__(self, path: object) -> bool:
        return path in self.files

    def content(self, path: str) -> str | None:
        return self.files.get(path)

    def paths(self) -> list[str]:
        return list(self.files.keys())


class PathsModifiedFiles(BaseModel):
    """Files modified with only paths tracked (no content).

    Used by backends that don't track content (e.g., ClaudeCodeBackend).
    Content must be read from disk if needed.

    Example:
        changes = PathsModifiedFiles(file_paths=["main.py", "utils.py"])
        changes.content("main.py")  # Returns None (must read from disk)
    """

    file_paths: list[str] = Field(default_factory=list)

    @property
    def paths_only(self) -> bool:
        return True

    @property
    def files(self) -> dict[str, str]:
        """Empty content dict for compatibility with ContentModifiedFiles interface."""
        return dict.fromkeys(self.file_paths, "")

    def __len__(self) -> int:
        return len(self.file_paths)

    def __contains__(self, path: object) -> bool:
        return path in self.file_paths

    def content(self, path: str) -> str | None:
        """Always returns None - content must be read from disk."""
        logger.debug("content_paths_only_mode", path=path)
        return None

    def paths(self) -> list[str]:
        return list(self.file_paths)


# Union type for backend return values
ModifiedFiles = ContentModifiedFiles | PathsModifiedFiles


class Budget(BaseModel):
    """Budget limits for a coding session."""

    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_time_seconds: float | None = None
    max_iterations: int = DEFAULT_MAX_ITERATIONS


class CodingRequest(BaseModel):
    """Request for a coding task.

    Attributes:
        prompt: The task prompt for the agent.
        cwd: Working directory for file operations.
        system_prompt: Custom system prompt for the LLM. If None, uses the
            backend's default system prompt. Use this to provide domain-specific
            instructions (e.g., APE's RED TEAM OPERATOR context).
        max_iterations: Max validation/refinement iterations (default: 5).
        budget: Optional full budget control (tokens, cost, time).
        allowed_write_paths: Glob patterns restricting file writes.
        output_type: Optional Pydantic model for typed output. When set,
            the agent returns structured output matching this schema.
    """

    model_config = {"arbitrary_types_allowed": True}

    prompt: str
    cwd: str
    system_prompt: str | None = None  # Custom system prompt (overrides default)
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    budget: Budget | None = None  # Optional full budget control
    allowed_write_paths: list[str] | None = None  # Glob patterns restricting writes
    output_type: type[BaseModel] | None = None  # Typed structured output


class CodingResult(BaseModel):
    """Result of a coding session.

    Attributes:
        status: Final status of the coding session.
        files: Mapping of file path to content for modified files.
        summary: Human-readable summary of what was done.
        iterations: Number of agent iterations executed.
        total_tokens: Total tokens consumed (input + output).
        total_cost_usd: Estimated total cost in USD.
        total_duration_seconds: Wall clock time for the session.
        exceeded_limit: Which budget limit was exceeded (if status is BUDGET_EXCEEDED).
            One of: "iterations", "tokens", "cost", "time", or None.
        exceeded_values: Details about the exceeded limit (if status is BUDGET_EXCEEDED).
            Contains "limit_value" (configured max) and "actual_value" (value that exceeded).
        output: Typed structured output when output_type was specified in request.
            Contains the Pydantic model instance with all fields including code.
    """

    model_config = {"arbitrary_types_allowed": True}

    status: CodingStatus
    files: dict[str, str]  # path -> content
    summary: str
    iterations: int
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_seconds: float = 0.0
    usage_reported: bool | None = None  # False when the backend did not emit usage
    exceeded_limit: str | None = None  # Which limit was exceeded (max_iterations, max_tokens, etc.)
    exceeded_values: dict[str, float | int] | None = None  # {limit_value: X, actual_value: Y}
    output: BaseModel | None = None  # Typed structured output
    trace_dir: Path | None = None  # Directory where traces were saved (if tracer was used)
    session_id: str | None = None  # Backend session shared by persistent follow-ups
    request_index: int | None = None  # 1-based request number within a persistent session
    session_total_tokens: int | None = None  # Cumulative usage across persistent requests
    session_total_cost_usd: float | None = None
    session_total_duration_seconds: float | None = None

    def _repr_markdown_(self) -> str:
        """Jupyter notebook rich display."""
        parts = [f"**Status:** {self.status.value}", f"**Summary:** {self.summary}"]
        for path, content in self.files.items():
            ext = path.rsplit(".", 1)[-1] if "." in path else ""
            parts.append(f"\n**{path}**\n```{ext}\n{content}\n```")
        return "\n\n".join(parts)


class CodingEvent(BaseModel):
    """Event emitted during coding session for observability."""

    model_config = {"arbitrary_types_allowed": True}

    type: CodingEventType
    data: EventData | None = None
    message: BackendMessage | None = None  # Typed message for BACKEND_MESSAGE events
    result: CodingResult | None = None  # Final result for terminal events (COMPLETED, FAILED)
    session_id: str | None = None  # Backend session shared by persistent follow-ups
    request_index: int | None = None  # 1-based request number within a persistent session
    timestamp: float = Field(default_factory=time.time)


class ValidationResult(BaseModel):
    """Result of running validators on modified files."""

    passed: bool
    errors: list[str] = Field(default_factory=list)
