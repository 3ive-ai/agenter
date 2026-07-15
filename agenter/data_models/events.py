"""Typed event data classes for CodingEvent.

Event names follow DDD (Domain-Driven Design) conventions:
- Events represent facts that happened
- Named in past tense: SessionStarted, TaskCompleted
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SessionStarted(BaseModel):
    """Session has started with configuration."""

    cwd: str
    model: str
    max_iterations: int


class IterationStarted(BaseModel):
    """An iteration has started."""

    iteration: int


class IterationCompleted(BaseModel):
    """An iteration has completed."""

    iteration: int
    passed: bool
    files_modified: int
    tokens_used: int
    cost_usd: float
    elapsed_seconds: float


class MessageReceived(BaseModel):
    """A message was received from the backend."""

    message_type: str
    content: str
    tool_name: str | None = None


class ValidationStarted(BaseModel):
    """Validation has started."""

    validators: list[str]
    file_count: int


class ValidationCompleted(BaseModel):
    """A validator has completed."""

    validator: str
    passed: bool
    errors: list[str] = Field(default_factory=list)


class RequestRefused(BaseModel):
    """The request was refused by the model."""

    reason: str
    category: str | None = None


class TaskCompleted(BaseModel):
    """The coding task has completed successfully."""

    model_config = {"arbitrary_types_allowed": True}

    status: str
    files: dict[str, str]
    summary: str
    iterations: int
    total_tokens: int
    cost_usd: float
    duration_seconds: float
    usage_reported: bool | None = None
    output: Any | None = None


class TaskFailed(BaseModel):
    """The coding task has failed."""

    status: str
    files: dict[str, str]
    summary: str
    iterations: int
    total_tokens: int
    cost_usd: float
    duration_seconds: float
    usage_reported: bool | None = None
    limit_type: str | None = None
    limit_value: float | int | None = None
    actual_value: float | int | None = None
    refusal_reason: str | None = None
    refusal_category: str | None = None


class SessionEnded(BaseModel):
    """Session has ended."""

    status: str
    iterations: int
    total_tokens: int
    cost_usd: float
    duration_seconds: float
    usage_reported: bool | None = None


# Union type for type hints
EventData = (
    SessionStarted
    | IterationStarted
    | IterationCompleted
    | MessageReceived
    | ValidationStarted
    | ValidationCompleted
    | RequestRefused
    | TaskCompleted
    | TaskFailed
    | SessionEnded
)

__all__ = [
    "EventData",
    "IterationCompleted",
    "IterationStarted",
    "MessageReceived",
    "RequestRefused",
    "SessionEnded",
    "SessionStarted",
    "TaskCompleted",
    "TaskFailed",
    "ValidationCompleted",
    "ValidationStarted",
]
