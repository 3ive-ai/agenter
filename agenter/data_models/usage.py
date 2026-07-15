"""Usage and budget tracking types.

This module contains types for tracking token usage, cost, and resource consumption.
"""

from __future__ import annotations

from pydantic import BaseModel, model_validator


class Usage(BaseModel):
    """Token usage and cost for a backend execution.

    This is the structured return type from CodingBackend.usage().

    Args:
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens generated.
        cost_usd: Estimated cost in USD.
        model: Model identifier used (e.g., "claude-sonnet-4-20250514").
        provider: Provider/backend type ("anthropic-sdk", "bedrock", "claude-code").
        reported: Whether the backend actually supplied these usage values. A
            false value distinguishes unavailable usage from measured zero usage.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str | None = None
    provider: str | None = None
    reported: bool = True

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens


class UsageDelta(BaseModel):
    """Delta for budget tracking updates.

    Use this with BudgetMeter.add_usage() to record resource consumption.
    Negative values are rejected to prevent budget accounting errors.

    Example:
        tracker.add_usage(UsageDelta(tokens=500, cost_usd=0.25))

    Raises:
        ValueError: If tokens or cost_usd is negative.
    """

    tokens: int = 0
    cost_usd: float = 0.0

    @model_validator(mode="after")
    def validate_non_negative(self) -> UsageDelta:
        """Validate that delta values are non-negative."""
        if self.tokens < 0:
            raise ValueError(f"UsageDelta.tokens cannot be negative: {self.tokens}")
        if self.cost_usd < 0:
            raise ValueError(f"UsageDelta.cost_usd cannot be negative: {self.cost_usd}")
        return self
