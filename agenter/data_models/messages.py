"""Backend message types.

This module contains the message types exchanged between the backend and session.
Uses Pydantic for validation of message structure.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from .tools import ToolResult


class PromptMessage(BaseModel):
    """Prompt being sent to LLM."""

    type: Literal["prompt"] = "prompt"
    user_prompt: str
    system_prompt: str | None = None


class TextMessage(BaseModel):
    """Text response from LLM."""

    type: Literal["text"] = "text"
    content: str
    tokens: int = 0


class ToolCallMessage(BaseModel):
    """Tool being called."""

    type: Literal["tool_call"] = "tool_call"
    tool_name: str
    args: dict[str, Any]


class RefusalMessage(BaseModel):
    """Message indicating LLM refused the request.

    Used when the LLM cannot complete a request due to safety,
    policy, or capability limitations.
    """

    type: Literal["refusal"] = "refusal"
    reason: str
    category: Literal["safety", "policy", "capability"] | None = None


class TurnError(BaseModel):
    """Side-channel signal that a turn aborted for a non-task reason.

    Unlike RefusalMessage (the model chose not to act) this indicates the
    turn did not complete because of a transport/provider fault: the model
    stream disconnected mid-response, an ACP/RPC error occurred, the turn was
    cancelled, or the agent stopped early. Backends expose the most recent one
    via ``turn_error()`` (mirroring the ``refusal()`` channel); the session maps
    it to ``CodingStatus.ERROR``.
    """

    type: Literal["turn_error"] = "turn_error"
    reason: str
    kind: Literal[
        "provider_disconnect",
        "provider_capacity",
        "rpc_error",
        "cancelled",
        "max_tokens",
        "unknown",
    ] = "unknown"
    retryable: bool = False
    stop_reason: str | None = None


# Union type for all backend messages
# ToolResult is imported from tools.py and serves as the tool result message
BackendMessage = PromptMessage | TextMessage | ToolCallMessage | ToolResult | RefusalMessage
