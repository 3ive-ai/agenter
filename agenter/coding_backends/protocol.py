"""Protocol for coding backends.

This module defines the abstract interface that all coding backends must implement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic import BaseModel

    from ..data_models import BackendMessage, ModifiedFiles, RefusalMessage, TurnError, Usage


class CodingBackend(Protocol):
    """Abstract interface for coding backends.

    Implementations wrap specific coding agent SDKs (Claude, Codex, OpenHands, etc.)
    and normalize their APIs to this common interface.

    Example:
        class MyBackend:
            async def connect(
                self,
                cwd: str,
                allowed_write_paths: list[str] | None = None,
                resume_session_id: str | None = None,
                output_type: type[BaseModel] | None = None,
                system_prompt: str | None = None,
            ) -> None:
                self.cwd = cwd
                self._output_type = output_type
                self._system_prompt = system_prompt

            async def execute(self, prompt: str) -> AsyncIterator[BackendMessage]:
                # Async generator that yields backend messages
                yield TextMessage(type="text", content="Hello")

            def modified_files(self) -> ModifiedFiles:
                return ContentModifiedFiles()

            def usage(self) -> Usage:
                return Usage(input_tokens=0, output_tokens=0, cost_usd=0.0)

            def structured_output(self) -> BaseModel | None:
                return self._structured_output  # Set during execute if output_type

            def refusal(self) -> RefusalMessage | None:
                return self._refusal  # Set during execute if LLM refuses

            async def disconnect(self) -> None:
                pass
    """

    async def connect(
        self,
        cwd: str,
        allowed_write_paths: list[str] | None = None,
        resume_session_id: str | None = None,
        output_type: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the backend with a working directory.

        Args:
            cwd: Working directory for file operations.
            allowed_write_paths: Optional glob patterns restricting writes.
            resume_session_id: Optional session ID to resume a previous session.
                Only supported by some backends (e.g., ClaudeCodeBackend).
            output_type: Optional Pydantic model for structured output.
                When set, the backend should return typed output matching this schema.
            system_prompt: Custom system prompt. If None, uses the backend's default.
        """
        ...

    def execute(self, prompt: str) -> AsyncIterator[BackendMessage]:
        """Execute a prompt and stream messages.

        This should be an async generator that yields BackendMessage objects
        as the backend processes the request.

        Args:
            prompt: The user prompt to execute.

        Yields:
            BackendMessage objects for each significant step.
        """
        ...

    def modified_files(self) -> ModifiedFiles:
        """Return files modified during execution.

        Returns:
            ContentModifiedFiles (with content) or PathsModifiedFiles (paths only).
            Use isinstance() to determine type, or check paths_only property.
        """
        ...

    def usage(self) -> Usage:
        """Return cumulative token usage and cost across all executions.

        IMPORTANT: Usage MUST be cumulative across all execute() calls within
        a session (between connect() and disconnect()). The session computes
        deltas by subtracting previous values, so non-cumulative tracking will
        cause incorrect budget accounting.

        Returns:
            Usage object with cumulative input_tokens, output_tokens, and cost_usd
            since connect() was called.
        """
        ...

    def structured_output(self) -> BaseModel | None:
        """Return structured output if output_type was set.

        Returns:
            Pydantic model instance if output_type was configured and
            structured output was successfully parsed, None otherwise.
        """
        ...

    def refusal(self) -> RefusalMessage | None:
        """Return refusal if LLM explicitly refused the request.

        When the LLM cannot complete a request due to safety, policy, or
        capability limitations, it should call the Refusal tool. This method
        returns the captured refusal message.

        Returns:
            RefusalMessage with reason and category if LLM refused,
            None otherwise.
        """
        ...

    def turn_error(self) -> TurnError | None:
        """Return a non-task turn failure, if any (optional).

        Distinct from refusal(): this reports a transport/provider fault that
        prevented the turn from completing — the model stream disconnected
        mid-response, an RPC error occurred, the turn was cancelled, or the agent
        stopped early. The session maps a returned TurnError to
        CodingStatus.ERROR. Backends that cannot detect such faults may omit this
        method entirely; the session accesses it defensively via getattr.

        Returns:
            TurnError with reason/kind/retryable if the turn aborted for a
            non-task reason, None otherwise.
        """
        ...

    async def disconnect(self) -> None:
        """Clean up resources."""
        ...
