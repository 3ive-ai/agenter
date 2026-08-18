"""Base class for coding backends.

Provides common state management and default implementations for methods
shared across all backend implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .refusal import RefusalDetector

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

    from ..data_models import TurnError


class BaseBackend(RefusalDetector):
    """Base class for all coding backends.

    Provides:
    - Token tracking state (`_input_tokens`, `_output_tokens`, `_cost_usd`)
    - Structured output state (`_output_type`, `_structured_output`)
    - Refusal handling (inherited from RefusalHandler)
    - Common method implementations

    Subclasses should:
    - Call `_init_state()` in `__init__` to initialize common state
    - Call `_reset_state()` in `connect()` and `disconnect()` to reset state
    - Override `usage()` if custom cost calculation is needed
    """

    # Type hints for common state (initialized in _init_state)
    _input_tokens: int
    _output_tokens: int
    _cost_usd: float
    _output_type: type[BaseModel] | None
    _structured_output: BaseModel | None

    def _init_state(self) -> None:
        """Initialize common state. Call this in __init__."""
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost_usd = 0.0
        self._output_type = None
        self._structured_output = None
        self._refusal = None

    def _reset_state(self) -> None:
        """Reset common state. Call this in connect() and disconnect()."""
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost_usd = 0.0
        self._output_type = None
        self._structured_output = None
        self._reset_refusal()

    def structured_output(self) -> BaseModel | None:
        """Return structured output if output_type was configured."""
        return self._structured_output

    # refusal() is inherited from RefusalDetector

    def turn_error(self) -> TurnError | None:
        """Return a non-task turn failure (transport/provider fault), if any.

        Default implementation: backends that cannot detect mid-stream transport
        faults report none. ACPBackend overrides this. See the CodingBackend
        protocol for the full contract.
        """
        return None

    def _build_system_prompt_from_template(
        self,
        cwd: Path,
        template: str,
        refusal_instructions: str,
        custom_prompt: str | None = None,
    ) -> str:
        """Build system prompt from template with refusal instructions.

        Args:
            cwd: Working directory to format into template.
            template: Prompt template with {cwd} placeholder.
            refusal_instructions: Refusal handling instructions to append.
            custom_prompt: Optional custom prompt to append at the end.

        Returns:
            Formatted system prompt.
        """
        base = template.format(cwd=cwd)
        base = f"{base}\n\n{refusal_instructions}"
        if custom_prompt:
            return f"{base}\n\n{custom_prompt}"
        return base
