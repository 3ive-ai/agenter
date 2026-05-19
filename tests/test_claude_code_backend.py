"""Tests for the Claude Code backend."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from agenter import AutonomousCodingAgent
from agenter.coding_backends.claude_code import ClaudeCodeBackend

if TYPE_CHECKING:
    from pathlib import Path


class FakeClaudeAgentOptions:
    """Capture ClaudeAgentOptions kwargs without invoking the real SDK."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs) -> None:
        FakeClaudeAgentOptions.last_kwargs = kwargs


async def fake_query(prompt, options):
    if False:
        yield None


class TestClaudeCodeBackend:
    """Behavior tests for ClaudeCodeBackend."""

    @pytest.mark.asyncio
    async def test_execute_passes_model_and_thinking_budget_to_sdk_options(self, tmp_path: Path) -> None:
        backend = ClaudeCodeBackend(model="claude-sonnet-4-5-20250929", max_thinking_tokens=8192)
        await backend.connect(str(tmp_path))

        fake_module = SimpleNamespace(ClaudeAgentOptions=FakeClaudeAgentOptions, query=fake_query)
        with patch.dict("sys.modules", {"claude_agent_sdk": fake_module}):
            _ = [message async for message in backend.execute("test")]

        assert FakeClaudeAgentOptions.last_kwargs is not None
        assert FakeClaudeAgentOptions.last_kwargs["model"] == "claude-sonnet-4-5-20250929"
        assert FakeClaudeAgentOptions.last_kwargs["max_thinking_tokens"] == 8192

    def test_agent_facade_passes_model_and_thinking_budget(self) -> None:
        agent = AutonomousCodingAgent(
            backend="claude-code",
            model="claude-sonnet-4-5-20250929",
            claude_max_thinking_tokens=8192,
        )

        backend = agent._create_backend()

        assert isinstance(backend, ClaudeCodeBackend)
        assert backend._configured_model == "claude-sonnet-4-5-20250929"
        assert backend._max_thinking_tokens == 8192
