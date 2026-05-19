"""Tests for the Codex backend."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from agenter import AutonomousCodingAgent
from agenter.coding_backends.codex import CodexBackend
from agenter.data_models import BackendError, ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path


class TestCodexBackend:
    """Behavior tests for CodexBackend."""

    def test_rejects_invalid_approval_policy(self) -> None:
        with pytest.raises(ConfigurationError, match="approval_policy"):
            CodexBackend(approval_policy="invalid")

    def test_rejects_invalid_sandbox(self) -> None:
        with pytest.raises(ConfigurationError, match="sandbox"):
            CodexBackend(sandbox="invalid")

    @pytest.mark.anyio
    async def test_execute_requires_connect(self) -> None:
        backend = CodexBackend()
        with pytest.raises(BackendError, match="not connected"):
            async for _ in backend.execute("test"):
                pass

    @pytest.mark.anyio
    async def test_connect_requires_openai_agents(self) -> None:
        backend = CodexBackend()
        with (
            patch.dict("sys.modules", {"agents": None, "agents.mcp": None}),
            pytest.raises(BackendError, match="openai-agents"),
        ):
            await backend.connect("/tmp")

    @pytest.mark.anyio
    async def test_disconnect_handles_cleanup_error_gracefully(self) -> None:
        backend = CodexBackend()
        mock_server = AsyncMock()
        mock_server.cleanup = AsyncMock(side_effect=Exception("boom"))
        backend._mcp_server = mock_server

        await backend.disconnect()  # Should not raise
        assert backend._mcp_server is None

    @pytest.mark.anyio
    async def test_disconnect_resets_public_state(self) -> None:
        backend = CodexBackend()
        await backend.disconnect()

        assert backend.usage().input_tokens == 0
        assert backend.modified_files().file_paths == []
        assert backend.structured_output() is None

    @pytest.mark.anyio
    async def test_execute_passes_reasoning_effort_to_codex_config(self, tmp_path: Path) -> None:
        backend = CodexBackend(model="gpt-5.4", reasoning_effort="high")
        mock_server = AsyncMock()
        mock_server.call_tool = AsyncMock(return_value={"content": [{"type": "text", "text": "done"}]})
        backend._mcp_server = mock_server
        backend._cwd = tmp_path

        messages = [message async for message in backend.execute("test")]

        assert messages[-1].content == "done"
        _, args = mock_server.call_tool.await_args.args
        assert args["model"] == "gpt-5.4"
        assert args["config"]["model_reasoning_effort"] == "high"

    def test_agent_facade_passes_codex_reasoning_effort(self) -> None:
        agent = AutonomousCodingAgent(backend="codex", model="gpt-5.4", codex_reasoning_effort="high")

        backend = agent._create_backend()

        assert isinstance(backend, CodexBackend)
        assert backend._model == "gpt-5.4"
        assert backend._reasoning_effort == "high"
