"""Tests for the CodingSession class."""

import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agenter.coding_backends.anthropic_sdk.backend import AnthropicSDKBackend
from agenter.config import DEFAULT_MODEL_ANTHROPIC
from agenter.data_models import (
    BackendError,
    BackendMessage,
    CodingEventType,
    CodingRequest,
    CodingStatus,
    ContentModifiedFiles,
    PathsModifiedFiles,
    TurnError,
    Usage,
)
from agenter.post_validators.syntax import SyntaxValidator
from agenter.runtime.session import CodingSession


@pytest.fixture
def backend():
    """Create a default backend for testing."""
    return AnthropicSDKBackend(model=DEFAULT_MODEL_ANTHROPIC)


class TestCodingSessionPrepareFiles:
    """Test _prepare_files_for_validation method."""

    def test_prepare_files_not_paths_only(self, backend):
        session = CodingSession(backend, validators=[])

        files = ContentModifiedFiles(files={"a.py": "print('hello')", "b.py": "x = 1"})
        result = session._prepare_files_for_validation(files, "/tmp")

        assert result == {"a.py": "print('hello')", "b.py": "x = 1"}

    def test_prepare_files_paths_only(self, backend):
        session = CodingSession(backend, validators=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("print('hello')")
            (Path(tmpdir) / "b.py").write_text("x = 1")

            files = PathsModifiedFiles(file_paths=["a.py", "b.py"])
            result = session._prepare_files_for_validation(files, tmpdir)

            assert "a.py" in result
            assert "b.py" in result
            assert result["a.py"] == "print('hello')"
            assert result["b.py"] == "x = 1"

    def test_prepare_files_paths_only_missing_file(self, backend):
        session = CodingSession(backend, validators=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "exists.py").write_text("x = 1")

            files = PathsModifiedFiles(file_paths=["exists.py", "missing.py"])
            result = session._prepare_files_for_validation(files, tmpdir)

            assert "exists.py" in result
            assert "missing.py" not in result

    def test_prepare_files_paths_only_skips_python_bytecode_cache(self, backend):
        session = CodingSession(backend, validators=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "module.cpython-314.pyc").write_bytes(b"\x00\x01")
            (Path(tmpdir) / "module.py").write_text("x = 1")

            files = PathsModifiedFiles(file_paths=["module.py", "__pycache__/module.cpython-314.pyc"])
            result = session._prepare_files_for_validation(files, tmpdir)

            assert result == {"module.py": "x = 1"}

    def test_prepare_files_empty(self, backend):
        session = CodingSession(backend, validators=[])

        files = ContentModifiedFiles(files={})
        result = session._prepare_files_for_validation(files, "/tmp")

        assert result == {}

    def test_prepare_files_with_binary_extension(self, backend):
        session = CodingSession(backend, validators=[SyntaxValidator()])

        files = ContentModifiedFiles(files={"test.pyc": "binary content"})
        result = session._prepare_files_for_validation(files, "/tmp")

        assert "test.pyc" in result


class TestBackendExecuteRequiresConnect:
    """Test that execute fails after disconnect without reconnecting."""

    @pytest.mark.asyncio
    async def test_anthropic_backend(self):
        backend = AnthropicSDKBackend(model=DEFAULT_MODEL_ANTHROPIC)

        with tempfile.TemporaryDirectory() as tmpdir:
            await backend.connect(tmpdir)
            await backend.disconnect()

            with pytest.raises(BackendError, match="not connected"):
                async for _ in backend.execute("test"):
                    pass

    @pytest.mark.asyncio
    async def test_claude_code_backend(self):
        from agenter.coding_backends.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend(sandbox=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            await backend.connect(tmpdir)
            await backend.disconnect()

            # Match either "not connected" or SDK missing error
            with pytest.raises(BackendError, match=r"not connected|claude-agent-sdk is required"):
                async for _ in backend.execute("test"):
                    pass


class _TurnErrorBackend:
    """Minimal backend that reports a turn_error, to test session mapping."""

    model = "fake"

    def __init__(self, turn_error: TurnError) -> None:
        self._turn_error = turn_error

    async def connect(self, cwd, allowed_write_paths=None, **kwargs) -> None:
        self._cwd = cwd

    async def execute(self, prompt: str) -> AsyncIterator[BackendMessage]:
        # A turn that produced no content — exactly the shape of a stream disconnect.
        return
        yield  # pragma: no cover - makes this an async generator

    def modified_files(self) -> PathsModifiedFiles:
        return PathsModifiedFiles(file_paths=[])

    def usage(self) -> Usage:
        return Usage(input_tokens=0, output_tokens=0, cost_usd=0.0, provider="acp", reported=False)

    def structured_output(self):
        return None

    def refusal(self):
        return None

    def turn_error(self) -> TurnError | None:
        return self._turn_error

    async def disconnect(self) -> None:
        return None


class TestCodingSessionTurnError:
    """The session maps a backend turn_error to CodingStatus.ERROR."""

    @pytest.mark.asyncio
    async def test_turn_error_maps_to_error_status(self) -> None:
        turn_error = TurnError(
            reason="stream disconnected before completion: high demand",
            kind="provider_disconnect",
            retryable=True,
            stop_reason="end_turn",
        )
        backend = _TurnErrorBackend(turn_error)
        session = CodingSession(backend, validators=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            request = CodingRequest(prompt="do work", cwd=tmpdir)
            result = await session.run(request)

        assert result.status == CodingStatus.ERROR
        assert result.error_kind == "provider_disconnect"
        assert result.retryable is True
        assert "provider_disconnect" in result.summary

    @pytest.mark.asyncio
    async def test_turn_error_emits_error_event_and_no_completed_event(self) -> None:
        turn_error = TurnError(reason="rpc boom", kind="rpc_error", retryable=True)
        backend = _TurnErrorBackend(turn_error)
        session = CodingSession(backend, validators=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            request = CodingRequest(prompt="do work", cwd=tmpdir)
            event_types = [event.type async for event in session.stream_run(request)]

        assert CodingEventType.ERROR in event_types
        assert CodingEventType.COMPLETED not in event_types
        assert event_types[-1] == CodingEventType.SESSION_END
