"""Tests for the ACP backend."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from agenter import AutonomousCodingAgent
from agenter.data_models import BackendError, ConfigurationError, PathsModifiedFiles, TextMessage, ToolCallMessage
from agenter.data_models.tools import ToolResult


def _prompt_text(prompt: list[object]) -> str:
    block = prompt[0]
    if isinstance(block, dict):
        return str(block["text"])
    if isinstance(block, BaseModel):
        return str(block.model_dump()["text"])
    raise AssertionError(f"Unexpected prompt block: {block!r}")


class TestACPBackendFacade:
    """Facade behavior for backend='acp'."""

    def test_acp_backend_is_exported_from_backend_package(self) -> None:
        from agenter.coding_backends import ACPBackend

        assert ACPBackend.__name__ == "ACPBackend"

    def test_acp_backend_requires_command(self) -> None:
        with pytest.raises(ConfigurationError, match="acp_command"):
            AutonomousCodingAgent(backend="acp")

    def test_create_backend_returns_acp_backend(self) -> None:
        from agenter.coding_backends.acp import ACPBackend

        agent = AutonomousCodingAgent(backend="acp", acp_command="fake-acp-agent")

        assert isinstance(agent._create_backend(), ACPBackend)

    def test_agent_facade_passes_acp_permission_policy(self) -> None:
        from agenter.coding_backends.acp import ACPBackend

        agent = AutonomousCodingAgent(
            backend="acp",
            acp_command="fake-acp-agent",
            acp_permission_policy="allow",
        )

        backend = agent._create_backend()

        assert isinstance(backend, ACPBackend)
        assert backend.permission_policy == "allow"

    def test_agent_facade_passes_acp_autonomous(self) -> None:
        from agenter.coding_backends.acp import ACPBackend

        agent = AutonomousCodingAgent(
            backend="acp",
            acp_command="fake-acp-agent",
            acp_autonomous=False,
        )

        backend = agent._create_backend()

        assert isinstance(backend, ACPBackend)
        assert backend.autonomous is False

    def test_agent_facade_passes_live_acp_update_callback(self) -> None:
        from agenter.coding_backends.acp import ACPBackend

        def callback(_update):
            return None

        agent = AutonomousCodingAgent(
            backend="acp",
            acp_command="fake-acp-agent",
            acp_update_callback=callback,
        )

        backend = agent._create_backend()

        assert isinstance(backend, ACPBackend)
        assert backend.update_callback is callback


class FakeACPProcessContext:
    """Async context manager returned by fake spawn_agent_process."""

    def __init__(self, connection: SimpleNamespace) -> None:
        self.connection = connection
        self.process = SimpleNamespace(pid=123)
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> tuple[SimpleNamespace, SimpleNamespace]:
        self.entered = True
        return self.connection, self.process

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.exited = True


class TestACPBackendLifecycle:
    """Lifecycle behavior for ACPBackend."""

    @pytest.mark.asyncio
    async def test_connect_initializes_and_creates_session(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
        )
        context = FakeACPProcessContext(connection)
        spawned = {}

        def fake_spawn_agent_process(client, command, *args, **kwargs):
            spawned["client"] = client
            spawned["command"] = command
            spawned["args"] = args
            spawned["kwargs"] = kwargs
            return context

        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", fake_spawn_agent_process, raising=False)

        backend = ACPBackend(command="fake-acp-agent", args=["--stdio"], env={"TOKEN": "x"})
        await backend.connect(str(tmp_path))

        assert context.entered is True
        assert spawned["command"] == "fake-acp-agent"
        assert spawned["args"] == ("--stdio",)
        connection.initialize.assert_awaited_once_with(protocol_version=1)
        connection.new_session.assert_awaited_once_with(cwd=str(tmp_path.resolve()), mcp_servers=[])
        assert backend._session_id == "session-1"

    @pytest.mark.asyncio
    async def test_connect_accepts_frames_larger_than_the_default_stream_limit(
        self, tmp_path
    ) -> None:
        from agenter.coding_backends.acp import ACPBackend

        default_stream_limit = asyncio.StreamReader()._limit
        agent_script = tmp_path / "large_frame_acp_agent.py"
        agent_script.write_text(
            f"""
import json
import sys

padding = "x" * {default_stream_limit + 1}
for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "initialize":
        result = {{
            "protocolVersion": 1,
            "agentCapabilities": {{}},
            "agentInfo": {{"name": "large-frame-agent", "version": "1"}},
            "_padding": padding,
        }}
    elif method == "session/new":
        result = {{"sessionId": "large-frame-session"}}
    else:
        result = {{}}
    print(
        json.dumps({{"jsonrpc": "2.0", "id": request["id"], "result": result}}),
        flush=True,
    )
""",
            encoding="utf-8",
        )
        backend = ACPBackend(command=sys.executable, args=[str(agent_script)])

        try:
            await backend.connect(str(tmp_path))
            assert backend.session_id == "large-frame-session"
        finally:
            await backend.disconnect()

    @pytest.mark.asyncio
    async def test_execute_requires_connect(self) -> None:
        from agenter.coding_backends.acp import ACPBackend

        backend = ACPBackend(command="fake-acp-agent")

        with pytest.raises(BackendError, match="not connected"):
            async for _ in backend.execute("hello"):
                pass

    @pytest.mark.asyncio
    async def test_update_callback_runs_before_prompt_finishes(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        observed = []

        async def prompt_side_effect(session_id, prompt):
            update = {"sessionUpdate": "tool_call", "toolCallId": "call-1", "title": "Inspect files"}
            await spawned["client"].session_update(session_id, update)
            assert observed == [update]
            return SimpleNamespace(stop_reason="end_turn")

        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
            prompt=AsyncMock(side_effect=prompt_side_effect),
        )
        context = FakeACPProcessContext(connection)
        spawned = {}

        def fake_spawn(client, *args, **kwargs):
            spawned["client"] = client
            return context

        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", fake_spawn, raising=False)
        backend = ACPBackend(command="fake-acp-agent", update_callback=observed.append)
        await backend.connect(str(tmp_path))

        _ = [message async for message in backend.execute("inspect")]

        assert observed[0]["toolCallId"] == "call-1"

    @pytest.mark.asyncio
    async def test_disconnect_exits_process_context(self, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
        )
        context = FakeACPProcessContext(connection)
        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", lambda *args, **kwargs: context, raising=False)

        backend = ACPBackend(command="fake-acp-agent")
        await backend.connect("/tmp")
        await backend.disconnect()

        assert context.exited is True
        assert backend._connection is None
        assert backend._session_id is None


class TestACPBackendExecution:
    """Prompt execution, update mapping, and file tracking."""

    @pytest.mark.asyncio
    async def test_execute_maps_text_updates(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        spawned = {}

        async def prompt_side_effect(session_id, prompt):
            await spawned["client"].session_update(
                session_id,
                {"type": "agent_message_chunk", "content": {"type": "text", "text": "hello from acp"}},
            )
            return SimpleNamespace(stop_reason="end_turn")

        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
            prompt=AsyncMock(side_effect=prompt_side_effect),
        )
        context = FakeACPProcessContext(connection)

        def fake_spawn_agent_process(client, command, *args, **kwargs):
            spawned["client"] = client
            return context

        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", fake_spawn_agent_process, raising=False)

        backend = ACPBackend(command="fake-acp-agent")
        await backend.connect(str(tmp_path))
        messages = [message async for message in backend.execute("hello")]

        assert messages == [TextMessage(content="hello from acp")]
        connection.prompt.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_coalesces_streaming_text_chunks(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        spawned = {}

        async def prompt_side_effect(session_id, prompt):
            for text in ("Using", " `", "core", "-engineering", "`", " now."):
                await spawned["client"].session_update(
                    session_id,
                    {"type": "agent_message_chunk", "content": {"type": "text", "text": text}},
                )
            return SimpleNamespace(stop_reason="end_turn")

        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
            prompt=AsyncMock(side_effect=prompt_side_effect),
        )
        context = FakeACPProcessContext(connection)

        def fake_spawn_agent_process(client, command, *args, **kwargs):
            spawned["client"] = client
            return context

        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", fake_spawn_agent_process, raising=False)

        backend = ACPBackend(command="fake-acp-agent")
        await backend.connect(str(tmp_path))
        messages = [message async for message in backend.execute("hello")]

        assert messages == [TextMessage(content="Using `core-engineering` now.")]

    @pytest.mark.asyncio
    async def test_execute_maps_plan_and_tool_updates(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        spawned = {}

        async def prompt_side_effect(session_id, prompt):
            await spawned["client"].session_update(
                session_id,
                {
                    "sessionUpdate": "plan",
                    "entries": [
                        {"content": "Inspect workspace", "status": "completed", "priority": "medium"},
                        {"content": "Write math_tools.py", "status": "in_progress", "priority": "high"},
                    ],
                },
            )
            await spawned["client"].session_update(
                session_id,
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tool-1",
                    "title": "edit",
                    "kind": "edit",
                    "status": "pending",
                    "content": [
                        {
                            "type": "diff",
                            "path": "math_tools.py",
                            "oldText": "",
                            "newText": "def add(a, b):\n    return a + b\n",
                        }
                    ],
                },
            )
            await spawned["client"].session_update(
                session_id,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tool-1",
                    "title": "edit",
                    "kind": "edit",
                    "status": "completed",
                    "content": [
                        {
                            "type": "diff",
                            "path": "math_tools.py",
                            "oldText": "",
                            "newText": "def add(a, b):\n    return a + b\n",
                        }
                    ],
                },
            )
            return SimpleNamespace(stop_reason="end_turn")

        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
            prompt=AsyncMock(side_effect=prompt_side_effect),
        )
        context = FakeACPProcessContext(connection)

        def fake_spawn_agent_process(client, command, *args, **kwargs):
            spawned["client"] = client
            return context

        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", fake_spawn_agent_process, raising=False)

        backend = ACPBackend(command="fake-acp-agent")
        await backend.connect(str(tmp_path))
        messages = [message async for message in backend.execute("create math tools")]

        assert messages == [
            TextMessage(content=("Plan:\n- [completed] Inspect workspace\n- [in_progress] Write math_tools.py")),
            ToolCallMessage(
                tool_name="edit",
                args={
                    "id": "tool-1",
                    "kind": "edit",
                    "status": "pending",
                    "path": "math_tools.py",
                },
            ),
            ToolResult(
                tool_name="edit",
                output="completed: math_tools.py",
                success=True,
            ),
        ]

    @pytest.mark.asyncio
    async def test_execute_tracks_usage_updates(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        spawned = {}

        async def prompt_side_effect(session_id, prompt):
            await spawned["client"].session_update(
                session_id,
                {
                    "sessionUpdate": "usage_update",
                    "used": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18},
                    "size": {"inputTokens": 100, "outputTokens": 100, "totalTokens": 200},
                    "cost": 0.002,
                },
            )
            return SimpleNamespace(stop_reason="end_turn")

        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
            prompt=AsyncMock(side_effect=prompt_side_effect),
        )
        context = FakeACPProcessContext(connection)

        def fake_spawn_agent_process(client, command, *args, **kwargs):
            spawned["client"] = client
            return context

        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", fake_spawn_agent_process, raising=False)

        backend = ACPBackend(command="fake-acp-agent")
        await backend.connect(str(tmp_path))
        messages = [message async for message in backend.execute("hello")]

        assert messages == []
        assert backend.usage().input_tokens == 11
        assert backend.usage().output_tokens == 7
        assert backend.usage().cost_usd == 0.002
        assert backend.usage().reported is True

    @pytest.mark.asyncio
    async def test_execute_adds_autonomous_contract_to_prompt_by_default(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        captured = {}

        async def prompt_side_effect(session_id, prompt):
            captured["prompt"] = prompt
            return SimpleNamespace(stop_reason="end_turn")

        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
            prompt=AsyncMock(side_effect=prompt_side_effect),
        )
        context = FakeACPProcessContext(connection)
        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", lambda *args, **kwargs: context, raising=False)

        backend = ACPBackend(command="fake-acp-agent")
        await backend.connect(str(tmp_path))
        _ = [message async for message in backend.execute("create a file")]

        sent_text = _prompt_text(captured["prompt"])

        assert "You are being called as a backend by Agenter." in sent_text
        assert "Do not stop to ask for confirmation." in sent_text
        assert "User task:\ncreate a file" in sent_text

    @pytest.mark.asyncio
    async def test_execute_can_disable_autonomous_contract(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        captured = {}

        async def prompt_side_effect(session_id, prompt):
            captured["prompt"] = prompt
            return SimpleNamespace(stop_reason="end_turn")

        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
            prompt=AsyncMock(side_effect=prompt_side_effect),
        )
        context = FakeACPProcessContext(connection)
        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", lambda *args, **kwargs: context, raising=False)

        backend = ACPBackend(command="fake-acp-agent", autonomous=False)
        await backend.connect(str(tmp_path))
        _ = [message async for message in backend.execute("create a file")]

        assert _prompt_text(captured["prompt"]) == "create a file"

    @pytest.mark.asyncio
    async def test_execute_auto_continues_once_when_agent_waits_for_confirmation(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        spawned = {}

        async def prompt_side_effect(session_id, prompt):
            if connection.prompt.await_count == 1:
                await spawned["client"].session_update(
                    session_id,
                    {
                        "type": "agent_message_chunk",
                        "content": {"type": "text", "text": "Reply yes and I will write it."},
                    },
                )
                return SimpleNamespace(stop_reason="end_turn")
            (tmp_path / "created.py").write_text("print('done')\n")
            await spawned["client"].session_update(
                session_id,
                {"type": "agent_message_chunk", "content": {"type": "text", "text": "Created created.py"}},
            )
            return SimpleNamespace(stop_reason="end_turn")

        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
            prompt=AsyncMock(side_effect=prompt_side_effect),
        )
        context = FakeACPProcessContext(connection)

        def fake_spawn_agent_process(client, command, *args, **kwargs):
            spawned["client"] = client
            return context

        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", fake_spawn_agent_process, raising=False)

        backend = ACPBackend(command="fake-acp-agent")
        await backend.connect(str(tmp_path))
        messages = [message async for message in backend.execute("create a file")]

        assert connection.prompt.await_count == 2
        assert "Proceed now" in _prompt_text(connection.prompt.await_args_list[1].kwargs["prompt"])
        assert messages == [
            TextMessage(content="Reply yes and I will write it."),
            TextMessage(content="Created created.py"),
        ]
        assert backend.modified_files().file_paths == ["created.py"]

    @pytest.mark.asyncio
    async def test_execute_refuses_if_auto_continue_still_waits_for_confirmation(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        spawned = {}

        async def prompt_side_effect(session_id, prompt):
            await spawned["client"].session_update(
                session_id,
                {"type": "agent_message_chunk", "content": {"type": "text", "text": "Please confirm before I write."}},
            )
            return SimpleNamespace(stop_reason="end_turn")

        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
            prompt=AsyncMock(side_effect=prompt_side_effect),
        )
        context = FakeACPProcessContext(connection)

        def fake_spawn_agent_process(client, command, *args, **kwargs):
            spawned["client"] = client
            return context

        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", fake_spawn_agent_process, raising=False)

        backend = ACPBackend(command="fake-acp-agent")
        await backend.connect(str(tmp_path))
        _ = [message async for message in backend.execute("create a file")]

        assert connection.prompt.await_count == 2
        assert backend.modified_files().file_paths == []
        assert backend.refusal() is not None
        assert backend.refusal().reason == "ACP agent asked for confirmation instead of modifying files."

    def test_confirmation_detection_matches_common_interactive_pause(self) -> None:
        from agenter.coding_backends.acp import ACPBackend

        backend = ACPBackend(command="fake-acp-agent")

        assert backend._looks_like_confirmation_request(
            [
                {
                    "type": "agent_message_chunk",
                    "content": {
                        "type": "text",
                        "text": "If that demo shape is fine, reply with `yes` and I'll write it.",
                    },
                }
            ]
        )

    @pytest.mark.asyncio
    async def test_permission_requests_are_cancelled_by_default(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        spawned = {}
        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
        )
        context = FakeACPProcessContext(connection)

        def fake_spawn_agent_process(client, command, *args, **kwargs):
            spawned["client"] = client
            return context

        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", fake_spawn_agent_process, raising=False)

        backend = ACPBackend(command="fake-acp-agent")
        await backend.connect(str(tmp_path))

        response = await spawned["client"].request_permission([], "session-1", {"name": "write_file"})

        assert response == {"outcome": {"outcome": "cancelled"}}

    @pytest.mark.asyncio
    async def test_permission_requests_can_select_allow_option(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        spawned = {}
        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
        )
        context = FakeACPProcessContext(connection)

        def fake_spawn_agent_process(client, command, *args, **kwargs):
            spawned["client"] = client
            return context

        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", fake_spawn_agent_process, raising=False)

        backend = ACPBackend(command="fake-acp-agent", permission_policy="allow")
        await backend.connect(str(tmp_path))

        response = await spawned["client"].request_permission(
            [
                {"kind": "reject_once", "optionId": "reject-1"},
                {"kind": "allow_once", "optionId": "allow-1"},
            ],
            "session-1",
            {"name": "write_file"},
        )

        assert response == {"outcome": {"outcome": "selected", "optionId": "allow-1"}}

    @pytest.mark.asyncio
    async def test_client_file_methods_read_and_write_workspace_files(self, tmp_path) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        backend = ACPBackend(command="fake-acp-agent")
        backend._cwd = tmp_path.resolve()
        client = acp_backend_module._AgenterACPClient(backend)

        write_response = await client.write_text_file(
            "def add(a, b):\n    return a + b\n",
            "pkg/math_tools.py",
            "session-1",
        )
        read_response = await client.read_text_file("pkg/math_tools.py", "session-1")

        assert (tmp_path / "pkg" / "math_tools.py").read_text() == "def add(a, b):\n    return a + b\n"
        assert write_response is not None
        assert read_response.content == "def add(a, b):\n    return a + b\n"

    @pytest.mark.asyncio
    async def test_client_file_methods_reject_paths_outside_workspace(self, tmp_path) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        backend = ACPBackend(command="fake-acp-agent")
        backend._cwd = tmp_path.resolve()
        client = acp_backend_module._AgenterACPClient(backend)

        with pytest.raises(BackendError, match="outside cwd"):
            await client.write_text_file("nope", "../outside.txt", "session-1")

        with pytest.raises(BackendError, match="outside cwd"):
            await client.read_text_file("../outside.txt", "session-1")

    @pytest.mark.asyncio
    async def test_modified_files_reports_paths_changed_during_execute(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        async def prompt_side_effect(session_id, prompt):
            (tmp_path / "created.py").write_text("print('hello')\n")
            return SimpleNamespace(stop_reason="end_turn")

        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
            prompt=AsyncMock(side_effect=prompt_side_effect),
        )
        context = FakeACPProcessContext(connection)
        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", lambda *args, **kwargs: context, raising=False)

        backend = ACPBackend(command="fake-acp-agent")
        await backend.connect(str(tmp_path))
        _ = [message async for message in backend.execute("create a file")]

        modified = backend.modified_files()

        assert isinstance(modified, PathsModifiedFiles)
        assert modified.file_paths == ["created.py"]

    @pytest.mark.asyncio
    async def test_modified_files_ignores_generated_python_bytecode_cache(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        async def prompt_side_effect(session_id, prompt):
            (tmp_path / "math_tools.py").write_text("def add(a, b):\n    return a + b\n")
            pycache = tmp_path / "__pycache__"
            pycache.mkdir()
            (pycache / "math_tools.cpython-314.pyc").write_bytes(b"\x00\x01")
            return SimpleNamespace(stop_reason="end_turn")

        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
            prompt=AsyncMock(side_effect=prompt_side_effect),
        )
        context = FakeACPProcessContext(connection)
        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", lambda *args, **kwargs: context, raising=False)

        backend = ACPBackend(command="fake-acp-agent")
        await backend.connect(str(tmp_path))
        _ = [message async for message in backend.execute("create a file")]

        assert backend.modified_files().file_paths == ["math_tools.py"]

    @pytest.mark.asyncio
    async def test_modified_files_are_cumulative_across_executes(self, tmp_path, monkeypatch) -> None:
        from agenter.coding_backends.acp import ACPBackend
        from agenter.coding_backends.acp import backend as acp_backend_module

        async def prompt_side_effect(session_id, prompt):
            if connection.prompt.await_count == 1:
                (tmp_path / "first.py").write_text("print('first')\n")
            else:
                (tmp_path / "second.py").write_text("print('second')\n")
            return SimpleNamespace(stop_reason="end_turn")

        connection = SimpleNamespace(
            initialize=AsyncMock(),
            new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
            prompt=AsyncMock(side_effect=prompt_side_effect),
        )
        context = FakeACPProcessContext(connection)
        monkeypatch.setattr(acp_backend_module, "spawn_agent_process", lambda *args, **kwargs: context, raising=False)

        backend = ACPBackend(command="fake-acp-agent")
        await backend.connect(str(tmp_path))
        _ = [message async for message in backend.execute("create first")]
        _ = [message async for message in backend.execute("create second")]

        modified = backend.modified_files()

        assert modified.file_paths == ["first.py", "second.py"]
