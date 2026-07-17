"""Tests for persistent multi-request ACP sessions."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agenter import AutonomousCodingAgent, Budget, CodingStatus
from agenter.data_models import BackendError


class FakeACPProcessContext:
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


def _new_session_initialization() -> dict[str, object]:
    return {"agentCapabilities": {}}


@pytest.mark.asyncio
async def test_facade_reuses_one_acp_session_across_followups(tmp_path, monkeypatch) -> None:
    from agenter.coding_backends.acp import backend as acp_backend_module

    spawned: dict[str, object] = {}

    async def prompt_side_effect(session_id, prompt):
        prompt_number = connection.prompt.await_count
        (tmp_path / f"turn_{prompt_number}.py").write_text(f"turn = {prompt_number}\n")
        await spawned["client"].session_update(
            session_id,
            {
                "sessionUpdate": "usage_update",
                "used": {"inputTokens": prompt_number * 6, "outputTokens": prompt_number * 4},
                "cost": prompt_number * 0.01,
            },
        )
        return SimpleNamespace(stop_reason="end_turn")

    connection = SimpleNamespace(
        initialize=AsyncMock(return_value=_new_session_initialization()),
        new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
        prompt=AsyncMock(side_effect=prompt_side_effect),
    )
    context = FakeACPProcessContext(connection)

    def fake_spawn_agent_process(client, command, *args, **kwargs):
        spawned["client"] = client
        return context

    monkeypatch.setattr(acp_backend_module, "spawn_agent_process", fake_spawn_agent_process, raising=False)

    agent = AutonomousCodingAgent(
        backend="acp",
        acp_command="fake-acp-agent",
        acp_autonomous=False,
        validators=[],
    )
    session = await agent.open_session(str(tmp_path))

    first = await session.execute("create the first file")
    second = await session.execute("now create the second file")

    connection.new_session.assert_awaited_once()
    assert [call.kwargs["session_id"] for call in connection.prompt.await_args_list] == ["session-1", "session-1"]
    assert context.exited is False
    assert first.files == {"turn_1.py": "turn = 1\n"}
    assert second.files == {"turn_2.py": "turn = 2\n"}
    assert first.total_tokens == 10
    assert second.total_tokens == 10
    assert first.usage_reported is True
    assert second.usage_reported is True
    assert first.session_total_tokens == 10
    assert second.session_total_tokens == 20
    assert first.request_index == 1
    assert second.request_index == 2
    assert first.session_id == second.session_id == session.session_id == "session-1"
    assert session.modified_files().file_paths == ["turn_1.py", "turn_2.py"]

    await session.close()
    assert context.exited is True
    assert session.usage().total_tokens == 20
    assert session.modified_files().file_paths == ["turn_1.py", "turn_2.py"]
    with pytest.raises(BackendError, match="closed"):
        await session.execute("too late")


@pytest.mark.asyncio
async def test_streamed_followup_events_have_stable_trace_ids(tmp_path, monkeypatch) -> None:
    from agenter.coding_backends.acp import backend as acp_backend_module

    connection = SimpleNamespace(
        initialize=AsyncMock(return_value=_new_session_initialization()),
        new_session=AsyncMock(return_value=SimpleNamespace(session_id="trace-session")),
        prompt=AsyncMock(return_value=SimpleNamespace(stop_reason="end_turn")),
    )
    context = FakeACPProcessContext(connection)
    monkeypatch.setattr(acp_backend_module, "spawn_agent_process", lambda *args, **kwargs: context, raising=False)

    agent = AutonomousCodingAgent(
        backend="acp",
        acp_command="fake-acp-agent",
        acp_autonomous=False,
        validators=[],
    )
    session = await agent.open_session(str(tmp_path))

    events = [event async for event in session.stream_execute("inspect the workspace")]

    assert events
    assert all(event.session_id == "trace-session" for event in events)
    assert all(event.request_index == 1 for event in events)
    terminal_result = next(event.result for event in events if event.result is not None)
    assert terminal_result.session_id == "trace-session"
    assert terminal_result.request_index == 1
    assert terminal_result.usage_reported is False
    await session.close()


@pytest.mark.asyncio
async def test_concurrent_followups_are_serialized(tmp_path, monkeypatch) -> None:
    from agenter.coding_backends.acp import backend as acp_backend_module

    active_prompts = 0
    max_active_prompts = 0

    async def prompt_side_effect(session_id, prompt):
        nonlocal active_prompts, max_active_prompts
        active_prompts += 1
        max_active_prompts = max(max_active_prompts, active_prompts)
        await asyncio.sleep(0.01)
        active_prompts -= 1
        return SimpleNamespace(stop_reason="end_turn")

    connection = SimpleNamespace(
        initialize=AsyncMock(return_value=_new_session_initialization()),
        new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
        prompt=AsyncMock(side_effect=prompt_side_effect),
    )
    context = FakeACPProcessContext(connection)
    monkeypatch.setattr(acp_backend_module, "spawn_agent_process", lambda *args, **kwargs: context, raising=False)
    agent = AutonomousCodingAgent(
        backend="acp",
        acp_command="fake-acp-agent",
        acp_autonomous=False,
        validators=[],
    )
    session = await agent.open_session(str(tmp_path))

    first, second = await asyncio.gather(session.execute("first"), session.execute("second"))

    assert max_active_prompts == 1
    assert (first.request_index, second.request_index) == (1, 2)
    await session.close()


@pytest.mark.asyncio
async def test_cancel_keeps_persistent_session_reusable(tmp_path, monkeypatch) -> None:
    from agenter.coding_backends.acp import backend as acp_backend_module

    prompt_started = asyncio.Event()
    prompt_released = asyncio.Event()

    async def prompt_side_effect(session_id, prompt):
        if connection.prompt.await_count == 1:
            prompt_started.set()
            await prompt_released.wait()
        return SimpleNamespace(stop_reason="end_turn")

    async def cancel_side_effect(session_id):
        prompt_released.set()

    connection = SimpleNamespace(
        initialize=AsyncMock(return_value=_new_session_initialization()),
        new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
        prompt=AsyncMock(side_effect=prompt_side_effect),
        cancel=AsyncMock(side_effect=cancel_side_effect),
    )
    context = FakeACPProcessContext(connection)
    monkeypatch.setattr(acp_backend_module, "spawn_agent_process", lambda *args, **kwargs: context, raising=False)
    agent = AutonomousCodingAgent(
        backend="acp",
        acp_command="fake-acp-agent",
        acp_autonomous=False,
        validators=[],
    )
    session = await agent.open_session(str(tmp_path))

    active_request = asyncio.create_task(session.execute("long task"))
    await prompt_started.wait()
    await session.cancel()
    await active_request
    followup = await session.execute("follow-up after cancellation")

    connection.cancel.assert_awaited_once_with(session_id="session-1")
    assert followup.status == CodingStatus.COMPLETED
    assert followup.request_index == 2
    await session.close()


@pytest.mark.asyncio
async def test_session_budget_is_cumulative_across_followups(tmp_path, monkeypatch) -> None:
    from agenter.coding_backends.acp import backend as acp_backend_module

    connection = SimpleNamespace(
        initialize=AsyncMock(return_value=_new_session_initialization()),
        new_session=AsyncMock(return_value=SimpleNamespace(session_id="session-1")),
        prompt=AsyncMock(return_value=SimpleNamespace(stop_reason="end_turn")),
    )
    context = FakeACPProcessContext(connection)
    monkeypatch.setattr(acp_backend_module, "spawn_agent_process", lambda *args, **kwargs: context, raising=False)
    agent = AutonomousCodingAgent(
        backend="acp",
        acp_command="fake-acp-agent",
        acp_autonomous=False,
        validators=[],
    )
    session = await agent.open_session(str(tmp_path), session_budget=Budget(max_iterations=1))

    first = await session.execute("first")
    second = await session.execute("second")

    assert first.status == CodingStatus.COMPLETED_WITH_LIMIT_EXCEEDED
    assert second.status == CodingStatus.BUDGET_EXCEEDED
    assert connection.prompt.await_count == 1
    await session.close()


@pytest.mark.asyncio
async def test_acp_resume_uses_advertised_resume_capability(tmp_path, monkeypatch) -> None:
    from agenter.coding_backends.acp import ACPBackend
    from agenter.coding_backends.acp import backend as acp_backend_module

    connection = SimpleNamespace(
        initialize=AsyncMock(return_value={"agentCapabilities": {"sessionCapabilities": {"resume": {}}}}),
        new_session=AsyncMock(),
        resume_session=AsyncMock(return_value={}),
    )
    context = FakeACPProcessContext(connection)
    monkeypatch.setattr(acp_backend_module, "spawn_agent_process", lambda *args, **kwargs: context, raising=False)
    backend = ACPBackend(command="fake-acp-agent")

    await backend.connect(str(tmp_path), resume_session_id="existing-session")

    connection.new_session.assert_not_awaited()
    connection.resume_session.assert_awaited_once_with(
        cwd=str(tmp_path.resolve()),
        session_id="existing-session",
        mcp_servers=[],
    )
    assert backend.session_id == "existing-session"
    await backend.disconnect()


@pytest.mark.asyncio
async def test_acp_resume_falls_back_to_legacy_load(tmp_path, monkeypatch) -> None:
    from agenter.coding_backends.acp import ACPBackend
    from agenter.coding_backends.acp import backend as acp_backend_module

    connection = SimpleNamespace(
        initialize=AsyncMock(return_value={"agentCapabilities": {"loadSession": True}}),
        new_session=AsyncMock(),
        load_session=AsyncMock(return_value={}),
    )
    context = FakeACPProcessContext(connection)
    monkeypatch.setattr(acp_backend_module, "spawn_agent_process", lambda *args, **kwargs: context, raising=False)
    backend = ACPBackend(command="fake-acp-agent")

    await backend.connect(str(tmp_path), resume_session_id="legacy-session")

    connection.new_session.assert_not_awaited()
    connection.load_session.assert_awaited_once_with(
        cwd=str(tmp_path.resolve()),
        session_id="legacy-session",
        mcp_servers=[],
    )
    assert backend.session_id == "legacy-session"
    await backend.disconnect()


@pytest.mark.asyncio
async def test_acp_resume_fails_explicitly_when_not_supported(tmp_path, monkeypatch) -> None:
    from agenter.coding_backends.acp import ACPBackend
    from agenter.coding_backends.acp import backend as acp_backend_module

    connection = SimpleNamespace(
        initialize=AsyncMock(return_value=_new_session_initialization()),
        new_session=AsyncMock(),
    )
    context = FakeACPProcessContext(connection)
    monkeypatch.setattr(acp_backend_module, "spawn_agent_process", lambda *args, **kwargs: context, raising=False)
    backend = ACPBackend(command="fake-acp-agent")

    with pytest.raises(BackendError, match="does not advertise"):
        await backend.connect(str(tmp_path), resume_session_id="missing-session")

    assert context.exited is True
    connection.new_session.assert_not_awaited()
