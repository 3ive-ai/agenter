"""Agent Client Protocol backend."""

from __future__ import annotations

import asyncio
import os
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel

from ...data_models import (
    BackendError,
    BackendMessage,
    ConfigurationError,
    PathsModifiedFiles,
    RefusalMessage,
    TextMessage,
    ToolCallMessage,
    ToolResult,
    Usage,
)

try:
    from acp import schema as acp_schema
    from acp import spawn_agent_process, text_block
except ImportError:
    acp_schema = None  # type: ignore[assignment]
    spawn_agent_process = None  # type: ignore[assignment]
    text_block = None  # type: ignore[assignment]

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


ACP_AUTONOMOUS_CONTRACT = """You are being called as a backend by Agenter.
Do not stop to ask for confirmation.
The user has already approved direct implementation.
If implementation details are missing, choose conservative defaults that match the existing workspace.
Modify files in the current workspace and verify the result when possible.
Only ask a question if the task is impossible, unsafe, or blocked by missing external credentials."""

ACP_AUTONOMOUS_CONTINUE_PROMPT = """Proceed now.
Do not ask for confirmation.
Implement the requested change directly in the current workspace and verify it when possible."""

_CONFIRMATION_REQUEST_PATTERNS = (
    re.compile(r"\breply\s+(?:with\s+)?(?:yes|y|ok|okay|confirm|approved?|proceed)\b", re.IGNORECASE),
    re.compile(r"\b(?:please\s+)?(?:confirm|approve)\b.*\b(?:before|then|and)\b", re.IGNORECASE),
    re.compile(r"\b(?:if|once)\s+.*\b(?:confirm|approve|reply|say\s+yes)\b", re.IGNORECASE),
    re.compile(r"\bwait(?:ing)?\s+for\s+(?:your\s+)?(?:confirmation|approval)\b", re.IGNORECASE),
)

_IGNORED_WORKSPACE_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)
_IGNORED_WORKSPACE_SUFFIXES = frozenset({".pyc", ".pyo"})


class _AgenterACPClient:
    """Minimal ACP client implementation used by the spawned agent."""

    def __init__(self, backend: ACPBackend) -> None:
        self._backend = backend

    async def request_permission(
        self,
        options: Any,
        session_id: str,
        tool_call: Any,
        **kwargs: Any,
    ) -> dict[str, dict[str, str]]:
        """Handle ACP permission requests according to backend policy."""
        if self._backend.permission_policy == "allow":
            option_id = self._select_allow_option_id(options)
            if option_id is not None:
                return {"outcome": {"outcome": "selected", "optionId": option_id}}
        return {"outcome": {"outcome": "cancelled"}}

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        """Capture streamed session updates for execute() to emit."""
        self._backend._pending_updates.append(update)
        if self._backend.update_callback is not None:
            try:
                self._backend.update_callback(update)
            except Exception:
                logger.exception("acp_update_callback_failed")

    async def write_text_file(self, content: str, path: str, session_id: str, **kwargs: Any) -> Any:
        """Write text into the connected workspace on behalf of the ACP agent."""
        resolved_path = self._backend._resolve_workspace_path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(content, encoding="utf-8")
        if acp_schema is not None:
            return acp_schema.WriteTextFileResponse()
        return {}

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Read text from the connected workspace for the ACP agent."""
        resolved_path = self._backend._resolve_workspace_path(path)
        content = resolved_path.read_text(encoding="utf-8")
        if line is not None or limit is not None:
            lines = content.splitlines(keepends=True)
            start = max((line or 1) - 1, 0)
            end = start + limit if limit is not None else None
            content = "".join(lines[start:end])
        if acp_schema is not None:
            return acp_schema.ReadTextFileResponse(content=content)
        return SimpleNamespace(content=content)

    def _select_allow_option_id(self, options: Any) -> str | None:
        option_list = options if isinstance(options, list) else []
        for option in option_list:
            data = self._backend._to_plain_data(option)
            if not isinstance(data, dict):
                continue
            kind = data.get("kind")
            option_id = data.get("optionId") or data.get("option_id")
            if kind in {"allow_once", "allow_always"} and option_id:
                return str(option_id)
        return None


class ACPBackend:
    """Backend that drives an ACP-compatible agent subprocess."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        mcp_servers: list[Any] | None = None,
        sandbox: bool = True,
        permission_policy: str = "deny",
        autonomous: bool = True,
        update_callback: Callable[[Any], None] | None = None,
    ) -> None:
        if permission_policy not in {"deny", "allow"}:
            raise ConfigurationError(
                "permission_policy must be 'deny' or 'allow'.",
                parameter="permission_policy",
                value=permission_policy,
            )
        self.command = command
        self.model = command
        self.args = args or []
        self.env = env or {}
        self.mcp_servers = mcp_servers or []
        self.sandbox = sandbox
        self.permission_policy = permission_policy
        self.autonomous = autonomous
        self.update_callback = update_callback
        self._cwd: Path | None = None
        self._connection: Any = None
        self._process_context: Any = None
        self._process: Any = None
        self._session_id: str | None = None
        self._pending_updates: list[Any] = []
        self._file_snapshot: dict[str, tuple[int, int]] = {}
        self._modified_paths: list[str] = []
        self._request_snapshot: dict[str, tuple[int, int]] | None = None
        self._request_modified_paths: list[str] = []
        self._turn_modified_paths: list[str] = []
        self._execute_lock = asyncio.Lock()
        self._prompt_active = False
        self._output_type: type[BaseModel] | None = None
        self._structured_output: BaseModel | None = None
        self._refusal: RefusalMessage | None = None
        self._system_prompt: str | None = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost_usd = 0.0
        self._usage_reported = False

    async def connect(
        self,
        cwd: str,
        allowed_write_paths: list[str] | None = None,
        resume_session_id: str | None = None,
        output_type: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Spawn the ACP agent process and create or resume a session."""
        if spawn_agent_process is None:
            raise BackendError(
                "agent-client-protocol is required for ACPBackend. Install with: pip install agenter[acp]",
                backend="acp",
            )

        if self._connection is not None:
            raise BackendError("ACPBackend is already connected.", backend="acp")

        self._cwd = Path(cwd).resolve()
        if output_type is not None:
            logger.warning("acp_output_type_ignored", reason="ACPBackend does not support structured output")
        self._output_type = output_type
        self._system_prompt = system_prompt
        client = _AgenterACPClient(self)
        spawn_kwargs: dict[str, Any] = {"transport_kwargs": {"limit": sys.maxsize}}
        if self.env:
            spawn_kwargs["env"] = {**os.environ, **self.env}

        if self.sandbox:
            logger.warning("acp_sandbox_depends_on_agent", command=self.command)

        spawn: Any = spawn_agent_process
        self._process_context = spawn(client, self.command, *self.args, **spawn_kwargs)
        self._connection, self._process = await self._process_context.__aenter__()
        initialization = await self._connection.initialize(protocol_version=1)
        if resume_session_id is None:
            session = await self._connection.new_session(cwd=str(self._cwd), mcp_servers=self.mcp_servers)
            self._session_id = self._extract_session_id(session)
        elif self._supports_resume(initialization):
            await self._connection.resume_session(
                cwd=str(self._cwd),
                session_id=resume_session_id,
                mcp_servers=self.mcp_servers,
            )
            self._session_id = resume_session_id
        elif self._supports_load(initialization):
            await self._connection.load_session(
                cwd=str(self._cwd),
                session_id=resume_session_id,
                mcp_servers=self.mcp_servers,
            )
            self._session_id = resume_session_id
        else:
            await self.disconnect()
            raise BackendError(
                "The ACP agent does not advertise session/resume or session/load support.",
                backend="acp",
            )
        self._file_snapshot = self._snapshot_files()
        self._request_snapshot = self._file_snapshot

    @property
    def session_id(self) -> str | None:
        """The live ACP session identifier, if connected."""
        return self._session_id

    @property
    def prompt_active(self) -> bool:
        """Whether an ACP prompt is currently running."""
        return self._prompt_active

    def begin_request(self) -> None:
        """Start request-local file tracking without resetting session state."""
        snapshot = self._snapshot_files()
        self._request_snapshot = snapshot
        self._request_modified_paths = []
        self._turn_modified_paths = []

    async def execute(self, prompt: str) -> AsyncIterator[BackendMessage]:
        """Execute a prompt through the connected ACP agent."""
        async with self._execute_lock:
            if self._connection is None or self._session_id is None:
                raise BackendError("ACPBackend is not connected. Call connect() before execute().", backend="acp")
            self._prompt_active = True
            try:
                self._refusal = None
                session_start_snapshot = self._file_snapshot
                request_start_snapshot = self._request_snapshot or session_start_snapshot
                turn_start_snapshot = self._snapshot_files()

                updates, after_snapshot = await self._send_prompt(self._format_prompt(prompt))
                turn_modified_paths = self._diff_snapshot(turn_start_snapshot, after_snapshot)
                modified_paths = self._diff_snapshot(session_start_snapshot, after_snapshot)
                request_modified_paths = self._diff_snapshot(request_start_snapshot, after_snapshot)
                all_updates = list(updates)
                update_batches = [list(updates)]

                if self.autonomous and not turn_modified_paths and self._looks_like_confirmation_request(updates):
                    logger.info("acp_auto_continuing_confirmation_request")
                    continuation_updates, after_snapshot = await self._send_prompt(ACP_AUTONOMOUS_CONTINUE_PROMPT)
                    update_batches.append(list(continuation_updates))
                    all_updates.extend(continuation_updates)
                    turn_modified_paths = self._diff_snapshot(turn_start_snapshot, after_snapshot)
                    modified_paths = self._diff_snapshot(session_start_snapshot, after_snapshot)
                    request_modified_paths = self._diff_snapshot(request_start_snapshot, after_snapshot)
                    if not turn_modified_paths and self._looks_like_confirmation_request(all_updates):
                        self._refusal = RefusalMessage(
                            reason="ACP agent asked for confirmation instead of modifying files.",
                            category="capability",
                        )

                self._turn_modified_paths = turn_modified_paths
                self._request_modified_paths = request_modified_paths
                self._modified_paths = modified_paths

                for updates_batch in update_batches:
                    for message in self._map_updates(updates_batch):
                        yield message
            finally:
                self._prompt_active = False

    async def _send_prompt(self, prompt: str) -> tuple[list[Any], dict[str, tuple[int, int]]]:
        self._pending_updates = []
        prompt_block = self._text_block(prompt)
        await self._connection.prompt(session_id=self._session_id, prompt=[prompt_block])
        return list(self._pending_updates), self._snapshot_files()

    def _format_prompt(self, prompt: str) -> str:
        parts: list[str] = []
        if self.autonomous:
            parts.append(ACP_AUTONOMOUS_CONTRACT)
        if self._system_prompt:
            parts.append(f"System instructions:\n{self._system_prompt}")
        if parts:
            parts.append(f"User task:\n{prompt}")
            return "\n\n".join(parts)
        return prompt

    @staticmethod
    def _text_block(prompt: str) -> Any:
        return text_block(prompt) if text_block is not None else {"type": "text", "text": prompt}

    @staticmethod
    def _extract_session_id(session: Any) -> str:
        if isinstance(session, dict):
            value = session.get("session_id") or session.get("sessionId")
        else:
            value = getattr(session, "session_id", None) or getattr(session, "sessionId", None)
        if not value:
            raise BackendError("ACP session/new response did not include a session ID.", backend="acp")
        return str(value)

    def _supports_resume(self, initialization: Any) -> bool:
        data = self._to_plain_data(initialization)
        if not isinstance(data, dict):
            return False
        capabilities = data.get("agentCapabilities") or data.get("agent_capabilities")
        if not isinstance(capabilities, dict):
            return False
        session_capabilities = capabilities.get("sessionCapabilities") or capabilities.get("session_capabilities")
        if not isinstance(session_capabilities, dict):
            return False
        return session_capabilities.get("resume") is not None

    def _supports_load(self, initialization: Any) -> bool:
        data = self._to_plain_data(initialization)
        if not isinstance(data, dict):
            return False
        capabilities = data.get("agentCapabilities") or data.get("agent_capabilities")
        if not isinstance(capabilities, dict):
            return False
        return bool(capabilities.get("loadSession") or capabilities.get("load_session"))

    def modified_files(self) -> PathsModifiedFiles:
        """Return cumulative file changes since connect()."""
        return PathsModifiedFiles(file_paths=self._modified_paths)

    def request_modified_files(self) -> PathsModifiedFiles:
        """Return file changes made by the current high-level request."""
        return PathsModifiedFiles(file_paths=self._request_modified_paths)

    def turn_modified_files(self) -> PathsModifiedFiles:
        """Return file changes made by the most recent ACP prompt."""
        return PathsModifiedFiles(file_paths=self._turn_modified_paths)

    def usage(self) -> Usage:
        return Usage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=self._cost_usd,
            provider="acp",
            reported=self._usage_reported,
        )

    def structured_output(self) -> BaseModel | None:
        return self._structured_output

    def refusal(self) -> RefusalMessage | None:
        return self._refusal

    async def cancel(self) -> None:
        """Cancel the active ACP prompt while keeping the session reusable."""
        if self._connection is None or self._session_id is None:
            raise BackendError("ACPBackend is not connected. Call connect() before cancel().", backend="acp")
        if not self._prompt_active:
            return
        await self._connection.cancel(session_id=self._session_id)

    async def disconnect(self) -> None:
        """Close the ACP process context."""
        if self._process_context is not None:
            try:
                await self._process_context.__aexit__(None, None, None)
            except Exception as exc:
                logger.warning("acp_disconnect_error", error=str(exc))
        self._connection = None
        self._process_context = None
        self._process = None
        self._session_id = None
        self._pending_updates = []
        self._file_snapshot = {}
        self._modified_paths = []
        self._request_snapshot = None
        self._request_modified_paths = []
        self._turn_modified_paths = []
        self._prompt_active = False
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost_usd = 0.0
        self._usage_reported = False

    def _snapshot_files(self) -> dict[str, tuple[int, int]]:
        if self._cwd is None or not self._cwd.exists():
            return {}

        snapshot: dict[str, tuple[int, int]] = {}
        for path in self._cwd.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(self._cwd)
            if self._should_ignore_workspace_file(relative_path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[str(relative_path)] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

    @staticmethod
    def _should_ignore_workspace_file(path: Path) -> bool:
        return any(part in _IGNORED_WORKSPACE_DIRS for part in path.parts) or path.suffix in _IGNORED_WORKSPACE_SUFFIXES

    @staticmethod
    def _diff_snapshot(
        before: dict[str, tuple[int, int]],
        after: dict[str, tuple[int, int]],
    ) -> list[str]:
        paths = set(before) | set(after)
        return sorted(path for path in paths if before.get(path) != after.get(path))

    def _resolve_workspace_path(self, path: str) -> Path:
        if self._cwd is None:
            raise BackendError("ACPBackend has no connected cwd.", backend="acp")
        root = self._cwd.resolve()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise BackendError(f"ACP file path is outside cwd: {path}", backend="acp", cause=exc) from exc
        return resolved

    def _looks_like_confirmation_request(self, updates: list[Any]) -> bool:
        text = self._updates_text(updates)
        if not text:
            return False
        return any(pattern.search(text) for pattern in _CONFIRMATION_REQUEST_PATTERNS)

    def _updates_text(self, updates: list[Any]) -> str:
        parts: list[str] = []
        for update in updates:
            data = self._to_plain_data(update)
            text = self._extract_text(data)
            if text:
                parts.append(text)
        return " ".join(parts)

    def _map_updates(self, updates: list[Any]) -> list[BackendMessage]:
        messages: list[BackendMessage] = []
        text_parts: list[str] = []
        ignored_update_types: Counter[str] = Counter()

        def flush_text() -> None:
            if text_parts:
                messages.append(TextMessage(content="".join(text_parts)))
                text_parts.clear()

        for update in updates:
            data = self._to_plain_data(update)
            update_kind = self._update_kind(data)

            if update_kind == "usage_update":
                self._capture_usage_update(data)
                continue

            if update_kind == "plan":
                flush_text()
                plan_message = self._map_plan_update(data)
                if plan_message is not None:
                    messages.append(plan_message)
                continue

            if update_kind == "tool_call":
                flush_text()
                messages.append(self._map_tool_call_start(data))
                continue

            if update_kind == "tool_call_update":
                flush_text()
                messages.append(self._map_tool_call_progress(data))
                continue

            if update_kind in {
                "agent_thought_chunk",
                "available_commands_update",
                "config_option_update",
                "current_mode_update",
                "session_info_update",
            }:
                ignored_update_types[update_kind] += 1
                continue

            text = self._extract_text(data)
            if text:
                text_parts.append(text)
                continue

            tool_name = self._extract_tool_name(data)
            if tool_name:
                flush_text()
                messages.append(ToolCallMessage(tool_name=tool_name, args=self._extract_tool_args(data)))
                continue

            ignored_update_types[str(update_kind or "unknown")] += 1

        flush_text()
        if ignored_update_types:
            logger.debug("acp_updates_ignored", update_types=dict(ignored_update_types))
        return messages

    def _update_kind(self, data: Any) -> str | None:
        if not isinstance(data, dict):
            return None
        value = data.get("sessionUpdate") or data.get("session_update") or data.get("type")
        return str(value) if value else None

    def _map_plan_update(self, data: Any) -> TextMessage | None:
        if not isinstance(data, dict):
            return None
        entries = data.get("entries")
        if not isinstance(entries, list) or not entries:
            return None

        lines = ["Plan:"]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content")
            if not content:
                continue
            status = entry.get("status")
            prefix = f"[{status}] " if status else ""
            lines.append(f"- {prefix}{content}")
        if len(lines) == 1:
            return None
        return TextMessage(content="\n".join(lines))

    def _map_tool_call_start(self, data: Any) -> ToolCallMessage:
        tool_name = self._tool_update_title(data)
        return ToolCallMessage(tool_name=tool_name, args=self._tool_update_args(data))

    def _map_tool_call_progress(self, data: Any) -> ToolResult:
        tool_name = self._tool_update_title(data)
        status = self._tool_update_status(data)
        target = ", ".join(self._tool_update_paths(data))
        output = f"{status}: {target}" if target else status
        success = status.lower() not in {"cancelled", "error", "failed", "failure"}
        return ToolResult(tool_name=tool_name, output=output, success=success)

    def _tool_update_title(self, data: Any) -> str:
        if not isinstance(data, dict):
            return "tool"
        value = data.get("title") or data.get("kind") or data.get("toolCallId") or data.get("tool_call_id") or "tool"
        return str(value)

    def _tool_update_status(self, data: Any) -> str:
        if not isinstance(data, dict):
            return "updated"
        return str(data.get("status") or "updated")

    def _tool_update_args(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            return {}

        args: dict[str, Any] = {}
        tool_call_id = data.get("toolCallId") or data.get("tool_call_id")
        if tool_call_id:
            args["id"] = str(tool_call_id)
        for source_key, arg_key in (("kind", "kind"), ("status", "status"), ("rawInput", "raw_input")):
            value = data.get(source_key)
            if value is not None:
                args[arg_key] = value

        paths = self._tool_update_paths(data)
        if len(paths) == 1:
            args["path"] = paths[0]
        elif paths:
            args["paths"] = paths
        return args

    def _tool_update_paths(self, data: Any) -> list[str]:
        if not isinstance(data, dict):
            return []

        paths: list[str] = []
        content = data.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("path"):
                    paths.append(str(item["path"]))

        locations = data.get("locations")
        if isinstance(locations, list):
            for item in locations:
                if isinstance(item, dict) and item.get("path"):
                    location = str(item["path"])
                    if item.get("line") is not None:
                        location = f"{location}:{item['line']}"
                    paths.append(location)

        return list(dict.fromkeys(paths))

    def _capture_usage_update(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        used = data.get("used")
        if not isinstance(used, dict):
            return
        self._usage_reported = True
        # ACP usage_update reports cumulative session totals, not per-turn deltas.
        # Assignment (not +=) is correct: the last update already includes all prior usage.
        self._input_tokens = self._usage_int(used, "inputTokens", "input_tokens")
        self._output_tokens = self._usage_int(used, "outputTokens", "output_tokens")
        if data.get("cost") is not None:
            self._cost_usd = float(data["cost"])

    @staticmethod
    def _usage_int(data: dict[str, Any], *keys: str) -> int:
        for key in keys:
            value = data.get(key)
            if value is not None:
                return int(value)
        return 0

    def _to_plain_data(self, value: Any, _depth: int = 0) -> Any:
        if _depth > 10:
            return str(value)
        if isinstance(value, BaseModel):
            return value.model_dump(mode="python", by_alias=True)
        if isinstance(value, dict):
            return {key: self._to_plain_data(item, _depth + 1) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_plain_data(item, _depth + 1) for item in value]
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="python", by_alias=True)
        if hasattr(value, "__dict__") and not isinstance(value, str):
            return {
                key: self._to_plain_data(item, _depth + 1)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        return value

    def _extract_text(self, data: Any) -> str | None:
        if isinstance(data, str):
            return data
        if isinstance(data, list):
            parts = [part for item in data if (part := self._extract_text(item))]
            return "".join(parts) if parts else None
        if not isinstance(data, dict):
            return None

        for key in ("text", "delta"):
            value = data.get(key)
            if isinstance(value, str):
                return value

        for key in ("content", "message", "agentMessage", "agent_message"):
            value = data.get(key)
            text = self._extract_text(value)
            if text:
                return text
        return None

    def _extract_tool_name(self, data: Any) -> str | None:
        if not isinstance(data, dict):
            return None
        tool = data.get("toolCall") or data.get("tool_call") or data.get("tool")
        if isinstance(tool, dict):
            name = tool.get("name") or tool.get("title") or tool.get("tool_name")
            return str(name) if name else None
        name = data.get("tool_name")
        return str(name) if name else None

    def _extract_tool_args(self, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            return {}
        tool = data.get("toolCall") or data.get("tool_call") or data.get("tool")
        if isinstance(tool, dict):
            args = tool.get("args") or tool.get("input") or tool.get("arguments")
            return args if isinstance(args, dict) else {}
        args = data.get("args")
        return args if isinstance(args, dict) else {}
