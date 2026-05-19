"""Codex backend using OpenAI Codex CLI via MCP server.

This backend wraps the Codex CLI running as an MCP server to provide
coding agent capabilities powered by OpenAI's reasoning models.

Requires the codex optional dependency:
    pip install agenter[codex]

The Codex CLI must be installed separately:
    npm install -g @openai/codex

Features:
    - Persistent conversation sessions via conversationId
    - Custom MCP server configurations for additional tools
    - Configurable approval policies and sandbox modes
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel

from ...data_models import (
    BackendError,
    BackendMessage,
    ConfigurationError,
    PathsModifiedFiles,
    PromptMessage,
    TextMessage,
    ToolCallMessage,
    ToolResult,
    Usage,
)
from ...pricing import calculate_cost_usd
from ..base import BaseBackend
from ..output_parser import parse_structured_output
from ..refusal import parse_refusal_from_tool_call
from .constants import (
    CODEX_APPROVAL_POLICIES,
    CODEX_DEFAULT_APPROVAL_POLICY,
    CODEX_DEFAULT_MODEL,
    CODEX_DEFAULT_SANDBOX,
    CODEX_DEFAULT_TIMEOUT_SECONDS,
    CODEX_REASONING_EFFORTS,
    CODEX_SANDBOX_MODES,
    CODEX_TOOL_REPLY,
    CODEX_TOOL_START,
    FILE_MODIFICATION_TOOLS,
    FILE_PATH_ARG_KEYS,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from ...tools import Tool

logger = structlog.get_logger(__name__)

# Pattern to extract params dict from MCP validation warning logs
# Example: "params={'type': 'codex_event', 'msg': {...}} jsonrpc='2.0'"
_PARAMS_PATTERN = re.compile(r"params=(\{.+\})\s+jsonrpc=", re.DOTALL)


@contextmanager
def _capture_codex_events() -> Iterator[list[dict[str, Any]]]:
    """Capture codex/event notifications from MCP warning logs.

    Codex MCP server sends non-standard 'codex/event' notifications that fail
    MCP validation. The MCP library logs these as warnings. This context manager
    captures those warnings and extracts the event params using ast.literal_eval.

    Yields:
        List that will be populated with captured event params dicts.
    """
    events: list[dict[str, Any]] = []

    class CodexEventHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if "codex/event" not in msg:
                return
            if match := _PARAMS_PATTERN.search(msg):
                with contextlib.suppress(ValueError, SyntaxError):
                    events.append(ast.literal_eval(match.group(1)))

    handler = CodexEventHandler(level=logging.WARNING)
    logging.getLogger().addHandler(handler)
    try:
        yield events
    finally:
        logging.getLogger().removeHandler(handler)


class CodexMCPServer(BaseModel):
    """Configuration for a custom MCP server to pass to Codex.

    Custom MCP servers give Codex access to additional tools beyond its
    built-in capabilities (file operations, shell commands, etc.).

    Example:
        CodexMCPServer(
            name="context7",
            command="npx",
            args=["-y", "@upstash/context7-mcp"],
        )
    """

    name: str
    command: str
    args: list[str] | None = None
    env: dict[str, str] | None = None


class CodexBackend(BaseBackend):
    """Backend using OpenAI Codex CLI via MCP server.

    Spawns `codex mcp-server` subprocess and communicates via MCP protocol.
    Uses the OpenAI Agents SDK's MCPServerStdio for process lifecycle management.

    Custom MCP servers can be passed to give Codex access to additional tools
    during execution.

    Example:
        # Basic usage
        backend = CodexBackend()
        await backend.connect("/path/to/project")
        async for message in backend.execute("Fix the bug"):
            print(message)

        # With custom MCP servers
        backend = CodexBackend(
            mcp_servers=[
                CodexMCPServer(name="playwright", command="npx", args=["@playwright/mcp"]),
            ]
        )

    Note:
        Requires Codex CLI installed: npm install -g @openai/codex
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        approval_policy: str = CODEX_DEFAULT_APPROVAL_POLICY,
        sandbox: str = CODEX_DEFAULT_SANDBOX,
        mcp_servers: list[CodexMCPServer] | None = None,
        extra_tools: list[Tool] | None = None,
        timeout_seconds: float = CODEX_DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the Codex backend.

        Args:
            model: Model to use. Defaults to "gpt-5.4".
            reasoning_effort: Optional OpenAI/Codex reasoning effort.
                Valid values: "minimal", "low", "medium", "high".
            approval_policy: Approval policy for tool execution.
                Options: "untrusted", "on-request", "on-failure", "never".
                Defaults to "never" for autonomous operation.
            sandbox: Sandbox mode for file operations.
                Options: "read-only", "workspace-write", "danger-full-access".
                Defaults to "workspace-write".
            mcp_servers: Custom MCP servers to pass to Codex, giving it
                access to additional tools.
            extra_tools: Custom agenter tools to add. These are serialized
                and run in a subprocess MCP server.
            timeout_seconds: Timeout for MCP session in seconds.
                Defaults to 3600 (1 hour) since complex coding tasks take time.

        Raises:
            ConfigurationError: If approval_policy or sandbox is invalid.
        """
        if approval_policy not in CODEX_APPROVAL_POLICIES:
            raise ConfigurationError(
                f"Invalid approval_policy: {approval_policy!r}. "
                f"Must be one of: {', '.join(sorted(CODEX_APPROVAL_POLICIES))}",
                parameter="approval_policy",
                value=approval_policy,
            )
        if sandbox not in CODEX_SANDBOX_MODES:
            raise ConfigurationError(
                f"Invalid sandbox: {sandbox!r}. Must be one of: {', '.join(sorted(CODEX_SANDBOX_MODES))}",
                parameter="sandbox",
                value=sandbox,
            )
        if reasoning_effort is not None and reasoning_effort not in CODEX_REASONING_EFFORTS:
            raise ConfigurationError(
                f"Invalid reasoning_effort: {reasoning_effort!r}. "
                f"Must be one of: {', '.join(sorted(CODEX_REASONING_EFFORTS))}",
                parameter="reasoning_effort",
                value=reasoning_effort,
            )

        self._model = model or CODEX_DEFAULT_MODEL
        self.model = self._model  # Public attribute for session display
        self._reasoning_effort = reasoning_effort
        self._approval_policy = approval_policy
        self._sandbox = sandbox
        self._mcp_servers = mcp_servers or []
        self._extra_tools = extra_tools or []
        self._timeout_seconds = timeout_seconds

        # Initialize common state (tokens, structured output, refusal)
        self._init_state()

        # Connection state
        self._mcp_server: Any = None  # MCPServerStdio instance
        self._tool_pickle_path: Path | None = None  # Temp file for serialized tools
        self._conversation_id: str | None = None
        self._cwd: Path | None = None
        self._custom_system_prompt: str | None = None

        # File tracking
        self._modified_files: set[str] = set()

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken for accurate cost estimation."""
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(self._model)
            return len(encoding.encode(text))
        except Exception:
            return len(text) // 4  # Fallback estimate

    def _create_mcp_server_for_tools(self) -> CodexMCPServer | None:
        """Convert agenter Tools to subprocess MCP server.

        Creates a subprocess MCP server that hosts the custom tools.
        Tools are serialized with cloudpickle and loaded by the subprocess.

        Returns:
            CodexMCPServer config, or None if no extra tools configured.
        """
        if not self._extra_tools:
            return None

        try:
            import cloudpickle  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("cloudpickle not available for custom tools. Install with: pip install agenter[codex]")
            return None

        import tempfile

        # Serialize tools to temp file
        fd, path = tempfile.mkstemp(suffix=".pkl", prefix="agenter_tools_")
        try:
            with open(fd, "wb") as f:
                cloudpickle.dump(self._extra_tools, f)
            self._tool_pickle_path = Path(path)
        except TypeError as e:
            # Tools contain unpicklable objects (e.g., thread locks, async clients)
            logger.warning("custom_tools_not_picklable", error=str(e))
            # Clean up temp file on failure
            with contextlib.suppress(Exception):
                Path(path).unlink(missing_ok=True)
            return None

        logger.debug("tools_serialized", path=path, count=len(self._extra_tools))

        # Return CodexMCPServer config (subprocess-based)
        # Use sys.executable to ensure we use the same Python that has agenter installed
        import sys

        return CodexMCPServer(
            name="agenter-tools",
            command=sys.executable,
            args=["-m", "agenter.coding_backends.codex.mcp_tool_server", "--tools", path],
        )

    def _build_codex_config(self) -> dict[str, Any] | None:
        """Build config dict with MCP servers for Codex.

        Returns:
            Config dict to pass to Codex tool, or None if no custom config.
        """
        # Combine user-provided MCP servers with auto-generated tool server
        all_servers = list(self._mcp_servers)

        # Add tool server if we have custom tools
        tool_server = self._create_mcp_server_for_tools()
        if tool_server:
            all_servers.append(tool_server)

        config: dict[str, Any] = {}

        if self._reasoning_effort:
            config["model_reasoning_effort"] = self._reasoning_effort

        if all_servers:
            mcp_config: dict[str, Any] = {}
            for server in all_servers:
                server_config: dict[str, Any] = {"command": server.command}
                if server.args:
                    server_config["args"] = server.args
                if server.env:
                    server_config["env"] = server.env
                mcp_config[server.name] = server_config
            config["mcp_servers"] = mcp_config

        return config or None

    async def connect(
        self,
        cwd: str,
        allowed_write_paths: list[str] | None = None,
        resume_session_id: str | None = None,
        output_type: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize with a working directory and start Codex MCP server.

        Args:
            cwd: Working directory for file operations.
            allowed_write_paths: Optional glob patterns restricting writes.
                Note: Codex uses sandbox mode instead; this is logged but
                actual enforcement depends on sandbox setting.
            resume_session_id: Optional session ID to resume. Codex uses
                conversationId internally; this will be used if provided.
            output_type: Optional Pydantic model for structured output.
                Note: Codex MCP doesn't directly support structured output;
                this will attempt to parse from response.
            system_prompt: Custom system prompt to include with prompts.
        """
        try:
            from agents.mcp import MCPServerStdio
        except ImportError as e:
            raise BackendError(
                "openai-agents is required for CodexBackend. Install with: pip install agenter[codex]",
                backend="codex",
                cause=e,
            ) from e

        self._cwd = Path(cwd)
        self._custom_system_prompt = system_prompt
        self._conversation_id = resume_session_id
        self._modified_files = set()

        # Snapshot existing files so we can diff after execution
        self._pre_existing_files: dict[str, float] = {}
        for f in self._cwd.rglob("*"):
            if f.is_file():
                self._pre_existing_files[str(f)] = f.stat().st_mtime

        # Reset common state and set output_type
        self._reset_state()
        self._output_type = output_type

        if allowed_write_paths:
            logger.warning(
                f"allowed_write_paths is set but Codex uses sandbox mode instead. Current sandbox: {self._sandbox}",
            )

        if output_type:
            logger.debug(
                "structured_output_via_prompt",
                output_type=output_type.__name__,
            )

        # Check codex CLI is installed
        if not shutil.which("codex"):
            raise BackendError(
                "Codex CLI not found. Install with: npm install -g @openai/codex",
                backend="codex",
            )

        # The codex CLI reads API keys from ~/.codex/auth.json, not from
        # OPENAI_API_KEY env var. Sync the env var into codex via `codex login`.
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            proc = await asyncio.create_subprocess_exec(
                "codex",
                "login",
                "--with-api-key",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate(input=api_key.encode())
            if proc.returncode == 0:
                logger.info("codex_auth_synced", source="OPENAI_API_KEY")
            else:
                logger.warning("codex_auth_sync_failed", returncode=proc.returncode)

        # Start Codex MCP server subprocess
        self._mcp_server = MCPServerStdio(
            name="codex",
            params={
                "command": "codex",
                "args": ["mcp-server"],
            },
            client_session_timeout_seconds=self._timeout_seconds,
        )
        try:
            await self._mcp_server.connect()
        except Exception:
            # Clean up on connection failure
            with contextlib.suppress(Exception):
                await self._mcp_server.cleanup()
            self._mcp_server = None
            raise
        logger.debug("codex_mcp_server_started", cwd=str(self._cwd))

    async def execute(self, prompt: str) -> AsyncIterator[BackendMessage]:
        """Execute a prompt using Codex via MCP.

        Args:
            prompt: The user prompt to execute.

        Yields:
            BackendMessage objects for each significant step.
        """
        if self._mcp_server is None or self._cwd is None:
            raise BackendError(
                "Backend not connected. Call connect() first.",
                backend="codex",
            )

        # Build full prompt with optional system context
        full_prompt = f"{self._custom_system_prompt}\n\n{prompt}" if self._custom_system_prompt else prompt

        # If structured output is requested, instruct the LLM to respond with JSON
        if self._output_type is not None:
            schema = json.dumps(self._output_type.model_json_schema(), indent=2)
            full_prompt += f"\n\nRespond ONLY with a JSON object matching this schema:\n```json\n{schema}\n```"

        # Track input tokens using tiktoken
        self._input_tokens += self._count_tokens(full_prompt)

        # Yield PromptMessage for tracing
        yield PromptMessage(user_prompt=prompt, system_prompt=self._custom_system_prompt)

        # Choose tool based on conversation state
        if self._conversation_id is None:
            tool_name = CODEX_TOOL_START
            args: dict[str, Any] = {
                "prompt": full_prompt,
                "cwd": str(self._cwd),
                "approval-policy": self._approval_policy,
                "sandbox": self._sandbox,
                "model": self._model,
            }
            # Add custom MCP server config if provided
            config = self._build_codex_config()
            if config:
                args["config"] = config
        else:
            tool_name = CODEX_TOOL_REPLY
            args = {
                "prompt": full_prompt,
                "conversationId": self._conversation_id,
            }

        logger.debug(
            "calling_codex_tool",
            tool=tool_name,
            has_conversation_id=self._conversation_id is not None,
        )

        try:
            # Capture codex/event notifications from MCP warning logs
            with _capture_codex_events() as captured_events:
                result = await self._mcp_server.call_tool(tool_name, args)

            # Parse and yield messages from captured codex events first
            for params in captured_events:
                for msg in self._parse_codex_event(params):
                    yield msg

            # Parse and yield messages from MCP response, tracking last text for structured output
            last_text_content = None
            for msg in self._parse_codex_response(result):
                if isinstance(msg, TextMessage):
                    last_text_content = msg.content
                yield msg

            # Parse structured output from last text response
            self._structured_output = parse_structured_output(last_text_content, self._output_type)

        except Exception as e:
            # Handle Pydantic ValidationError from malformed MCP responses
            # Codex CLI returns {'error': '...'} instead of proper CallToolResult
            if hasattr(e, "errors") and callable(e.errors):
                for err in e.errors():
                    if "input" in err and isinstance(err["input"], dict) and "error" in err["input"]:
                        error_msg = str(err["input"]["error"])
                        if "model_not_found" in error_msg.lower():
                            raise BackendError(
                                f"Invalid model '{self._model}' for Codex backend",
                                backend="codex",
                                cause=e,
                            ) from e
                        raise BackendError(
                            f"Codex error: {error_msg}",
                            backend="codex",
                            cause=e,
                        ) from e

            # Fallback: check error string for known patterns
            error_str = str(e).lower()
            if "model_not_found" in error_str or "invalid model" in error_str:
                raise BackendError(
                    f"Invalid model '{self._model}' for Codex backend",
                    backend="codex",
                    cause=e,
                ) from e
            logger.error("codex_tool_call_failed", error=str(e), tool=tool_name)
            raise BackendError(
                f"Codex tool call failed: {e}",
                backend="codex",
                cause=e,
            ) from e

    def _parse_codex_response(self, result: Any) -> list[BackendMessage]:
        """Convert Codex MCP response to BackendMessages.

        Args:
            result: Response from MCPServerStdio.call_tool()

        Returns:
            List of BackendMessages converted from the response.
        """
        messages: list[BackendMessage] = []

        # Handle different response formats
        if isinstance(result, dict):
            response_data = result
        elif hasattr(result, "content"):
            # MCP CallToolResult format
            response_data = self._extract_response_data(result)
        else:
            # Try to parse as JSON string
            try:
                response_data = json.loads(str(result)) if result else {}
            except (json.JSONDecodeError, TypeError):
                response_data = {"text": str(result) if result else ""}

        # Extract conversation ID for session continuity
        if "conversationId" in response_data:
            self._conversation_id = response_data["conversationId"]
            logger.debug("conversation_id_captured", id=self._conversation_id)

        # Parse content array (Codex response format)
        content = response_data.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        text = item.get("text", "")
                        if text:
                            # Detect error JSON responses from codex
                            self._check_error_response(text)
                            messages.append(TextMessage(content=text))
                            self._output_tokens += self._count_tokens(text)
                    elif item_type == "tool_use":
                        tool_name = item.get("name", "unknown")
                        tool_args = item.get("input", {})

                        # Check for refusal tool call (using shared helper)
                        refusal = parse_refusal_from_tool_call(tool_name, tool_args)
                        if refusal is not None:
                            # Log the tool call first (shows in display with 🔧)
                            messages.append(ToolCallMessage(tool_name=tool_name, args=tool_args))
                            self._capture_refusal(refusal.reason, refusal.category)
                            messages.append(refusal)
                            logger.info("refusal_captured", reason=refusal.reason, category=refusal.category)
                            continue

                        messages.append(ToolCallMessage(tool_name=tool_name, args=tool_args))
                        # Track file modifications from tool calls
                        self._track_tool_file_modification(tool_name, tool_args)
                    elif item_type == "tool_result":
                        tool_name = item.get("name", "unknown")
                        result_content = item.get("content", "")
                        success = not item.get("is_error", False)
                        messages.append(
                            ToolResult(
                                tool_name=tool_name,
                                output=str(result_content),
                                success=success,
                            )
                        )
                elif isinstance(item, str):
                    messages.append(TextMessage(content=item))
                    self._output_tokens += self._count_tokens(item)
        elif isinstance(content, str) and content:
            messages.append(TextMessage(content=content))
            self._output_tokens += self._count_tokens(content)

        # Handle direct text response (avoid duplicates)
        if (text := response_data.get("text")) and not any(
            isinstance(m, TextMessage) and m.content == text for m in messages
        ):
            messages.append(TextMessage(content=text))
            self._output_tokens += self._count_tokens(text)

        # Extract usage information if available
        usage = response_data.get("usage", {})
        if usage:
            self._input_tokens += int(usage.get("input_tokens", 0) or 0)
            self._output_tokens += int(usage.get("output_tokens", 0) or 0)
            self._cost_usd += float(usage.get("cost_usd", 0.0) or 0.0)

        return messages

    def _extract_response_data(self, result: Any) -> dict[str, Any]:
        """Extract response data from MCP CallToolResult.

        Args:
            result: MCP tool result object.

        Returns:
            Extracted response data as dict.
        """
        data: dict[str, Any] = {}

        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, list):
                data["content"] = []
                for item in content:
                    if hasattr(item, "text"):
                        data["content"].append({"type": "text", "text": item.text})
                    elif isinstance(item, dict):
                        data["content"].append(item)
                    else:
                        data["content"].append({"type": "text", "text": str(item)})
            elif isinstance(content, str):
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    data["text"] = content

        return data

    def _check_error_response(self, text: str) -> None:
        """Raise BackendError if the text is a codex error JSON response."""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(data, dict) and data.get("type") == "error":
            error = data.get("error", {})
            msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise BackendError(f"Codex API error: {msg}", backend="codex")

    def _track_tool_file_modification(self, tool_name: str, args: dict[str, Any]) -> None:
        """Track file modifications from tool calls.

        Args:
            tool_name: Name of the tool being called.
            args: Tool arguments.
        """
        if tool_name in FILE_MODIFICATION_TOOLS:
            for key in FILE_PATH_ARG_KEYS:
                if key in args:
                    path = args[key]
                    if path:
                        self._modified_files.add(str(path))
                        break

    def _parse_codex_event(self, params: dict[str, Any]) -> Iterator[BackendMessage]:
        """Parse a codex/event notification into BackendMessages.

        Args:
            params: The params dict from the codex/event notification.

        Yields:
            BackendMessage objects extracted from the event.
        """
        msg = params.get("msg", {})
        if not isinstance(msg, dict):
            return

        msg_type = msg.get("type", "")

        if msg_type == "task_complete":
            # Final agent response
            if text := msg.get("last_agent_message"):
                yield TextMessage(content=str(text))

        elif msg_type in ("tool_call", "mcp_tool_call"):
            # Tool invocation
            tool_name = msg.get("tool") or msg.get("name", "unknown")
            tool_args = msg.get("input", {})

            # Check for refusal
            refusal = parse_refusal_from_tool_call(tool_name, tool_args)
            if refusal is not None:
                yield ToolCallMessage(tool_name=tool_name, args=tool_args)
                self._capture_refusal(refusal.reason, refusal.category)
                yield refusal
                logger.info("refusal_captured", reason=refusal.reason, category=refusal.category)
                return

            yield ToolCallMessage(tool_name=tool_name, args=tool_args)
            self._track_tool_file_modification(tool_name, tool_args)

        elif msg_type == "tool_result":
            # Tool execution result
            tool_name = msg.get("tool") or msg.get("name", "unknown")
            output = msg.get("output", "")
            success = not msg.get("is_error", False)
            yield ToolResult(tool_name=tool_name, output=str(output), success=success)

    def modified_files(self) -> PathsModifiedFiles:
        """Return files modified during execution.

        Combines event-parsed paths with filesystem diff (new/changed files
        since connect()) to catch files the codex CLI wrote without emitting
        tracked tool-call events.
        """
        paths = set(self._modified_files)
        if self._cwd:
            for f in self._cwd.rglob("*"):
                if not f.is_file():
                    continue
                rel = str(f.relative_to(self._cwd))
                fstr = str(f)
                if fstr not in self._pre_existing_files or f.stat().st_mtime > self._pre_existing_files[fstr]:
                    paths.add(rel)
        return PathsModifiedFiles(file_paths=list(paths))

    def usage(self) -> Usage:
        """Return cumulative token usage and cost across all executions.

        Note: Codex may not provide detailed usage information in all cases.
        Values may be estimates or zeros if not available from the response.
        """
        # Calculate cost from tokens if not provided in response
        cost = self._cost_usd or calculate_cost_usd(self._model, self._input_tokens, self._output_tokens)
        return Usage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=cost,
            model=self._model,
            provider="codex",
        )

    # structured_output() is inherited from BaseBackend

    async def disconnect(self) -> None:
        """Clean up resources and reset connection state.

        After disconnect(), the backend is in the same state as a newly
        constructed instance. Call connect() again before using execute().
        """
        if self._mcp_server is not None:
            try:
                # Use wait_for with timeout to prevent hanging during cleanup
                # Shield from cancellation to ensure cleanup completes
                # Also suppress RuntimeError from anyio task context mismatches
                # (common when running in pytest-asyncio)
                await asyncio.shield(asyncio.wait_for(self._mcp_server.cleanup(), timeout=5.0))
            except TimeoutError:
                logger.warning("mcp_server_cleanup_timeout")
            except asyncio.CancelledError:
                # CancelledError from anyio cancel scope doesn't affect functionality
                logger.debug("mcp_cleanup_cancelled")
            except RuntimeError as e:
                # Suppress "cancel scope in different task" errors from anyio
                # These don't affect functionality, just cleanup ordering
                if "cancel scope" in str(e).lower():
                    logger.debug("mcp_cleanup_anyio_context_error", error=str(e))
                else:
                    logger.warning("mcp_server_cleanup_error", error=str(e))
            except Exception as e:
                logger.warning("mcp_server_cleanup_error", error=str(e))
            self._mcp_server = None

        # Clean up temp pickle file for custom tools
        if self._tool_pickle_path is not None:
            try:
                if self._tool_pickle_path.exists():
                    self._tool_pickle_path.unlink()
            except Exception as e:
                logger.warning("tool_pickle_cleanup_error", error=str(e))
            self._tool_pickle_path = None

        # Reset connection state
        self._cwd = None
        self._conversation_id = None
        self._custom_system_prompt = None
        self._modified_files = set()

        # Reset common state (tokens, structured output, refusal)
        self._reset_state()

        logger.debug("codex_backend_disconnected")
