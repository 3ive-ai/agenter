"""Claude Code backend using claude-code-sdk.

This backend wraps claude-code-sdk (Claude Code as a library) to provide
battle-tested tools and agent loop maintained by Anthropic.

SECURITY MODES
==============
This backend supports two security modes via the `sandbox` parameter:

1. SANDBOX MODE (recommended, default):
   - Uses Claude Code's native OS-level sandboxing
   - Restricts filesystem and network access
   - Set sandbox=True or pass a sandbox config dict

2. UNRESTRICTED MODE:
   - No path restrictions - files can be read/written ANYWHERE
   - The agent runs with bypassPermissions mode
   - Set sandbox=False to enable this mode

Use sandbox=True for production environments.

Requires the claude-code optional dependency:
    pip install agenter[claude-code]

Supports:
    - Direct Anthropic API (ANTHROPIC_API_KEY)
    - AWS Bedrock (CLAUDE_CODE_USE_BEDROCK=1)
    - Google Vertex AI (CLAUDE_CODE_USE_VERTEX=1)

Features:
    - Native sandbox support via Claude Code's sandbox infrastructure
    - Setting sources for loading .claude/skills, MCP servers, etc.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

try:
    from claude_agent_sdk import ClaudeSDKError
except ImportError:
    # Placeholder when claude-agent-sdk is not installed (for testing without claude-code extra)
    class ClaudeSDKError(Exception):  # type: ignore[no-redef]
        """Placeholder when claude-agent-sdk is not installed."""

        pass


from ...data_models import (
    BackendError,
    BackendMessage,
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
from ..prompts import CLAUDE_CODE_PROMPT
from ..refusal import (
    REFUSAL_INSTRUCTIONS,
    REFUSAL_TOOL,
    parse_refusal_from_tool_call,
)
from .constants import (
    FILE_MODIFICATION_TOOLS,
    PATH_INPUT_KEYS,
    SDK_STRUCTURED_OUTPUT_KEY,
    SDK_STRUCTURED_OUTPUT_TOOL,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic import BaseModel

    from ...tools import Tool

logger = structlog.get_logger(__name__)


def _is_sdk_error(e: BaseException) -> bool:
    """Type guard for ClaudeSDKError in exception groups."""
    return isinstance(e, ClaudeSDKError)


class ClaudeCodeBackend(BaseBackend):
    """Backend using claude-agent-sdk (Claude Code as library).

    This uses Anthropic's official agent SDK which includes:
    - Built-in file tools (Read, Edit, Write, Bash, Glob, etc.)
    - Battle-tested agent loop
    - Native OS-level sandboxing
    - Bedrock/Vertex/Foundry support via environment variables

    Example:
        # Safe mode with sandbox (recommended)
        backend = ClaudeCodeBackend(sandbox=True)
        await backend.connect("/path/to/project")
        async for message in backend.execute("Fix the bug"):
            print(message)

        # Custom sandbox config
        backend = ClaudeCodeBackend(sandbox={
            "enabled": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
        })

        # No sandbox (for trusted local development)
        backend = ClaudeCodeBackend(sandbox=False)
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        max_thinking_tokens: int | None = None,
        sandbox: bool | dict = True,
        allowed_tools: list[str] | None = None,
        setting_sources: list[str] | None = None,
        extra_tools: list[Tool] | None = None,
    ) -> None:
        """Initialize the backend.

        Args:
            model: Optional Claude model to pass to Claude Code.
            max_thinking_tokens: Optional thinking budget for Claude Code.
            sandbox: Enable Claude Code's native sandbox. Can be:
                - True: Use default safe sandbox config
                - dict: Custom sandbox config with keys like:
                    - enabled: bool
                    - autoAllowBashIfSandboxed: bool (auto-allow bash in sandbox)
                    - allowUnsandboxedCommands: bool (allow escape requests)
                    - excludedCommands: list[str] (commands excluded from sandbox)
                    - network: dict with allowLocalBinding, allowUnixSockets
                - False: No sandbox (unrestricted filesystem access)
            allowed_tools: List of tools to allow. Defaults to file tools.
                Available: Read, Edit, Write, Bash, Glob, Grep, etc.
            setting_sources: Sources for loading settings (e.g., ["project", "user"]).
                "project" loads .claude/skills, MCP servers from project directory.
                "user" loads from user's home .claude directory.
            extra_tools: Custom tools to add. These are converted to SDK MCP tools
                and registered as an in-process MCP server.
        """
        self._configured_model = model
        self._max_thinking_tokens = max_thinking_tokens
        self._sandbox = sandbox

        self._allowed_tools = allowed_tools or ["Read", "Edit", "Write", "Bash", "Glob"]
        self._setting_sources = setting_sources
        self._extra_tools = extra_tools or []

        # Initialize common state (tokens, structured output, refusal)
        self._init_state()

        # Connection state
        self._cwd: Path | None = None
        self._files_modified: dict[str, str] = {}
        self._tool_use_id_to_name: dict[str, str] = {}
        self._model: str | None = None
        self._last_text_content: str = ""

        # SDK client (for proper lifecycle management)
        self._client: Any = None

        # Cached SDK types for isinstance checks (None = use string matching)
        self._sdk_types: dict[str, type] | None = None
        self._sdk_types_loaded: bool = False

    async def connect(
        self,
        cwd: str,
        allowed_write_paths: list[str] | None = None,
        resume_session_id: str | None = None,
        output_type: type[BaseModel] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize with a working directory.

        Args:
            cwd: Working directory for file operations.
            allowed_write_paths: Optional glob patterns restricting writes.
                WARNING: These are NOT enforced by claude-agent-sdk.
                The agent can still write anywhere.
            resume_session_id: Optional session ID to resume a previous session.
                NOTE: Not currently supported by claude-agent-sdk.
            output_type: Optional Pydantic model for structured output.
            system_prompt: Custom system prompt. If None, uses cwd context.
        """
        self._custom_system_prompt = system_prompt
        if resume_session_id:
            logger.warning(
                "resume_session_id_not_supported",
                backend="claude-code",
                session_id=resume_session_id,
            )
        self._cwd = Path(cwd)
        self._files_modified = {}
        self._tool_use_id_to_name = {}
        self._model = None
        self._last_text_content = ""

        # Reset common state and set output_type
        self._reset_state()
        self._output_type = output_type

        if output_type:
            logger.warning(
                "structured_output_sdk_workaround",
                message="Using workaround for SDK Issue #105: intercepting StructuredOutput tool calls. "
                "See https://github.com/anthropics/claude-agent-sdk-typescript/issues/105",
                output_type=output_type.__name__,
            )

        # Log info about allowed_write_paths
        if allowed_write_paths:
            if self._sandbox:
                logger.info(
                    "allowed_write_paths is set; sandbox mode enforces path restrictions "
                    "via Claude Code's native sandboxing.",
                )
            else:
                logger.warning(
                    "allowed_write_paths is set but NOT enforced when sandbox=False. "
                    "The agent can write files anywhere. Use sandbox=True for restrictions.",
                )

    async def execute(self, prompt: str) -> AsyncIterator[BackendMessage]:
        """Execute a prompt using claude-agent-sdk.

        Args:
            prompt: The user prompt to execute.

        Yields:
            BackendMessage objects for each significant step.
        """
        try:
            from claude_agent_sdk import ClaudeAgentOptions
            from claude_agent_sdk import query as sdk_query
        except ImportError as e:
            raise BackendError(
                "claude-agent-sdk is required for ClaudeCodeBackend. Install with: pip install agenter[claude-code]",
                backend="claude-code",
                cause=e,
            ) from e

        if self._cwd is None:
            raise BackendError(
                "Backend not connected. Call connect() first.",
                backend="claude-code",
            )

        # Build options with all configured settings
        options_kwargs: dict[str, Any] = {
            "cwd": str(self._cwd),
            "allowed_tools": self._allowed_tools,
        }
        if self._configured_model:
            options_kwargs["model"] = self._configured_model
        if self._max_thinking_tokens is not None:
            options_kwargs["max_thinking_tokens"] = self._max_thinking_tokens

        # Configure sandbox or bypass permissions
        if self._sandbox:
            if isinstance(self._sandbox, dict):
                options_kwargs["sandbox"] = self._sandbox
            else:
                # Default safe sandbox config
                options_kwargs["sandbox"] = {
                    "enabled": True,
                    "autoAllowBashIfSandboxed": True,
                    "allowUnsandboxedCommands": False,
                }
            options_kwargs["permission_mode"] = "default"
        else:
            options_kwargs["permission_mode"] = "bypassPermissions"

        # Add setting sources if configured
        if self._setting_sources:
            options_kwargs["setting_sources"] = self._setting_sources

        # Add output_format for structured output
        if self._output_type is not None:
            options_kwargs["output_format"] = {
                "type": "json_schema",
                "schema": self._output_type.model_json_schema(),
            }

        # Build MCP servers dict
        mcp_servers: dict[str, Any] = {}
        extra_allowed_tools: list[str] = []

        # Create MCP server with refusal tool + custom tools
        all_tools = [REFUSAL_TOOL, *self._extra_tools]
        tools_server = self._create_mcp_server_for_tools()
        if tools_server:
            mcp_servers["agenter-tools"] = tools_server
            extra_allowed_tools.extend(f"mcp__agenter-tools__{t.name}" for t in all_tools)

        if mcp_servers:
            options_kwargs["mcp_servers"] = mcp_servers
            options_kwargs["allowed_tools"] = self._allowed_tools + extra_allowed_tools

        options = ClaudeAgentOptions(**options_kwargs)

        logger.debug(
            "executing_with_claude_agent_sdk",
            cwd=str(self._cwd),
            sandbox_enabled=bool(self._sandbox),
            permission_mode=options_kwargs.get("permission_mode"),
        )

        # Build system context: cwd info + optional custom instructions
        default_context = CLAUDE_CODE_PROMPT.format(cwd=self._cwd)
        if self._custom_system_prompt:
            system_context = f"{default_context}\n\n{self._custom_system_prompt}"
        else:
            system_context = default_context
        full_prompt = f"{system_context}\n\n{prompt}"

        # Add structured output instructions if output_type is configured
        # The SDK adds a StructuredOutput tool but doesn't force the model to use it
        # The tool expects input wrapped in a "parameter" key
        if self._output_type is not None:
            schema_json = json.dumps(self._output_type.model_json_schema(), indent=2)
            structured_instructions = (
                f"\n\n## REQUIRED: CALL StructuredOutput TOOL\n"
                f"When you have completed the task, you MUST call the `StructuredOutput` tool.\n"
                f"The tool input must wrap your response in a 'parameter' key:\n"
                f'```json\n{{\n  "parameter": <your response matching schema below>\n}}\n```\n'
                f"Schema for your response:\n"
                f"```json\n{schema_json}\n```\n"
                f"Do NOT just return JSON as text - you MUST use the StructuredOutput tool."
            )
            full_prompt = full_prompt + structured_instructions

        # Always add refusal instructions so LLM knows how to signal refusal
        full_prompt = full_prompt + f"\n\n{REFUSAL_INSTRUCTIONS}"

        # Yield PromptMessage for tracing (like AnthropicSDKBackend does)
        yield PromptMessage(user_prompt=prompt, system_prompt=system_context)

        # SDK bug #386: MCP servers only work with async generator prompts, not strings.
        # When MCP servers are present, wrap the prompt as an async generator.
        # See: https://github.com/anthropics/claude-agent-sdk-typescript/issues/386
        prompt_source: Any  # str or AsyncIterator - SDK accepts both
        if mcp_servers:
            prompt_source = self._wrap_prompt_as_generator(full_prompt)
            logger.debug("using_async_generator_prompt", reason="mcp_servers_present")
        else:
            prompt_source = full_prompt

        # Use the SDK's one-shot query() helper for a single prompt/response.
        # This runs Claude Code in non-interactive (--print) mode for string prompts,
        # which avoids lingering sessions and is typically more robust under test runners.
        try:
            async for message in sdk_query(prompt=prompt_source, options=options):
                for converted in self._convert_message(message):
                    yield converted
                # Stop SDK execution immediately when refusal is detected
                if self._refusal is not None:
                    logger.info("stopping_sdk_execution_after_refusal")
                    break
        except BaseExceptionGroup as eg:
            # Handle SDK errors (subprocess crashes, connection issues, etc.)
            sdk_errors, other_errors = eg.split(_is_sdk_error)
            if sdk_errors:
                logger.warning(
                    "sdk_error",
                    message="SDK error during execution",
                    error_count=len(sdk_errors.exceptions),
                )
            if other_errors:
                raise other_errors from None
        except ClaudeSDKError as e:
            # Catch SDK errors: ProcessError, CLIConnectionError, etc.
            logger.warning("sdk_error", error=str(e))

        # Fallback: parse structured output from last text response if not captured
        if self._output_type is not None and self._structured_output is None:
            self._structured_output = parse_structured_output(self._last_text_content, self._output_type)

    async def _wrap_prompt_as_generator(self, text: str) -> AsyncIterator[dict[str, Any]]:
        """Wrap a string prompt as an async generator for SDK compatibility.

        SDK bug #386: MCP servers only work with async generator prompts.
        This wraps a string prompt in the required format.

        Args:
            text: The prompt text to wrap.

        Yields:
            User message dict in SDK's expected format.
        """
        yield {"type": "user", "message": {"role": "user", "content": text}}

    def _get_sdk_types(self) -> dict[str, type] | None:
        """Get SDK types for isinstance checks, or None to use string matching.

        Results are cached after first call to avoid repeated import attempts.
        """
        if self._sdk_types_loaded:
            return self._sdk_types

        try:
            from claude_agent_sdk.types import (
                AssistantMessage,
                ResultMessage,
                TextBlock,
                ToolResultBlock,
                ToolUseBlock,
            )

            self._sdk_types = {
                "AssistantMessage": AssistantMessage,
                "ResultMessage": ResultMessage,
                "TextBlock": TextBlock,
                "ToolResultBlock": ToolResultBlock,
                "ToolUseBlock": ToolUseBlock,
            }
        except ImportError:
            self._sdk_types = None
        finally:
            self._sdk_types_loaded = True

        return self._sdk_types

    def _convert_message(self, message: Any) -> list[BackendMessage]:
        """Convert claude-agent-sdk message to list of BackendMessages.

        Uses isinstance checks when SDK types are available, falls back to
        string-based type checking if imports fail.

        Args:
            message: Message from claude-agent-sdk.

        Returns:
            List of converted BackendMessages (may be empty).
        """
        type_map = self._get_sdk_types()

        def is_type(obj: Any, type_name: str) -> bool:
            if type_map:
                return isinstance(obj, type_map.get(type_name, type(None)))
            return type(obj).__name__ == type_name

        results: list[BackendMessage] = []

        # Check for structured_output attribute on any message
        # SDK populates this when output_format is configured
        if self._output_type is not None and hasattr(message, "structured_output"):
            structured_data = message.structured_output
            if structured_data is not None:
                try:
                    self._structured_output = self._output_type.model_validate(structured_data)
                    logger.debug("structured_output_captured", type=type(self._structured_output).__name__)
                except (ValueError, TypeError) as e:
                    # Pydantic ValidationError inherits from ValueError
                    logger.warning("structured_output_validation_failed", error=str(e))

        if is_type(message, "AssistantMessage"):
            # Text or tool use from assistant - process ALL blocks
            content = getattr(message, "content", [])
            for block in content:
                if is_type(block, "TextBlock"):
                    self._last_text_content = block.text
                    results.append(TextMessage(content=block.text))
                elif is_type(block, "ToolUseBlock"):
                    # Capture structured output from SDK's internal tool (Issue #105 workaround)
                    self._try_capture_structured_output(block.name, block.input)

                    # Capture refusal if LLM called the Refusal tool (using shared helper)
                    tool_args = dict(block.input) if hasattr(block.input, "items") else {}
                    refusal = parse_refusal_from_tool_call(block.name, tool_args)
                    if refusal is not None:
                        # Log the tool call first (shows in display with 🔧)
                        results.append(ToolCallMessage(tool_name=block.name, args=tool_args))
                        self._capture_refusal(refusal.reason, refusal.category)
                        results.append(refusal)
                        logger.info("refusal_captured", reason=refusal.reason, category=refusal.category)
                        continue

                    # Track tool_use_id -> tool_name for ToolResult blocks
                    if getattr(block, "id", None):
                        self._tool_use_id_to_name[block.id] = block.name

                    # Track file modifications using centralized constants
                    if block.name in FILE_MODIFICATION_TOOLS:
                        path = None
                        for key in PATH_INPUT_KEYS:
                            path = block.input.get(key)
                            if path:
                                break
                        if path and self._cwd:
                            abs_path = Path(path)
                            if abs_path.is_absolute():
                                with contextlib.suppress(ValueError):
                                    path = str(abs_path.relative_to(self._cwd))
                            self._files_modified[path] = ""  # Content tracked by SDK

                    results.append(
                        ToolCallMessage(
                            tool_name=block.name,
                            args=dict(block.input) if hasattr(block.input, "items") else {},
                        )
                    )

                elif is_type(block, "ToolResultBlock"):
                    tool_use_id = getattr(block, "tool_use_id", None)
                    if isinstance(tool_use_id, str):
                        tool_name = self._tool_use_id_to_name.get(tool_use_id, "unknown")
                    else:
                        tool_name = "unknown"
                    content_val: Any = getattr(block, "content", block.content if hasattr(block, "content") else None)
                    if isinstance(content_val, (dict, list)):
                        content_str = json.dumps(content_val)
                    else:
                        content_str = "" if content_val is None else str(content_val)
                    results.append(
                        ToolResult(
                            tool_name=tool_name,
                            output=content_str,
                            success=not bool(getattr(block, "is_error", False)),
                        )
                    )

        elif is_type(message, "ResultMessage"):
            # Final result - accumulate usage (not overwrite)
            usage = getattr(message, "usage", None)
            if isinstance(usage, dict):
                self._input_tokens += int(usage.get("input_tokens", 0) or 0)
                self._output_tokens += int(usage.get("output_tokens", 0) or 0)
                self._model = (usage.get("model") if isinstance(usage.get("model"), str) else None) or self._model
                cost = getattr(message, "total_cost_usd", None)
                if cost is None:
                    cost = usage.get("cost_usd")
                self._cost_usd += float(cost or 0.0)
            elif usage:
                # Backward/forward compat if SDK ever returns an object
                self._input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                self._output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
                cost_obj = getattr(usage, "cost_usd", None)
                self._cost_usd += float(cost_obj or getattr(message, "total_cost_usd", 0.0) or 0.0)
                self._model = getattr(usage, "model", None) or self._model

        return results

    def modified_files(self) -> PathsModifiedFiles:
        """Return files modified during execution.

        Note: claude-agent-sdk tracks files internally. We return paths only;
        content must be read from disk if needed.
        """
        return PathsModifiedFiles(file_paths=list(self._files_modified.keys()))

    def usage(self) -> Usage:
        """Return cumulative token usage and cost across all executions."""
        model = self._model or "unknown"
        cost = self._cost_usd or calculate_cost_usd(model, self._input_tokens, self._output_tokens)
        return Usage(
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=cost,
            model=model,
            provider="claude-code",
        )

    # structured_output() and refusal() are inherited from BaseBackend

    def _try_capture_structured_output(self, tool_name: str, tool_input: Any) -> None:
        """Capture structured output from SDK's StructuredOutput tool.

        Workaround for SDK Issue #105: output_format doesn't populate
        message.structured_output. We intercept the tool call directly.

        See: https://github.com/anthropics/claude-agent-sdk-typescript/issues/105
        """
        # Debug logging to see all tool calls when structured output is expected
        if self._output_type is not None:
            logger.debug(
                "tool_call_received",
                tool_name=tool_name,
                expected_tool=SDK_STRUCTURED_OUTPUT_TOOL,
                is_match=tool_name == SDK_STRUCTURED_OUTPUT_TOOL,
            )

        if tool_name != SDK_STRUCTURED_OUTPUT_TOOL or self._output_type is None:
            return

        try:
            data = tool_input
            # SDK wraps output in a wrapper key
            if isinstance(data, dict) and SDK_STRUCTURED_OUTPUT_KEY in data:
                data = data[SDK_STRUCTURED_OUTPUT_KEY]
            self._structured_output = self._output_type.model_validate(data)
            logger.debug("structured_output_captured", type=type(self._structured_output).__name__)
        except (ValueError, TypeError) as e:
            # Pydantic ValidationError inherits from ValueError
            logger.warning(
                "structured_output_validation_failed",
                error=str(e),
                tool=SDK_STRUCTURED_OUTPUT_TOOL,
            )

    def _create_mcp_server_for_tools(self) -> Any | None:
        """Create MCP server with refusal tool + custom tools.

        Always includes REFUSAL_TOOL, plus any user-provided extra_tools.
        Tool names become mcp__agenter-tools__<name> in allowed_tools.

        Returns:
            SDK MCP server object, or None if SDK not available.
        """
        try:
            from claude_agent_sdk import create_sdk_mcp_server
            from claude_agent_sdk import tool as sdk_tool
        except ImportError:
            logger.warning(
                "claude-agent-sdk not available for custom tools. Install with: pip install agenter[claude-code]"
            )
            return None

        # Combine refusal tool with user's custom tools
        all_tools = [REFUSAL_TOOL, *self._extra_tools]

        sdk_tools = []
        for t in all_tools:
            # Create wrapper that calls our Tool.execute()
            # Use default arg to capture 't' in closure
            @sdk_tool(t.name, t.description, t.input_schema)
            async def wrapper(args: dict, _tool: Any = t) -> dict:
                result = await _tool.execute(args)
                return {"content": [{"type": "text", "text": result.output}]}

            sdk_tools.append(wrapper)

        return create_sdk_mcp_server(
            name="agenter-tools",
            version="1.0.0",
            tools=sdk_tools,
        )

    async def disconnect(self) -> None:
        """Clean up resources and reset connection state.

        After disconnect(), the backend is in the same state as a newly
        constructed instance. Call connect() again before using execute().
        """
        # Disconnect SDK client if still active (cleans up subprocess)
        # Shield from cancellation to ensure cleanup completes
        if self._client is not None:
            with contextlib.suppress(Exception):
                await asyncio.shield(self._client.disconnect())
            self._client = None

        # Reset connection state
        self._cwd = None
        self._files_modified = {}
        self._model = None
        self._last_text_content = ""

        # Reset common state (tokens, structured output, refusal)
        self._reset_state()
