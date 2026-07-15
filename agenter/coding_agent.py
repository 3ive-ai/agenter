"""AutonomousCodingAgent - the public facade for Agenter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from .coding_backends.anthropic_sdk import AnthropicSDKBackend
from .coding_backends.claude_code import ClaudeCodeBackend
from .config import (
    BACKEND_ACP,
    BACKEND_ANTHROPIC_SDK,
    BACKEND_CLAUDE_CODE,
    BACKEND_CODEX,
    BACKEND_OPENHANDS,
    default_backend,
)
from .data_models import Budget, CodingEvent, CodingRequest, CodingResult, Verbosity
from .logging import configure_logging
from .post_validators.syntax import SyntaxValidator
from .runtime import CodingSession, ConsoleDisplay, PersistentCodingSession, Tracer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from .coding_backends.codex import CodexBackend, CodexMCPServer
    from .post_validators.protocol import Validator
    from .tools import Tool

logger = structlog.get_logger(__name__)


class AutonomousCodingAgent:
    """Public interface for Agenter.

    This is the main entry point. It creates backends and validators,
    then runs a CodingSession to completion.

    All backends default to sandbox=True (safe mode).

    Example:
        # Default backend (anthropic-sdk) - sandbox enabled by default
        agent = AutonomousCodingAgent()
        result = await agent.execute(CodingRequest(...))

        # Use Claude Code backend (claude-code) - sandbox enabled by default
        agent = AutonomousCodingAgent(backend="claude-code")
        result = await agent.execute(CodingRequest(...))

        # Disable sandbox for full filesystem access
        agent = AutonomousCodingAgent(backend="claude-code", sandbox=False)

        # With custom tools (works with all backends)
        from agenter import tool

        @tool("search", "Search the web", {"query": str})
        async def search(args):
            return f"Results for {args['query']}"

        agent = AutonomousCodingAgent(tools=[search])
    """

    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        tools: list[Tool] | None = None,
        validators: Sequence[Validator] | None = None,
        use_anthropic_tools: bool = False,
        sandbox: bool = True,
        setting_sources: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        tracer: Tracer | None = None,
        # Codex-specific options
        codex_approval_policy: str = "never",
        codex_mcp_servers: list[CodexMCPServer] | None = None,
        codex_reasoning_effort: str | None = None,
        # Claude Code-specific options
        claude_max_thinking_tokens: int | None = None,
        # ACP-specific options
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_env: dict[str, str] | None = None,
        acp_mcp_servers: list[Any] | None = None,
        acp_permission_policy: str = "deny",
        acp_autonomous: bool = True,
        acp_update_callback: Callable[[Any], None] | None = None,
    ):
        """Initialize the agent.

        Args:
            backend: Backend to use. If None, uses ACA_DEFAULT_BACKEND env var
                (falls back to "anthropic-sdk"). Options:
                - "anthropic-sdk": Anthropic SDK with custom tools
                - "claude-code": Claude Code SDK with native sandbox
                - "codex": OpenAI Codex CLI via MCP server
                - "openhands": OpenHands SDK (requires sandbox=False)
                - "acp": Agent Client Protocol subprocess backend
            model: Model to use. Used by "anthropic-sdk", "claude-code", and "codex".
                If None, each backend uses its own default.
            tools: Additional custom tools. Works with all backends.
            validators: Validators to run on generated code. Defaults to [SyntaxValidator()].
            use_anthropic_tools: Use Anthropic's built-in text_editor_20250728 tool
                instead of our custom file tools. Only for "anthropic-sdk" backend.
            sandbox: Enable sandboxed execution (default True). When True, each
                backend runs in its safest mode:
                - anthropic-sdk: Enforces allowed_write_paths within cwd
                - claude-code: Uses Claude Code's native OS-level sandbox
                - codex: Uses "workspace-write" mode
                When False, backends run with full filesystem access.
            setting_sources: Sources for loading settings (claude-code only).
                e.g., ["project", "user"] to load .claude/skills from project/user dirs.
            allowed_tools: List of tools to allow (claude-code only).
                Defaults to ["Read", "Edit", "Write", "Bash", "Glob"].
            tracer: Optional tracer for recording agent interactions. Use FileTracer
                to save traces to files, or implement the Tracer protocol for custom
                tracing (e.g., to a logging service or database).
            codex_approval_policy: Approval policy for Codex tool execution (codex only).
                Options: "untrusted", "on-request", "on-failure", "never".
                Defaults to "never" for autonomous operation.
            codex_mcp_servers: Custom MCP servers to pass to Codex (codex only).
                Gives Codex access to additional tools during execution.
            codex_reasoning_effort: Optional Codex/OpenAI reasoning effort (codex only).
                Valid values: "minimal", "low", "medium", "high".
            claude_max_thinking_tokens: Optional thinking budget for Claude Code (claude-code only).
            acp_command: Executable for the ACP agent process (acp only).
            acp_args: Arguments passed to the ACP agent process (acp only).
            acp_env: Extra environment variables for the ACP agent process (acp only).
            acp_mcp_servers: MCP server descriptors passed to ACP session/new (acp only).
            acp_permission_policy: How Agenter answers ACP permission requests.
                Options: "deny" or "allow". Defaults to "deny".
            acp_autonomous: Add Agenter's autonomous backend contract to ACP prompts
                and auto-continue once when an ACP agent asks for confirmation.
                Defaults to True.
            acp_update_callback: Optional synchronous callback invoked for each raw ACP
                session update as it arrives (acp only).
        """
        if backend is None:
            backend = default_backend()
        logger.info("agent_init", backend=backend)
        if backend not in (BACKEND_ANTHROPIC_SDK, BACKEND_CLAUDE_CODE, BACKEND_CODEX, BACKEND_OPENHANDS, BACKEND_ACP):
            from .data_models import ConfigurationError

            raise ConfigurationError(
                f"Unknown backend: {backend!r}. "
                f"Use {BACKEND_ANTHROPIC_SDK!r}, {BACKEND_CLAUDE_CODE!r}, {BACKEND_CODEX!r}, "
                f"{BACKEND_OPENHANDS!r}, or {BACKEND_ACP!r}."
            )
        self._backend_type = backend
        self.model = model  # Let backend use its own default if None
        self._extra_tools = tools
        self._validators: Sequence[Validator] = validators if validators is not None else [SyntaxValidator()]
        self._use_anthropic_tools = use_anthropic_tools
        self._sandbox = sandbox
        self._setting_sources = setting_sources
        self._allowed_tools = allowed_tools
        self._tracer = tracer

        # Codex-specific options
        self._codex_approval_policy = codex_approval_policy
        self._codex_mcp_servers = codex_mcp_servers
        self._codex_reasoning_effort = codex_reasoning_effort

        # Claude Code-specific options
        self._claude_max_thinking_tokens = claude_max_thinking_tokens

        # ACP-specific options
        self._acp_command = acp_command
        self._acp_args = acp_args
        self._acp_env = acp_env
        self._acp_mcp_servers = acp_mcp_servers
        self._acp_permission_policy = acp_permission_policy
        self._acp_autonomous = acp_autonomous
        self._acp_update_callback = acp_update_callback

        if backend == BACKEND_CLAUDE_CODE and use_anthropic_tools:
            logger.warning(
                "use_anthropic_tools is ignored with claude-code backend. "
                "claude-code-sdk uses its own built-in file tools."
            )

        if backend == BACKEND_ANTHROPIC_SDK and (setting_sources or allowed_tools):
            logger.warning(
                "setting_sources and allowed_tools are ignored with anthropic-sdk backend. "
                "Use claude-code backend for these features."
            )

        if backend == BACKEND_CODEX and (use_anthropic_tools or setting_sources or allowed_tools):
            logger.warning("use_anthropic_tools, setting_sources, and allowed_tools are ignored with codex backend.")

        codex_opts_set = codex_approval_policy != "never" or codex_mcp_servers or codex_reasoning_effort
        if backend != BACKEND_CODEX and codex_opts_set:
            logger.warning(
                "codex_approval_policy, codex_mcp_servers, and codex_reasoning_effort are only used with codex backend."
            )

        if backend != BACKEND_CLAUDE_CODE and claude_max_thinking_tokens is not None:
            logger.warning("claude_max_thinking_tokens is only used with claude-code backend.")

        acp_opts_set = (
            acp_command
            or acp_args
            or acp_env
            or acp_mcp_servers
            or acp_permission_policy != "deny"
            or not acp_autonomous
            or acp_update_callback is not None
        )
        if backend != BACKEND_ACP and acp_opts_set:
            logger.warning(
                "acp_command, acp_args, acp_env, acp_mcp_servers, acp_permission_policy, "
                "acp_autonomous, and acp_update_callback are only used with acp backend."
            )

        if backend == BACKEND_OPENHANDS:
            if sandbox:
                from .data_models import ConfigurationError

                raise ConfigurationError(
                    "OpenHands backend requires sandbox=False. OpenHands SDK does not support sandboxed execution.",
                    parameter="sandbox",
                    value=str(sandbox),
                )
            if use_anthropic_tools or setting_sources or allowed_tools:
                logger.warning(
                    "use_anthropic_tools, setting_sources, and allowed_tools are ignored with openhands backend."
                )
        if backend == BACKEND_ACP and not acp_command:
            from .data_models import ConfigurationError

            raise ConfigurationError(
                "acp_command is required when backend='acp'.",
                parameter="acp_command",
                value=None,
            )

    def _setup_session(
        self,
        verbosity: Verbosity,
        log_dir: str | Path | None,
    ) -> CodingSession:
        """Create display and session for execution.

        Args:
            verbosity: Output verbosity level.
            log_dir: Optional path for logging.

        Returns:
            Configured CodingSession ready to run.
        """
        # Suppress structlog output when QUIET, even if a caller previously
        # configured Agenter logging at DEBUG.
        if verbosity == Verbosity.QUIET:
            configure_logging(quiet=True)

        display = None
        if verbosity != Verbosity.QUIET or log_dir:
            display = ConsoleDisplay(
                verbosity=verbosity,
                output_dir=Path(log_dir) if log_dir else None,
            )

        backend = self._create_backend()
        return CodingSession(backend, self._validators, display=display, tracer=self._tracer)

    async def open_session(
        self,
        cwd: str,
        *,
        allowed_write_paths: list[str] | None = None,
        system_prompt: str | None = None,
        session_budget: Budget | None = None,
        resume_session_id: str | None = None,
        verbosity: Verbosity = Verbosity.QUIET,
        log_dir: str | Path | None = None,
    ) -> PersistentCodingSession:
        """Open a persistent ACP session for multi-turn coding follow-ups.

        The returned object owns one ACP subprocess and serializes each call to
        ``execute`` or ``stream_execute`` onto the same remote session until
        ``close`` is called. Use it as an async context manager when possible.

        Args:
            cwd: Fixed working directory for every follow-up.
            allowed_write_paths: Optional write restrictions fixed for the session.
            system_prompt: Optional instructions fixed for the session.
            session_budget: Optional cumulative budget across all follow-ups.
            resume_session_id: Existing ACP session ID to resume or load. The
                ACP agent must advertise the corresponding capability.
            verbosity: Output verbosity level.
            log_dir: Optional path to save prompt/response logs.
        """
        if self._backend_type != BACKEND_ACP:
            from .data_models import ConfigurationError

            raise ConfigurationError(
                "open_session() currently supports only backend='acp'.",
                parameter="backend",
                value=self._backend_type,
            )

        coding_session = self._setup_session(verbosity, log_dir)
        base_request = CodingRequest(
            prompt="",
            cwd=cwd,
            allowed_write_paths=allowed_write_paths,
            system_prompt=system_prompt,
        )
        persistent = PersistentCodingSession(
            coding_session,
            base_request,
            session_budget=session_budget,
            resume_session_id=resume_session_id,
        )
        return await persistent.start()

    async def execute(
        self,
        request: CodingRequest,
        verbosity: Verbosity = Verbosity.QUIET,
        log_dir: str | Path | None = None,
        raise_on_budget_exceeded: bool = False,
    ) -> CodingResult:
        """Execute a coding task.

        Args:
            request: The coding task request
            verbosity: Output verbosity level (QUIET, NORMAL, or VERBOSE)
            log_dir: Optional path to save full prompts/responses for debugging
            raise_on_budget_exceeded: If True, raise BudgetExceededError when budget
                is exceeded. If False (default), return CodingResult with
                status=BUDGET_EXCEEDED and populated exceeded_limit/exceeded_values.

        Returns:
            CodingResult with status and modified files
        """
        logger.debug(
            "starting_execution",
            model=self.model,
            cwd=str(request.cwd),
        )

        session = self._setup_session(verbosity, log_dir)
        result = await session.run(request, raise_on_budget_exceeded)

        logger.debug(
            "execution_completed",
            status=result.status.value,
            iterations=result.iterations,
            total_tokens=result.total_tokens,
        )
        return result

    async def stream_execute(
        self,
        request: CodingRequest,
        verbosity: Verbosity = Verbosity.QUIET,
        log_dir: str | Path | None = None,
    ) -> AsyncIterator[CodingEvent]:
        """Execute a coding task, streaming events.

        Args:
            request: The coding task request
            verbosity: Output verbosity level (QUIET, NORMAL, or VERBOSE)
            log_dir: Optional path to save full prompts/responses for debugging

        Yields:
            CodingEvent for each significant step during execution
        """
        session = self._setup_session(verbosity, log_dir)
        async for event in session.stream_run(request):
            yield event

    def _create_backend(self) -> AnthropicSDKBackend | ClaudeCodeBackend | CodexBackend | Any:
        """Create the appropriate backend based on configuration."""
        if self._backend_type == BACKEND_CODEX:
            from .coding_backends.codex import CodexBackend

            # Map unified sandbox to codex sandbox mode
            codex_sandbox_mode = "workspace-write" if self._sandbox else "danger-full-access"

            return CodexBackend(
                model=self.model,
                reasoning_effort=self._codex_reasoning_effort,
                approval_policy=self._codex_approval_policy,
                sandbox=codex_sandbox_mode,
                mcp_servers=self._codex_mcp_servers,
                extra_tools=self._extra_tools,
            )
        elif self._backend_type == BACKEND_CLAUDE_CODE:
            return ClaudeCodeBackend(
                model=self.model,
                max_thinking_tokens=self._claude_max_thinking_tokens,
                sandbox=self._sandbox,
                allowed_tools=self._allowed_tools,
                setting_sources=self._setting_sources,
                extra_tools=self._extra_tools,
            )
        elif self._backend_type == BACKEND_ACP:
            from .coding_backends.acp import ACPBackend

            return ACPBackend(
                command=self._acp_command or "",
                args=self._acp_args,
                env=self._acp_env,
                mcp_servers=self._acp_mcp_servers,
                sandbox=self._sandbox,
                permission_policy=self._acp_permission_policy,
                autonomous=self._acp_autonomous,
                update_callback=self._acp_update_callback,
            )
        elif self._backend_type == BACKEND_OPENHANDS:
            from .coding_backends.openhands import OpenHandsBackend

            return OpenHandsBackend(
                sandbox=False,  # OpenHands requires sandbox=False
                extra_tools=self._extra_tools,
            )
        else:
            return AnthropicSDKBackend(
                model=self.model,
                extra_tools=self._extra_tools,
                use_anthropic_tools=self._use_anthropic_tools,
                sandbox=self._sandbox,
            )
