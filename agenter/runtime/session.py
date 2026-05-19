"""CodingSession - the core orchestration loop."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from ..coding_backends.retry import build_retry_prompt
from ..data_models import (
    BackendMessage,
    Budget,
    BudgetLimitType,
    CodingEvent,
    CodingEventType,
    CodingRequest,
    CodingResult,
    CodingStatus,
    IterationCompleted,
    IterationStarted,
    MessageReceived,
    ModifiedFiles,
    PathSecurityError,
    PromptMessage,
    RequestRefused,
    SessionEnded,
    SessionStarted,
    TaskCompleted,
    TaskFailed,
    TextMessage,
    ToolCallMessage,
    ToolResult,
    UsageDelta,
    ValidationCompleted,
    ValidationStarted,
)
from ..file_system import PathResolver
from ..post_validators import ValidatorChain
from .budget import BudgetMeter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from ..coding_backends.protocol import CodingBackend
    from ..post_validators.protocol import Validator
    from .display import ConsoleDisplay
    from .tracer import Tracer

logger = structlog.get_logger(__name__)

_IGNORED_PATH_ONLY_DIRS = frozenset({"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"})
_IGNORED_PATH_ONLY_SUFFIXES = frozenset({".pyc", ".pyo"})


class CodingSession:
    """Manages the iteration loop: execute -> validate -> retry.

    The session runs the backend, validates output, and retries with
    error context until validation passes or max iterations reached.
    """

    def __init__(
        self,
        backend: CodingBackend,
        validators: Sequence[Validator],
        display: ConsoleDisplay | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.backend = backend
        self._validator_chain = ValidatorChain(validators)
        self.display = display
        self.tracer = tracer

    @property
    def validators(self) -> Sequence[Validator]:
        """Access validators via chain for backwards compatibility."""
        return self._validator_chain.validators

    def _get_budget(self, request: CodingRequest) -> Budget:
        """Get budget from request, using max_iterations fallback."""
        if request.budget:
            return request.budget
        return Budget(max_iterations=request.max_iterations)

    def _prepare_files_for_validation(
        self,
        file_changes: ModifiedFiles,
        cwd: str,
    ) -> dict[str, str]:
        """Prepare files dict for validation, reading from disk if needed.

        When backends return ModifiedFiles(paths_only=True), the files dict
        contains empty strings. We read the actual content from disk to enable
        proper syntax validation.

        Security: All paths are validated through PathResolver to prevent
        directory traversal attacks (e.g., ../secret.txt or /etc/passwd).

        Args:
            file_changes: ModifiedFiles from backend.
            cwd: Working directory.

        Returns:
            Dict of path -> content suitable for validators.
        """
        if not file_changes.paths_only:
            return file_changes.files

        # Read actual content from disk for paths_only backends
        # Use PathResolver to prevent directory traversal attacks
        resolver = PathResolver(Path(cwd))
        files: dict[str, str] = {}

        for path in file_changes.paths():
            if self._should_skip_path_only_file(path):
                continue
            try:
                # Validate path is within cwd (prevents ../secret.txt attacks)
                resolved = resolver.resolve(path)
                if resolved.exists():
                    files[path] = resolved.read_text(encoding="utf-8")
            except PathSecurityError:
                logger.warning("skipping_unsafe_path", path=path, reason="security_violation")
            except OSError:
                logger.warning("file_read_failed", path=path, reason="os_error")
            except UnicodeDecodeError:
                logger.warning("file_read_failed", path=path, reason="not_utf8")

        return files

    @staticmethod
    def _should_skip_path_only_file(path: str) -> bool:
        candidate = Path(path)
        return (
            any(part in _IGNORED_PATH_ONLY_DIRS for part in candidate.parts)
            or candidate.suffix in _IGNORED_PATH_ONLY_SUFFIXES
        )

    def _dispatch_message(self, message: BackendMessage) -> MessageReceived:
        """Dispatch message to display/tracer and build event data.

        Single-dispatch pattern replaces triple match blocks.
        Adding new message types requires ONE change here.

        Args:
            message: Backend message to dispatch.

        Returns:
            Typed MessageReceived for CodingEvent.
        """
        content = ""
        tool_name: str | None = None

        match message:
            case PromptMessage():
                if self.display:
                    self.display.display_prompt(message.user_prompt, message.system_prompt)
                if self.tracer:
                    self.tracer.trace_prompt("Agenter", message.system_prompt or "", message.user_prompt)
                content = message.user_prompt

            case TextMessage():
                if self.display:
                    self.display.display_response(message.content, message.tokens)
                if self.tracer:
                    self.tracer.trace_response("Agenter", message.content)
                content = message.content

            case ToolCallMessage():
                if self.display:
                    self.display.report_tool_call(message.tool_name, message.args)
                if self.tracer:
                    self.tracer.log_tool_call("Agenter", message.tool_name, message.args)
                content = str(message.args)
                tool_name = message.tool_name

            case ToolResult():
                result_tool_name = message.tool_name or "unknown"
                if self.display:
                    self.display.report_tool_result(result_tool_name, message.output, message.success)
                if self.tracer:
                    self.tracer.log_tool_result("Agenter", result_tool_name, message.output, message.success)
                content = message.output
                tool_name = message.tool_name

        return MessageReceived(
            message_type=message.type,
            content=content,
            tool_name=tool_name,
        )

    async def run(
        self,
        request: CodingRequest,
        raise_on_budget_exceeded: bool = False,
    ) -> CodingResult:
        """Run the coding session to completion.

        Args:
            request: The coding task request
            raise_on_budget_exceeded: If True, raise BudgetExceededError when budget
                is exceeded. If False (default), return CodingResult with
                status=BUDGET_EXCEEDED and populated exceeded_limit/exceeded_values.

        Returns:
            CodingResult with status, files, and metrics. If output_type was
            specified in the request, result.output contains the structured output.
        """
        # Consume all events and return final result
        result = None

        async for event in self.stream_run(request):
            # Terminal events (COMPLETED, FAILED) carry the result directly
            if event.result is not None:
                result = event.result

        if result is None:
            # Should not happen, but handle gracefully
            result = CodingResult(
                status=CodingStatus.FAILED,
                files={},
                summary="Session ended without result",
                iterations=0,
                total_tokens=0,
                trace_dir=self.tracer.output_dir if self.tracer else None,
            )

        # Optionally raise exception for budget exceeded
        if raise_on_budget_exceeded and result.status == CodingStatus.BUDGET_EXCEEDED:
            from ..data_models import BudgetExceededError

            # Use values from result (populated from event data)
            if result.exceeded_limit and result.exceeded_values:
                raise BudgetExceededError(
                    result.summary,
                    limit_type=result.exceeded_limit,
                    limit_value=result.exceeded_values["limit_value"],
                    actual_value=result.exceeded_values["actual_value"],
                )

            # Fallback for edge cases (should be rare)
            budget = self._get_budget(request)
            raise BudgetExceededError(
                result.summary,
                limit_type="iterations",
                limit_value=budget.max_iterations,
                actual_value=result.iterations,
            )

        return result

    async def stream_run(
        self,
        request: CodingRequest,
    ) -> AsyncIterator[CodingEvent]:
        """Run the coding session, yielding events.

        Args:
            request: The coding task request

        Yields:
            CodingEvent for each significant step
        """
        budget = self._get_budget(request)
        tracker = BudgetMeter(budget)
        prev_tokens = 0
        prev_cost = 0.0

        await self.backend.connect(
            request.cwd,
            request.allowed_write_paths,
            output_type=request.output_type,
            system_prompt=request.system_prompt,
        )
        prompt = request.prompt
        logger.debug("session_started", cwd=str(request.cwd))

        # Emit SESSION_START event
        yield CodingEvent(
            type=CodingEventType.SESSION_START,
            data=SessionStarted(
                cwd=request.cwd,
                model=getattr(self.backend, "model", "unknown"),
                max_iterations=budget.max_iterations,
            ),
        )

        # Display session start
        if self.display:
            self.display.start_session(request.cwd, getattr(self.backend, "model", "unknown"))

        try:
            while True:
                # Check budget before starting iteration
                exceeded, reason = tracker.exceeded()
                if exceeded:
                    logger.debug("budget_exceeded", reason=reason)
                    usage = tracker.usage()

                    # Map BudgetLimitType enum to values for error reporting
                    limit_type = reason.value if reason else "iterations"
                    limit_value: int | float
                    actual_value: int | float

                    match reason:
                        case BudgetLimitType.ITERATIONS:
                            limit_value = budget.max_iterations
                            actual_value = tracker.iterations
                        case BudgetLimitType.TOKENS:
                            limit_value = budget.max_tokens if budget.max_tokens is not None else 0
                            actual_value = tracker.tokens_used
                        case BudgetLimitType.COST:
                            limit_value = budget.max_cost_usd if budget.max_cost_usd is not None else 0.0
                            actual_value = tracker.cost_usd
                        case BudgetLimitType.TIME:
                            limit_value = budget.max_time_seconds if budget.max_time_seconds is not None else 0.0
                            actual_value = float(usage["elapsed_seconds"])
                        case _:
                            # Defensive fallback
                            limit_value = budget.max_iterations
                            actual_value = tracker.iterations

                    # Prepare files with actual content (not empty strings)
                    file_changes = self.backend.modified_files()
                    result_files = self._prepare_files_for_validation(file_changes, request.cwd)
                    budget_failed_result = CodingResult(
                        status=CodingStatus.BUDGET_EXCEEDED,
                        files=result_files,
                        summary=f"Budget exceeded: {limit_type} ({actual_value} >= {limit_value})",
                        iterations=tracker.iterations,
                        total_tokens=self.backend.usage().total_tokens,
                        total_cost_usd=usage["cost_usd"],
                        total_duration_seconds=usage["elapsed_seconds"],
                        exceeded_limit=limit_type,
                        exceeded_values={"limit_value": limit_value, "actual_value": actual_value},
                        trace_dir=self.tracer.output_dir if self.tracer else None,
                    )
                    yield CodingEvent(
                        type=CodingEventType.FAILED,
                        data=TaskFailed(
                            status=CodingStatus.BUDGET_EXCEEDED.value,
                            files=result_files,
                            summary=f"Budget exceeded: {limit_type} ({actual_value} >= {limit_value})",
                            iterations=tracker.iterations,
                            total_tokens=self.backend.usage().total_tokens,
                            cost_usd=usage["cost_usd"],
                            duration_seconds=usage["elapsed_seconds"],
                            limit_type=limit_type,
                            limit_value=limit_value,
                            actual_value=actual_value,
                        ),
                        result=budget_failed_result,
                    )
                    # Emit SESSION_END for guaranteed cleanup event
                    yield CodingEvent(
                        type=CodingEventType.SESSION_END,
                        data=SessionEnded(
                            status="budget_exceeded",
                            iterations=tracker.iterations,
                            total_tokens=self.backend.usage().total_tokens,
                            cost_usd=usage["cost_usd"],
                            duration_seconds=usage["elapsed_seconds"],
                        ),
                    )
                    return

                tracker.add_iteration()
                logger.debug("starting_iteration", iteration=tracker.iterations)

                # Display iteration start
                if self.display:
                    self.display.start_iteration(tracker.iterations)

                yield CodingEvent(
                    type=CodingEventType.ITERATION_START,
                    data=IterationStarted(iteration=tracker.iterations),
                )

                # Execute backend and track tool failures
                tool_failures: list[str] = []
                async for message in self.backend.execute(prompt):
                    # Track tool failures ALWAYS (not just when display enabled)
                    if isinstance(message, ToolResult) and not message.success:
                        tool_failures.append(f"{message.tool_name}: {message.output}")

                    # Dispatch to display/tracer and build event data (single match)
                    event_data = self._dispatch_message(message)

                    yield CodingEvent(
                        type=CodingEventType.BACKEND_MESSAGE,
                        data=event_data,
                        message=message,
                    )

                # Update token usage and cost from backend
                # Use defensive max(0, delta) to handle backend misbehavior gracefully
                backend_usage = self.backend.usage()
                token_delta = backend_usage.total_tokens - prev_tokens
                cost_delta = backend_usage.cost_usd - prev_cost

                if token_delta < 0 or cost_delta < 0:
                    logger.warning(
                        "backend_usage_decreased_unexpectedly",
                        prev_tokens=prev_tokens,
                        new_tokens=backend_usage.total_tokens,
                        prev_cost=prev_cost,
                        new_cost=backend_usage.cost_usd,
                    )

                tracker.add_usage(
                    UsageDelta(
                        tokens=max(0, token_delta),
                        cost_usd=max(0.0, cost_delta),
                    )
                )
                prev_tokens = backend_usage.total_tokens
                prev_cost = backend_usage.cost_usd

                # Check budget after usage update - if exceeded but validation passes,
                # we'll return COMPLETED_WITH_LIMIT_EXCEEDED instead of COMPLETED
                budget_exceeded_after_execution, budget_exceeded_reason = tracker.exceeded()
                if budget_exceeded_after_execution:
                    logger.debug(
                        "budget_exceeded_after_execution",
                        reason=budget_exceeded_reason,
                        will_continue_to_validation=True,
                    )

                # Check if LLM explicitly refused the request
                refusal = getattr(self.backend, "refusal", lambda: None)()
                if refusal is not None:
                    logger.info("llm_refused_request", reason=refusal.reason, category=refusal.category)
                    usage = tracker.usage()
                    file_changes = self.backend.modified_files()
                    result_files = self._prepare_files_for_validation(file_changes, request.cwd)

                    refused_result = CodingResult(
                        status=CodingStatus.REFUSED,
                        files=result_files,
                        summary=f"LLM refused: {refusal.reason}",
                        iterations=tracker.iterations,
                        total_tokens=self.backend.usage().total_tokens,
                        total_cost_usd=usage["cost_usd"],
                        total_duration_seconds=usage["elapsed_seconds"],
                        trace_dir=self.tracer.output_dir if self.tracer else None,
                    )
                    yield CodingEvent(
                        type=CodingEventType.REFUSED,
                        data=RequestRefused(
                            reason=refusal.reason,
                            category=refusal.category,
                        ),
                        result=refused_result,
                    )
                    yield CodingEvent(
                        type=CodingEventType.FAILED,
                        data=TaskFailed(
                            status=CodingStatus.REFUSED.value,
                            files=result_files,
                            summary=f"LLM refused: {refusal.reason}",
                            iterations=tracker.iterations,
                            total_tokens=self.backend.usage().total_tokens,
                            cost_usd=usage["cost_usd"],
                            duration_seconds=usage["elapsed_seconds"],
                            refusal_reason=refusal.reason,
                            refusal_category=refusal.category,
                        ),
                        result=refused_result,
                    )
                    yield CodingEvent(
                        type=CodingEventType.SESSION_END,
                        data=SessionEnded(
                            status="refused",
                            iterations=tracker.iterations,
                            total_tokens=self.backend.usage().total_tokens,
                            cost_usd=usage["cost_usd"],
                            duration_seconds=usage["elapsed_seconds"],
                        ),
                    )
                    return

                file_changes = self.backend.modified_files()

                # Prepare files for validation (read from disk if paths_only)
                files_for_validation = self._prepare_files_for_validation(file_changes, request.cwd)

                # Emit VALIDATION_START before running validators
                yield CodingEvent(
                    type=CodingEventType.VALIDATION_START,
                    data=ValidationStarted(
                        validators=[v.__class__.__name__ for v in self._validator_chain.validators],
                        file_count=len(files_for_validation),
                    ),
                )

                # Run validators through chain and emit per-validator events
                chain_result = await self._validator_chain.validate(files_for_validation, request.cwd)

                for outcome in chain_result.validator_results:
                    yield CodingEvent(
                        type=CodingEventType.VALIDATION_RESULT,
                        data=ValidationCompleted(
                            validator=outcome.name,
                            passed=outcome.result.passed,
                            errors=outcome.result.errors,
                        ),
                    )

                all_passed = chain_result.passed
                all_errors = chain_result.errors

                # Display validation result
                if self.display:
                    self.display.report_validation(all_passed, all_errors)

                # Emit ITERATION_END with metrics
                iteration_usage = tracker.usage()
                yield CodingEvent(
                    type=CodingEventType.ITERATION_END,
                    data=IterationCompleted(
                        iteration=tracker.iterations,
                        passed=all_passed,
                        files_modified=len(file_changes),
                        tokens_used=iteration_usage["tokens_used"],
                        cost_usd=iteration_usage["cost_usd"],
                        elapsed_seconds=iteration_usage["elapsed_seconds"],
                    ),
                )

                if all_passed:
                    logger.debug(
                        "validation_passed",
                        iteration=tracker.iterations,
                        files_modified=len(file_changes),
                    )
                    usage = tracker.usage()

                    # Use files_for_validation which has actual content (not empty strings)
                    result_files = files_for_validation

                    # Determine final status - if budget exceeded after execution but
                    # validation passed, return COMPLETED_WITH_LIMIT_EXCEEDED
                    if budget_exceeded_after_execution:
                        final_status = CodingStatus.COMPLETED_WITH_LIMIT_EXCEEDED
                        exceeded_type = budget_exceeded_reason.value if budget_exceeded_reason else "unknown"
                        summary = f"Task completed but budget exceeded: {exceeded_type}"
                        session_status = "completed_with_limit_exceeded"
                    else:
                        final_status = CodingStatus.COMPLETED
                        summary = "Task completed successfully"
                        session_status = "completed"

                    # Display summary
                    if self.display:
                        self.display.display_summary(
                            success=True,
                            iterations=tracker.iterations,
                            tokens=self.backend.usage().total_tokens,
                            files=result_files,
                        )

                    # Get structured output if configured
                    structured_output = None
                    if request.output_type is not None:
                        structured_output = self.backend.structured_output()

                    completed_result = CodingResult(
                        status=final_status,
                        files=result_files,
                        summary=summary,
                        iterations=tracker.iterations,
                        total_tokens=self.backend.usage().total_tokens,
                        total_cost_usd=usage["cost_usd"],
                        total_duration_seconds=usage["elapsed_seconds"],
                        output=structured_output,
                        trace_dir=self.tracer.output_dir if self.tracer else None,
                    )
                    yield CodingEvent(
                        type=CodingEventType.COMPLETED,
                        data=TaskCompleted(
                            status=final_status.value,
                            files=result_files,
                            summary=summary,
                            iterations=tracker.iterations,
                            total_tokens=self.backend.usage().total_tokens,
                            cost_usd=usage["cost_usd"],
                            duration_seconds=usage["elapsed_seconds"],
                            output=structured_output,
                        ),
                        result=completed_result,
                    )
                    # Emit SESSION_END for guaranteed cleanup event
                    yield CodingEvent(
                        type=CodingEventType.SESSION_END,
                        data=SessionEnded(
                            status=session_status,
                            iterations=tracker.iterations,
                            total_tokens=self.backend.usage().total_tokens,
                            cost_usd=usage["cost_usd"],
                            duration_seconds=usage["elapsed_seconds"],
                        ),
                    )
                    return

                # Prepare retry prompt with errors and tool failures
                logger.debug(
                    "validation_failed_retrying",
                    iteration=tracker.iterations,
                    error_count=len(all_errors),
                )
                prompt = build_retry_prompt(
                    original_prompt=request.prompt,
                    validation_errors=all_errors,
                    tool_failures=tool_failures,
                )

            # This point is never reached due to budget check

        finally:
            # Shield cleanup from cancellation to ensure resources are freed
            # Catch CancelledError from anyio cancel scope mismatches in MCP backends
            try:  # noqa: SIM105 - can't use contextlib.suppress with await
                await asyncio.shield(self.backend.disconnect())
            except asyncio.CancelledError:
                # MCP backends may raise CancelledError during cleanup due to
                # anyio cancel scope task mismatches - this is expected in pytest
                pass
