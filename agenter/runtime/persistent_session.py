"""Persistent multi-request coding sessions."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ..data_models import (
    BackendError,
    Budget,
    BudgetExceededError,
    CodingRequest,
    CodingResult,
    CodingStatus,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    from ..data_models import CodingEvent, ModifiedFiles, Usage
    from .session import CodingSession


class PersistentCodingSession:
    """Own a backend connection across serialized follow-up requests.

    Instances are created by :meth:`AutonomousCodingAgent.open_session`.
    The remote ACP process and session stay alive until :meth:`close` is
    called, so each prompt sees the coding agent's prior conversation state.
    """

    def __init__(
        self,
        session: CodingSession,
        base_request: CodingRequest,
        session_budget: Budget | None = None,
        resume_session_id: str | None = None,
    ) -> None:
        self._session = session
        self._base_request = base_request
        self._session_budget = session_budget
        self._resume_session_id = resume_session_id
        self._lock = asyncio.Lock()
        self._closed = False
        self._started = False
        self._session_id: str | None = None
        self._request_count = 0
        self._session_iterations = 0
        self._session_tokens = 0
        self._session_cost_usd = 0.0
        self._started_at = time.monotonic()
        self._final_usage: Usage | None = None
        self._final_modified_files: ModifiedFiles | None = None

    async def start(self) -> PersistentCodingSession:
        """Connect the backend and create or resume its remote session."""
        if self._started:
            return self
        if self._closed:
            raise BackendError("PersistentCodingSession is closed.")
        await self._session.connect(self._base_request, resume_session_id=self._resume_session_id)
        self._session_id = getattr(self._session.backend, "session_id", None)
        self._started = True
        self._started_at = time.monotonic()
        return self

    async def __aenter__(self) -> PersistentCodingSession:
        return await self.start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    @property
    def session_id(self) -> str | None:
        """Stable ACP session ID for tracing and optional later resumption."""
        return self._session_id

    @property
    def request_count(self) -> int:
        """Number of follow-up requests started in this session."""
        return self._request_count

    @property
    def closed(self) -> bool:
        return self._closed

    def usage(self) -> Usage:
        """Return cumulative backend usage for the live remote session."""
        if self._final_usage is not None:
            return self._final_usage
        return self._session.backend.usage()

    def modified_files(self) -> ModifiedFiles:
        """Return cumulative file changes since the session was opened."""
        if self._final_modified_files is not None:
            return self._final_modified_files
        return self._session.backend.modified_files()

    async def execute(
        self,
        request: str | CodingRequest,
        *,
        budget: Budget | None = None,
        max_iterations: int | None = None,
        raise_on_budget_exceeded: bool = False,
    ) -> CodingResult:
        """Execute one serialized prompt while preserving remote agent state."""
        prepared = self._prepare_request(request, budget=budget, max_iterations=max_iterations)
        result = None
        async for event in self._stream_prepared(prepared):
            if event.result is not None:
                result = event.result

        if result is None:
            raise BackendError("Persistent request ended without a result.")
        if raise_on_budget_exceeded and result.status == CodingStatus.BUDGET_EXCEEDED:
            if result.exceeded_limit and result.exceeded_values:
                raise BudgetExceededError(
                    result.summary,
                    limit_type=result.exceeded_limit,
                    limit_value=result.exceeded_values["limit_value"],
                    actual_value=result.exceeded_values["actual_value"],
                )
            raise BudgetExceededError(
                result.summary,
                limit_type="iterations",
                limit_value=prepared.max_iterations,
                actual_value=result.iterations,
            )
        return result

    async def stream_execute(
        self,
        request: str | CodingRequest,
        *,
        budget: Budget | None = None,
        max_iterations: int | None = None,
    ) -> AsyncIterator[CodingEvent]:
        """Stream one serialized follow-up request with stable trace IDs."""
        prepared = self._prepare_request(request, budget=budget, max_iterations=max_iterations)
        async for event in self._stream_prepared(prepared):
            yield event

    async def _stream_prepared(self, request: CodingRequest) -> AsyncIterator[CodingEvent]:
        self._ensure_open()
        async with self._lock:
            self._ensure_open()
            self._request_count += 1
            request_index = self._request_count
            effective_request = self._apply_session_budget(request)
            recorded = False

            async for event in self._session.stream_request(effective_request):
                event.session_id = self._session_id
                event.request_index = request_index
                if event.result is not None and not recorded:
                    self._record_result(event.result, request_index)
                    recorded = True
                yield event

    async def cancel(self) -> None:
        """Cancel an active prompt without discarding the persistent session."""
        self._ensure_open()
        cancel = getattr(self._session.backend, "cancel", None)
        if cancel is None:
            raise BackendError("This backend does not support cancelling an active prompt.")
        await cancel()

    async def close(self) -> None:
        """Wait for any active prompt and close the backend process."""
        if self._closed:
            return
        async with self._lock:
            if self._closed:
                return
            self._final_usage = self._session.backend.usage()
            self._final_modified_files = self._session.backend.modified_files()
            await self._session.disconnect()
            self._closed = True

    def _prepare_request(
        self,
        request: str | CodingRequest,
        *,
        budget: Budget | None,
        max_iterations: int | None,
    ) -> CodingRequest:
        if isinstance(request, str):
            return self._base_request.model_copy(
                update={
                    "prompt": request,
                    "budget": budget,
                    "max_iterations": (
                        max_iterations if max_iterations is not None else self._base_request.max_iterations
                    ),
                }
            )

        if budget is not None or max_iterations is not None:
            raise ValueError("budget and max_iterations must be set on CodingRequest when a request object is used.")
        if Path(request.cwd).resolve() != Path(self._base_request.cwd).resolve():
            raise BackendError("A persistent coding session cannot change cwd between follow-ups.")
        if request.allowed_write_paths != self._base_request.allowed_write_paths:
            raise BackendError("A persistent coding session cannot change allowed_write_paths between follow-ups.")
        if request.system_prompt != self._base_request.system_prompt:
            raise BackendError("A persistent coding session cannot change system_prompt between follow-ups.")
        if request.output_type is not self._base_request.output_type:
            raise BackendError("A persistent coding session cannot change output_type between follow-ups.")
        return request

    def _apply_session_budget(self, request: CodingRequest) -> CodingRequest:
        if self._session_budget is None:
            return request

        request_budget = request.budget or Budget(max_iterations=request.max_iterations)
        elapsed = time.monotonic() - self._started_at
        remaining_iterations = max(0, self._session_budget.max_iterations - self._session_iterations)
        remaining_tokens = self._remaining_int(self._session_budget.max_tokens, self._session_tokens)
        remaining_cost = self._remaining_float(self._session_budget.max_cost_usd, self._session_cost_usd)
        remaining_time = self._remaining_float(self._session_budget.max_time_seconds, elapsed)
        effective = Budget(
            max_iterations=min(request_budget.max_iterations, remaining_iterations),
            max_tokens=self._minimum_int(request_budget.max_tokens, remaining_tokens),
            max_cost_usd=self._minimum_float(request_budget.max_cost_usd, remaining_cost),
            max_time_seconds=self._minimum_float(request_budget.max_time_seconds, remaining_time),
        )
        return request.model_copy(update={"budget": effective})

    @staticmethod
    def _remaining_int(limit: int | None, used: int) -> int | None:
        return None if limit is None else max(0, limit - used)

    @staticmethod
    def _remaining_float(limit: float | None, used: float) -> float | None:
        return None if limit is None else max(0, limit - used)

    @staticmethod
    def _minimum_int(left: int | None, right: int | None) -> int | None:
        if left is None:
            return right
        if right is None:
            return left
        return min(left, right)

    @staticmethod
    def _minimum_float(left: float | None, right: float | None) -> float | None:
        if left is None:
            return right
        if right is None:
            return left
        return min(left, right)

    def _record_result(self, result: CodingResult, request_index: int) -> None:
        self._session_iterations += result.iterations
        self._session_tokens += result.total_tokens
        self._session_cost_usd += result.total_cost_usd
        result.session_id = self._session_id
        result.request_index = request_index
        result.session_total_tokens = self._session_tokens
        result.session_total_cost_usd = self._session_cost_usd
        result.session_total_duration_seconds = time.monotonic() - self._started_at

    def _ensure_open(self) -> None:
        if not self._started:
            raise BackendError("PersistentCodingSession has not been started.")
        if self._closed:
            raise BackendError("PersistentCodingSession is closed.")


__all__ = ["PersistentCodingSession"]
