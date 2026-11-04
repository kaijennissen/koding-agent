"""Reporting progress: emit guards the callback, run_tool brackets a command."""

import asyncio
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import pytest

from fakes import ok
from koding_agent.core import (
    BashCall,
    BashResult,
    Event,
    ToolCancelled,
    ToolEnd,
    ToolStart,
    TurnStart,
)
from koding_agent.core.events import emit, run_tool

LS = BashCall(id="1", cmd="ls")


@pytest.mark.parametrize(
    "error,expectation,logged",
    [
        pytest.param(
            ValueError("widget gone"), nullcontext(), True, id="bug-is-logged"
        ),
        pytest.param(
            asyncio.CancelledError(),
            pytest.raises(asyncio.CancelledError),
            False,
            id="cancel-gets-through",
        ),
    ],
)
def test_emit_survives_a_broken_callback_but_not_a_cancel(
    caplog: pytest.LogCaptureFixture,
    error: BaseException,
    expectation: AbstractContextManager[Any],
    logged: bool,
) -> None:
    """A broken front end must not kill the agent, but Esc must still stop it."""

    def callback(_: Event) -> None:
        raise error

    with expectation:
        emit(callback, TurnStart(turn=0, max_turns=1))

    assert bool(caplog.records) is logged


class TestRunTool:
    @pytest.fixture
    def call(self) -> BashCall:
        return LS

    async def test_reports_start_and_end_with_the_result(
        self, call: BashCall, events: list[Event]
    ) -> None:
        async def tool(call: BashCall) -> BashResult:
            return ok(call.cmd)

        result = await run_tool(call, tool, events.append)

        assert result == ok("ls")
        assert events == [ToolStart(call), ToolEnd(call, result)]

    async def test_reports_cancellation(
        self, call: BashCall, events: list[Event]
    ) -> None:
        """A cancelled command still ends its row, so no bullet stays 'running'."""
        started = asyncio.Event()

        async def hang(_: BashCall) -> BashResult:
            started.set()
            await asyncio.Event().wait()  # never set
            raise AssertionError("unreachable")

        task = asyncio.create_task(run_tool(call, hang, events.append))
        await started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == [ToolStart(call), ToolCancelled(call)]
