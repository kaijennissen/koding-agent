"""Reporting progress: emit guards the callback, run_tool brackets a command,
print_event is the CLI's callback."""

import asyncio
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import pytest

from fakes import ok
from koding_agent.core.agent import (
    BashCall,
    BashResult,
    Event,
    ToolCancelled,
    ToolEnd,
    ToolStart,
    TurnStart,
    emit,
    print_event,
    run_tool,
)
from koding_agent.core.colours import GREEN, GREY, RED, RESET, YELLOW

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
        async def tool(cmd: str) -> BashResult:
            return ok(cmd)

        result = await run_tool(call, tool, events.append)

        assert result == ok("ls")
        assert events == [ToolStart(call), ToolEnd(call, result)]

    async def test_reports_cancellation(
        self, call: BashCall, events: list[Event]
    ) -> None:
        """A cancelled command still ends its row, so no bullet stays 'running'."""
        started = asyncio.Event()

        async def hang(_: str) -> BashResult:
            started.set()
            await asyncio.Event().wait()  # never set
            raise AssertionError("unreachable")

        task = asyncio.create_task(run_tool(call, hang, events.append))
        await started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == [ToolStart(call), ToolCancelled(call)]


@pytest.mark.parametrize(
    "event,line",
    [
        pytest.param(TurnStart(0, 5), f"{GREY}Turn: 0/5", id="turn"),
        pytest.param(ToolStart(LS), f"{YELLOW}running: ls", id="start"),
        pytest.param(ToolEnd(LS, ok("ls")), f"{GREEN}success: ls", id="exit-0"),
        pytest.param(
            ToolEnd(LS, BashResult.completed("ls", "", "", 2)),
            f"{RED}failure (exit 2): ls",
            id="exit-2",
        ),
        # must not fall through to the catch-all failure case above it
        pytest.param(
            ToolEnd(LS, BashResult.timeout("ls")), f"{RED}timeout: ls", id="timeout"
        ),
        pytest.param(ToolCancelled(LS), f"{RED}cancelled: ls", id="cancelled"),
    ],
)
def test_print_event(
    capsys: pytest.CaptureFixture[str], event: Event, line: str
) -> None:
    """One coloured line per event; the colour carries the status."""
    print_event(event)

    assert capsys.readouterr().out == f"{line}{RESET}\n"
