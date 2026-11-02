"""Running bash commands and bracketing them with events."""

import asyncio
import logging
import os
import signal
import uuid
from collections.abc import Awaitable, Callable

from koding_agent.core.tool_types import BashResult, ToolCall

from koding_agent.core.events import (
    Event,
    OnEvent,
    ToolCancelled,
    ToolEnd,
    ToolStart,
)

type RunTool = Callable[[str], Awaitable[BashResult]]  # cmd -> result


def create_uuid():
    return str(uuid.uuid4()).split("-")[0]


def format_observation(result: BashResult) -> str:
    """Turn a BashResult into the text the model reads."""
    if result.timed_out:
        return (
            f"Command `{result.cmd}` was killed because it hit the time limit. "
            "It did not finish, so its result and side effects are unknown, "
            "and any output it produced was discarded."
        )
    return (
        f"Command `{result.cmd}` finished with exit code {result.returncode}.\n"
        f'stdout: "{result.stdout}"\n'
        f'stderr: "{result.stderr}"'
    )


async def run_bash_tool(cmd: str, *, timeout: float) -> BashResult:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    proc_id = proc.pid

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return BashResult.completed(
            cmd=cmd,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            returncode=proc.returncode,
        )

    except TimeoutError:
        return BashResult.timeout(cmd)

    finally:
        try:
            os.killpg(proc_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()


def _ignore(_: Event) -> None: ...


def emit(on_event: OnEvent, event: Event) -> None:
    """Call the front end's callback; a broken callback must not kill the agent.

    Catches Exception, not BaseException, so CancelledError still passes through.
    """
    try:
        on_event(event)
    except Exception:
        logging.exception("on_event failed")


async def run_tool(
    tool_call: ToolCall,
    tool_fn: RunTool,
    on_event: OnEvent = _ignore,
) -> BashResult:
    emit(on_event, ToolStart(call=tool_call))
    try:
        result = await tool_fn(tool_call.cmd)
    except asyncio.CancelledError:
        emit(on_event, ToolCancelled(call=tool_call))
        raise
    emit(on_event, ToolEnd(call=tool_call, result=result))
    return result
