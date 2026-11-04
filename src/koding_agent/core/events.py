"""Event types, safe delivery to the front end, and the tool lifecycle."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from koding_agent.core.tools import RunTool, ToolCall, ToolResult


@dataclass(frozen=True)
class ToolStart:
    tool: ToolCall


@dataclass(frozen=True)
class ToolEnd:
    tool: ToolCall
    result: ToolResult


@dataclass(frozen=True)
class ToolCancelled:
    tool: ToolCall


@dataclass(frozen=True)
class TurnStart:
    turn: int
    max_turns: int


type Event = ToolStart | ToolEnd | ToolCancelled | TurnStart
type OnEvent = Callable[[Event], None]  # plain def, must return immediately


def ignore_event(_: Event) -> None: ...


def emit(on_event: OnEvent, event: Event) -> None:
    """Call the front end's callback; a broken callback must not kill the agent.

    Catches Exception, not BaseException, so CancelledError still passes through.
    """
    try:
        on_event(event)
    except Exception:
        logging.exception("on_event failed")


async def run_tool[C: ToolCall, R: ToolResult](
    tool_call: C,  # could be BashCall or SkillCall; function shouldn't know anything about the Call-attributes
    tool_fn: RunTool[C, R],
    on_event: OnEvent = ignore_event,
) -> R:
    emit(on_event, ToolStart(tool=tool_call))
    try:
        result = await tool_fn(tool_call)
    except asyncio.CancelledError:
        emit(on_event, ToolCancelled(tool=tool_call))
        raise
    emit(on_event, ToolEnd(tool=tool_call, result=result))
    return result
