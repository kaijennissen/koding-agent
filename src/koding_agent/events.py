"""Events the agent loop emits, and the plain-text printer for them."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import assert_never

from koding_agent.banner import GREY, RESET, rgb
from koding_agent.tools import RunTool, ToolCall, ToolResult

RED = rgb("#FF5555")
GREEN = rgb("#50FA7B")
YELLOW = rgb("#FFCC00")


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


type Event = ToolStart | ToolEnd | ToolCancelled | TurnStart
type OnEvent = Callable[[Event], None]  # plain def, must return immediately


@dataclass(frozen=True)
class TurnStart:
    turn: int
    max_turns: int


def print_event(event: Event) -> None:
    match event:
        case TurnStart(turn=turn, max_turns=max_turns):
            print(f"{GREY}Turn: {turn}/{max_turns}{RESET}")
        case ToolStart(tool=tool):
            print(f"{YELLOW}running: {tool.display}{RESET}")
        case ToolEnd(tool=tool, result=result) if result.status == "ok":
            print(f"{GREEN}success: {tool.display}{RESET}")
        case ToolEnd(tool=tool):
            print(f"{RED}failure: {tool.display}{RESET}")
        case ToolCancelled(tool=tool):
            print(f"{RED}cancelled: {tool.display}{RESET}")
        case _:
            assert_never(event)


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
