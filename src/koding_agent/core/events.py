"""Event types the core emits while running, and `print_event` to render them in a terminal."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import assert_never

from koding_agent.core import tools
from koding_agent.core.colours import GREEN, GREY, RED, RESET, YELLOW


@dataclass(frozen=True)
class TurnStart:
    turn: int
    max_turns: int


@dataclass(frozen=True)
class ToolStart:
    call: tools.ToolCall


@dataclass(frozen=True)
class ToolEnd:
    call: tools.ToolCall
    result: tools.BashResult


@dataclass(frozen=True)
class ToolCancelled:
    call: tools.ToolCall


type Event = ToolStart | ToolEnd | ToolCancelled | TurnStart
type OnEvent = Callable[[Event], None]  # plain def, must return immediately


def print_event(event: Event) -> None:
    match event:
        case TurnStart(turn=turn, max_turns=max_turns):
            print(f"{GREY}Turn: {turn}/{max_turns}{RESET}")
        case ToolStart(call=tools.BashCall(cmd=cmd)):
            print(f"{YELLOW}running: {cmd}{RESET}")

        case ToolEnd(
            call=tools.BashCall(cmd=cmd), result=tools.BashResult(returncode=0)
        ):
            print(f"{GREEN}success: {cmd}{RESET}")
        case ToolEnd(
            call=tools.BashCall(cmd=cmd), result=tools.BashResult(timed_out=True)
        ):
            print(f"{RED}timeout: {cmd}{RESET}")
        case ToolEnd(
            call=tools.BashCall(cmd=cmd), result=tools.BashResult(returncode=code)
        ):
            print(f"{RED}failure (exit {code}): {cmd}{RESET}")
        case ToolCancelled(call=tools.BashCall(cmd=cmd)):
            print(f"{RED}cancelled: {cmd}{RESET}")
        case _:
            assert_never(event)
