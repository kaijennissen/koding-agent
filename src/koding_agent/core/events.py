from collections.abc import Callable
from dataclasses import dataclass
from typing import assert_never

from koding_agent.core.colours import GREEN, GREY, RED, RESET, YELLOW
from koding_agent.core.tools import BashCall, BashResult, ToolCall


@dataclass(frozen=True)
class TurnStart:
    turn: int
    max_turns: int


@dataclass(frozen=True)
class ToolStart:
    call: ToolCall


@dataclass(frozen=True)
class ToolEnd:
    call: ToolCall
    result: BashResult


@dataclass(frozen=True)
class ToolCancelled:
    call: ToolCall


type Event = ToolStart | ToolEnd | ToolCancelled | TurnStart
type OnEvent = Callable[[Event], None]  # plain def, must return immediately


def print_event(event: Event) -> None:
    match event:
        case TurnStart(turn=turn, max_turns=max_turns):
            print(f"{GREY}Turn: {turn}/{max_turns}{RESET}")
        case ToolStart(call=BashCall(cmd=cmd)):
            print(f"{YELLOW}running: {cmd}{RESET}")

        case ToolEnd(call=BashCall(cmd=cmd), result=BashResult(returncode=0)):
            print(f"{GREEN}success: {cmd}{RESET}")
        case ToolEnd(call=BashCall(cmd=cmd), result=BashResult(timed_out=True)):
            print(f"{RED}timeout: {cmd}{RESET}")
        case ToolEnd(call=BashCall(cmd=cmd), result=BashResult(returncode=code)):
            print(f"{RED}failure (exit {code}): {cmd}{RESET}")
        case ToolCancelled(call=BashCall(cmd=cmd)):
            print(f"{RED}cancelled: {cmd}{RESET}")
        case _:
            assert_never(event)
