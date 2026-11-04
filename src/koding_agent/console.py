"""Terminal presentation of core events: coloured one-line progress for the CLI."""

from typing import assert_never

from koding_agent.banner import GREY, RESET, rgb
from koding_agent.core import (
    Event,
    ToolCancelled,
    ToolEnd,
    ToolStart,
    TurnStart,
)

RED = rgb("#FF5555")
GREEN = rgb("#50FA7B")
YELLOW = rgb("#FFCC00")


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
