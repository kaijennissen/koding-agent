"""Command line entry point: picks a mode and wires the core to a front end.

    koding-agent [PROMPT]       TUI; PROMPT, if given, is sent first
    koding-agent -p PROMPT      headless one-shot, pipe-clean output
    koding-agent --plain        line-based REPL

This is the only module that imports both the core and the front ends.
"""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Literal

import httpx2

from koding_agent import __version__
from koding_agent.agent import (
    DEFAULT_MAX_TURNS,
    MODEL,
    RED,
    RESET,
    SYSTEM_PROMPT,
    LlmError,
    Message,
    llm_call,
    print_event,
    run_agent,
)
from koding_agent.banner import print_banner, prompt_text

OLLAMA_URL = "http://host.docker.internal:11434"

type Mode = Literal["headless", "plain", "tui"]


def ollama_client() -> httpx2.AsyncClient:
    return httpx2.AsyncClient(base_url=OLLAMA_URL, timeout=120)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("koding-agent")
    parser.add_argument(
        "prompt", nargs="?", help="first message; opens the TUI with it already sent"
    )
    parser.add_argument(
        "-p",
        "--prompt",
        dest="headless_prompt",
        metavar="PROMPT",
        help="run PROMPT once without a UI and print the answer",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="use the line-based REPL instead of the TUI",
    )
    parser.add_argument(
        "-t", "--max-turns", default=DEFAULT_MAX_TURNS, type=int, help="max turns"
    )
    parser.add_argument(
        "--timeout",
        default=60,
        type=float,
        help="seconds a bash command may run before it is killed (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    args = parser.parse_args(argv)
    if args.headless_prompt and args.plain:
        parser.error("-p and --plain cannot be combined")
    if args.prompt and (args.headless_prompt or args.plain):
        parser.error("a positional PROMPT is only for the TUI; use -p PROMPT instead")
    return args


def select_mode(args: argparse.Namespace) -> Mode:
    if args.headless_prompt:
        return "headless"
    return "plain" if args.plain else "tui"


async def run_headless(prompt: str, max_turns: int, skills_dir: Path) -> None:
    async with ollama_client() as client:
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            answer, _ = await run_agent(
                messages,
                max_turns=max_turns,
                call=partial(llm_call, client),
                on_event=print_event,
            )
        except LlmError as e:
            sys.exit(f"{RED}{e}{RESET}")  # one-shot: report and fail
        print("\n" + answer + "\n")


async def run_plain(max_turns: int, skills_dir: Path) -> None:
    async with ollama_client() as client:
        call = partial(llm_call, client)
        messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
        print_banner(MODEL)
        while True:
            prompt = input(prompt_text())
            if prompt == "exit":
                return
            messages.append({"role": "user", "content": prompt})
            try:
                answer, messages = await run_agent(
                    messages,
                    max_turns=max_turns,
                    call=call,
                    skills_dir=skills_dir,
                    on_event=print_event,
                )
            except LlmError as e:
                # Keep the session: history is still valid, so the user can retry.
                print(f"{RED}{e}{RESET}")
                continue
            print("\n" + answer + "\n")


async def run_tui(max_turns: int, first_prompt: str | None, skills_dir: Path) -> None:
    from koding_agent.tui.app import AgentApp  # keeps Textual off the -p path

    async with ollama_client() as client:
        app = AgentApp(
            partial(llm_call, client),
            model=MODEL,
            max_turns=max_turns,
            first_prompt=first_prompt,
            skills_dir=skills_dir,
        )
        await app.run_async()


def cli(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    tool_fn = partial(run_bash_tool, timeout=args.timeout)
    match select_mode(args):
        case "headless":
            asyncio.run(
                run_headless(
                    prompt=args.headless_prompt,
                    max_turns=args.max_turns,
                    skills_dir=args.skills_dir,
                )
            )
        case "plain":
            asyncio.run(run_plain(max_turns=args.max_turns, skills_dir=args.skills_dir))
        case "tui":
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                sys.exit(
                    "The TUI needs an interactive terminal. "
                    "Use -p PROMPT for a one-shot run, or --plain for the REPL."
                )
            asyncio.run(
                run_tui(
                    max_turns=args.max_turns,
                    first_prompt=args.prompt,
                    skills_dir=args.skills_dir,
                )
            )


if __name__ == "__main__":
    cli()
