"""Command line entry point: picks a mode and wires the core to a front end.

    koding-agent [PROMPT]       TUI; PROMPT, if given, is sent first
    koding-agent -p PROMPT      headless one-shot, pipe-clean output
    koding-agent --plain        line-based REPL

This is the only module that imports both the core and the front ends.
"""

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Literal

import httpx2

from koding_agent import __version__
from koding_agent.banner import RESET, print_banner, prompt_text
from koding_agent.console import RED, print_event
from koding_agent.core import (
    DEFAULT_COMMAND_TIMEOUT,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    Context,
    LlmError,
    Message,
    SkillLibrary,
    build_system_prompt,
    llm_call,
    load_context,
    load_skills,
    run_agent,
)

DEFAULT_OLLAMA_URL = "http://host.docker.internal:11434"  # the Docker host gateway
LLM_TIMEOUT = 120  # seconds per model request; without streaming, the whole reply

type Mode = Literal["headless", "plain", "tui"]


def ollama_url() -> str:
    """Where Ollama listens: $OLLAMA_HOST (a full URL), else the Docker host gateway."""
    return os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL


def ollama_client() -> httpx2.AsyncClient:
    token = os.environ.get("OPENROUTER_API_KEY")
    return httpx2.AsyncClient(
        base_url=ollama_url(),
        headers={"Authorization": f"Bearer {token}"},
        timeout=LLM_TIMEOUT,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "koding-agent",
        epilog="environment: OLLAMA_HOST  Ollama base URL, e.g. http://localhost:11434 "
        f"(default: {DEFAULT_OLLAMA_URL})",
    )
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
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model to use (default: %(default)s)",
    )
    parser.add_argument(
        "-t", "--max-turns", default=DEFAULT_MAX_TURNS, type=int, help="max turns"
    )
    parser.add_argument(
        "--command-timeout",
        default=DEFAULT_COMMAND_TIMEOUT,
        type=float,
        help="seconds a bash command may run before it is killed (default: %(default)s)",
    )
    parser.add_argument(
        "-nc",
        "--no-context",
        action="store_true",
        help="do not load AGENTS.md from the current directory",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        metavar="DIR",
        help="load skills from DIR/*/SKILL.md",
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


async def run_headless(
    prompt: str,
    model: str,
    max_turns: int,
    command_timeout: float,
    skills: SkillLibrary,
    context: Context | None = None,
) -> None:
    async with ollama_client() as client:
        messages: list[Message] = [
            {
                "role": "system",
                "content": build_system_prompt(
                    context, skills.catalog(), command_timeout=command_timeout
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            answer, _ = await run_agent(
                messages,
                max_turns=max_turns,
                call=partial(llm_call, client, model),
                skills=skills,
                on_event=print_event,
                command_timeout=command_timeout,
            )
        except LlmError as e:
            sys.exit(f"{RED}{e}{RESET}")  # one-shot: report and fail
        print("\n" + answer + "\n")


async def run_plain(
    model: str,
    max_turns: int,
    command_timeout: float,
    skills: SkillLibrary,
    context: Context | None = None,
) -> None:
    async with ollama_client() as client:
        call = partial(llm_call, client, model)
        messages: list[Message] = [
            {
                "role": "system",
                "content": build_system_prompt(
                    context, skills.catalog(), command_timeout=command_timeout
                ),
            }
        ]
        print_banner(model, context.path.name if context else None, skills.skills)
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
                    skills=skills,
                    on_event=print_event,
                    command_timeout=command_timeout,
                )
            except LlmError as e:
                # Keep the session: history is still valid, so the user can retry.
                print(f"{RED}{e}{RESET}")
                continue
            print("\n" + answer + "\n")


async def run_tui(
    model: str,
    max_turns: int,
    first_prompt: str | None,
    command_timeout: float,
    skills: SkillLibrary,
    context: Context | None = None,
) -> None:
    from koding_agent.tui.app import AgentApp  # keeps Textual off the -p path

    async with ollama_client() as client:
        app = AgentApp(
            partial(llm_call, client, model),
            model=model,
            max_turns=max_turns,
            first_prompt=first_prompt,
            command_timeout=command_timeout,
            skills=skills,
            context=context,
        )
        await app.run_async()


def cli(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    context = None if args.no_context else load_context()
    skills = load_skills(args.skills_dir)
    match select_mode(args):
        case "headless":
            asyncio.run(
                run_headless(
                    prompt=args.headless_prompt,
                    model=args.model,
                    max_turns=args.max_turns,
                    command_timeout=args.command_timeout,
                    skills=skills,
                    context=context,
                )
            )
        case "plain":
            asyncio.run(
                run_plain(
                    model=args.model,
                    max_turns=args.max_turns,
                    command_timeout=args.command_timeout,
                    skills=skills,
                    context=context,
                )
            )
        case "tui":
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                sys.exit(
                    "The TUI needs an interactive terminal. "
                    "Use -p PROMPT for a one-shot run, or --plain for the REPL."
                )
            asyncio.run(
                run_tui(
                    model=args.model,
                    max_turns=args.max_turns,
                    first_prompt=args.prompt,
                    command_timeout=args.command_timeout,
                    skills=skills,
                    context=context,
                )
            )


if __name__ == "__main__":
    cli()
