"""run_agent: the act -> observe -> decide loop, driven by a fake model and fake tools."""

import asyncio
from pathlib import Path

import pytest

from fakes import ok, ok_tool, replies
from koding_agent.agent import (
    MIXED_ACTIONS_ERROR,
    SYSTEM_PROMPT,
    BashResult,
    Context,
    Event,
    LlmError,
    Message,
    ToolCancelled,
    ToolEnd,
    ToolStart,
    TurnStart,
    build_system_prompt,
    format_observation,
    load_context,
    run_agent,
)
from koding_agent.cli import parse_args

BASH_LS = "<action><bash>ls</bash></action>"
EXIT_DONE = "<action><exit>done</exit></action>"
MIXED = "<action><bash>ls</bash><exit>done</exit></action>"
LS_OBSERVATION = format_observation(ok("ls"))


def assistant(content: str) -> Message:
    return {"role": "assistant", "content": content}


def user(content: str) -> Message:
    return {"role": "user", "content": content}


def tool_log(events: list[Event]) -> list[tuple[str, str]]:
    """Tool events as (kind, cmd). Call ids are random, so they are left out."""
    log: list[tuple[str, str]] = []
    for event in events:
        match event:
            case ToolStart(call):
                log.append(("start", call.cmd))
            case ToolEnd(call, _):
                log.append(("end", call.cmd))
            case ToolCancelled(call):
                log.append(("cancelled", call.cmd))
            case TurnStart():
                pass
    return log


@pytest.fixture
def history() -> list[Message]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, user("task")]


class TestRunAgent:
    @pytest.mark.parametrize(
        "script,max_turns,answer,appended",
        [
            pytest.param(
                [EXIT_DONE], 5, "done", [assistant(EXIT_DONE)], id="exit-at-once"
            ),
            pytest.param(
                [BASH_LS, EXIT_DONE],
                5,
                "done",
                [assistant(BASH_LS), user(LS_OBSERVATION), assistant(EXIT_DONE)],
                id="bash-then-exit",
            ),
            pytest.param(
                [MIXED, EXIT_DONE],
                5,
                "done",
                [assistant(MIXED), user(MIXED_ACTIONS_ERROR), assistant(EXIT_DONE)],
                id="mixed-action-is-fed-back",
            ),
            pytest.param(
                [BASH_LS, BASH_LS],
                2,
                "Ran out of turns (2) without an <exit> action.",
                [assistant(BASH_LS), user(LS_OBSERVATION)] * 2,
                id="out-of-turns",
            ),
        ],
    )
    async def test_conversation(
        self,
        history: list[Message],
        script: list[str],
        max_turns: int,
        answer: str,
        appended: list[Message],
    ) -> None:
        """The answer, and exactly the messages the run adds to the history."""
        before = list(history)

        result, messages = await run_agent(
            history, max_turns, replies(*script), ok_tool
        )

        assert result == answer
        assert messages == before + appended

    async def test_tools_run_concurrently_and_observations_keep_command_order(
        self, history: list[Message], events: list[Event]
    ) -> None:
        """'slow' only finishes once 'fast' has run, so a sequential loop would hang."""
        fast_done = asyncio.Event()

        async def tool(cmd: str) -> BashResult:
            if cmd == "slow":
                await fast_done.wait()
            else:
                fast_done.set()
            return ok(cmd)

        bash = "<action><bash>slow</bash><bash>fast</bash></action>"
        async with asyncio.timeout(1):
            await run_agent(history, 5, replies(bash, EXIT_DONE), tool, events.append)

        assert tool_log(events) == [
            ("start", "slow"),
            ("start", "fast"),
            ("end", "fast"),  # finish order ...
            ("end", "slow"),
        ]
        observations = [format_observation(ok("slow")), format_observation(ok("fast"))]
        assert history[3] == user("\n\n".join(observations))  # ... command order

    async def test_llm_error_propagates_and_the_failed_turn_adds_nothing(
        self, history: list[Message]
    ) -> None:
        """Turn 1 is kept; turn 2 fails before appending, so a retry starts clean."""
        call = replies(BASH_LS, LlmError("model down"))

        with pytest.raises(LlmError, match="model down"):
            await run_agent(history, 5, call, ok_tool)

        assert history[2:] == [assistant(BASH_LS), user(LS_OBSERVATION)]

    async def test_cancel_during_tools_ends_every_row_and_appends_nothing(
        self, history: list[Message], events: list[Event]
    ) -> None:
        """A cancel must not leave a <bash> request without observations in history."""
        before = list(history)
        all_running = asyncio.Barrier(3)  # both tools and this test

        async def hang(_cmd: str) -> BashResult:
            await all_running.wait()
            await asyncio.Event().wait()  # never set
            raise AssertionError("unreachable")

        bash = "<action><bash>a</bash><bash>b</bash></action>"
        task = asyncio.create_task(
            run_agent(history, 5, replies(bash), hang, events.append)
        )
        async with asyncio.timeout(1):
            await all_running.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert sorted(tool_log(events)) == [
            ("cancelled", "a"),
            ("cancelled", "b"),
            ("start", "a"),
            ("start", "b"),
        ]
        assert history == before


@pytest.mark.parametrize("name", ["AGENTS.md", "agents.md", "Agents.MD"])
def test_load_context_matches_any_casing(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("be nice\n")
    ctx = load_context(tmp_path)
    assert ctx == Context(tmp_path / name, "be nice")


@pytest.mark.parametrize("content", [None, "", "  \n"])
def test_load_context_absent_or_empty(tmp_path: Path, content: str | None) -> None:
    if content is not None:
        (tmp_path / "AGENTS.md").write_text(content)
    assert load_context(tmp_path) is None


@pytest.mark.parametrize("with_context", [False, True])
def test_system_prompt_includes_context(tmp_path: Path, with_context: bool) -> None:
    path = tmp_path / "AGENTS.md"
    prompt = build_system_prompt(Context(path, "be nice") if with_context else None)
    assert prompt.startswith(SYSTEM_PROMPT)
    assert (f"## {path}" in prompt and "be nice" in prompt) == with_context


@pytest.mark.parametrize(
    ("argv", "expected"), [([], False), (["-nc"], True), (["--no-context"], True)]
)
def test_no_context_flag(argv: list[str], expected: bool) -> None:
    assert parse_args(argv).no_context is expected
