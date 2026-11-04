"""run_agent: the act -> observe -> decide loop, driven by a fake model and fake tools."""

import asyncio
import os
from pathlib import Path

import pytest

from fakes import replies
from koding_agent.agent import (
    MIXED_ACTIONS_ERROR,
    BashResult,
    Event,
    LlmError,
    Message,
    Skill,
    SkillLibrary,
    SkillResult,
    ToolCancelled,
    ToolEnd,
    ToolStart,
    TurnStart,
    build_system_prompt,
    format_bash_result,
    format_skill_result,
    run_agent,
)

BASH_ECHO = "<action><bash>echo hi</bash></action>"
EXIT_DONE = "<action><exit>done</exit></action>"
MIXED = "<action><bash>echo hi</bash><exit>done</exit></action>"
ECHO_OBSERVATION = format_bash_result(BashResult.completed("echo hi", "hi\n", "", 0))

PDF = Skill("pdf", "Work with PDFs", "Use pdftotext.", Path("/skills/pdf"))
SKILLS = SkillLibrary({"pdf": PDF})
NO_SKILLS = SkillLibrary({})


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
                log.append(("start", call.subject))
            case ToolEnd(call, _):
                log.append(("end", call.subject))
            case ToolCancelled(call):
                log.append(("cancelled", call.subject))
            case TurnStart():
                pass
    return log


@pytest.fixture
def history() -> list[Message]:
    return [{"role": "system", "content": build_system_prompt()}, user("task")]


class TestRunAgent:
    @pytest.mark.parametrize(
        "script,max_turns,answer,appended",
        [
            pytest.param(
                [EXIT_DONE], 5, "done", [assistant(EXIT_DONE)], id="exit-at-once"
            ),
            pytest.param(
                [BASH_ECHO, EXIT_DONE],
                5,
                "done",
                [assistant(BASH_ECHO), user(ECHO_OBSERVATION), assistant(EXIT_DONE)],
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
                [BASH_ECHO, BASH_ECHO],
                2,
                "Ran out of turns (2) without an <exit> action.",
                [assistant(BASH_ECHO), user(ECHO_OBSERVATION)] * 2,
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
            history, max_turns, replies(*script), NO_SKILLS
        )

        assert result == answer
        assert messages == before + appended

    async def test_tools_run_concurrently_and_observations_keep_command_order(
        self, history: list[Message], events: list[Event], tmp_path: Path
    ) -> None:
        """'slow' only finishes once 'fast' has run, so a sequential loop would hang."""
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)
        slow = f"cat {fifo} && sleep 0.2"  # blocks until the fast one writes
        fast = f"echo go > {fifo}"

        bash = f"<action><bash>{slow}</bash><bash>{fast}</bash></action>"
        async with asyncio.timeout(5):
            await run_agent(
                history, 5, replies(bash, EXIT_DONE), NO_SKILLS, events.append
            )

        assert tool_log(events) == [
            ("start", slow),
            ("start", fast),
            ("end", fast),  # finish order ...
            ("end", slow),
        ]
        observations = [
            format_bash_result(BashResult.completed(slow, "go\n", "", 0)),
            format_bash_result(BashResult.completed(fast, "", "", 0)),
        ]
        assert history[3] == user("\n\n".join(observations))  # ... command order

    async def test_command_timeout_is_enforced(self, history: list[Message]) -> None:
        bash = "<action><bash>sleep 5</bash></action>"
        expected = format_bash_result(BashResult.timeout("sleep 5"))

        async with asyncio.timeout(2):
            await run_agent(
                history,
                5,
                replies(bash, EXIT_DONE),
                NO_SKILLS,
                command_timeout=0.1,
            )

        assert history[3] == user(expected)

    async def test_llm_error_propagates_and_the_failed_turn_adds_nothing(
        self, history: list[Message]
    ) -> None:
        """Turn 1 is kept; turn 2 fails before appending, so a retry starts clean."""
        call = replies(BASH_ECHO, LlmError("model down"))

        with pytest.raises(LlmError, match="model down"):
            await run_agent(history, 5, call, NO_SKILLS)

        assert history[2:] == [assistant(BASH_ECHO), user(ECHO_OBSERVATION)]

    async def test_cancel_during_tools_ends_every_row_and_appends_nothing(
        self, history: list[Message], events: list[Event]
    ) -> None:
        """A cancel must not leave a <bash> request without observations in history."""
        before = list(history)
        a, b = "sleep 30", "sleep 31"

        bash = f"<action><bash>{a}</bash><bash>{b}</bash></action>"
        task = asyncio.create_task(
            run_agent(history, 5, replies(bash), NO_SKILLS, events.append)
        )
        async with asyncio.timeout(1):
            while len(tool_log(events)) < 2:  # both tools have started
                await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        # gather re-raises as soon as the first tool is cancelled; the other may
        # still be killing its process, so its end event can arrive a tick later.
        async with asyncio.timeout(1):
            while len(tool_log(events)) < 4:
                await asyncio.sleep(0.01)
        assert sorted(tool_log(events)) == [
            ("cancelled", a),
            ("cancelled", b),
            ("start", a),
            ("start", b),
        ]
        assert history == before


class TestSkillTurn:
    async def test_skill_then_exit(self, history: list[Message]) -> None:
        skill = "<action><skill>pdf</skill></action>"
        expected = format_skill_result(SkillResult("pdf", PDF, ("pdf",)))

        await run_agent(history, 5, replies(skill, EXIT_DONE), SKILLS)

        assert history[2:] == [assistant(skill), user(expected), assistant(EXIT_DONE)]

    async def test_unknown_skill_is_fed_back_with_the_available_names(
        self, history: list[Message], events: list[Event]
    ) -> None:
        skill = "<action><skill>nope</skill></action>"

        await run_agent(history, 5, replies(skill, EXIT_DONE), SKILLS, events.append)

        assert history[3] == user("Unknown skill `nope`. Available: pdf")
        assert [e.result.status for e in events if isinstance(e, ToolEnd)] == ["failed"]
