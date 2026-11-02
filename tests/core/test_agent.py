"""Regression tests for the agent's XML action parsing and the LLM call."""

import asyncio
import json
from collections.abc import Callable

import httpx2
import pytest
from httpx2 import AsyncClient

from fakes import ok, ok_tool, replies
from koding_agent.core.agent import (
    MIXED_ACTIONS_ERROR,
    MODEL,
    SYSTEM_PROMPT,
    LlmError,
    Message,
    llm_call,
    parse_action,
    parse_xml,
    run_agent,
)
from koding_agent.core.events import Event, ToolCancelled, ToolEnd, ToolStart, TurnStart
from koding_agent.core.tools import BashResult, format_observation


@pytest.mark.parametrize(
    "tag,text,expected",
    [
        pytest.param("bash", "<bash> ls -alh </bash>", ["ls -alh"], id="strips-edges"),
        pytest.param(
            "bash",
            "<action><bash> ls </bash><bash>pwd</bash></action>",
            ["ls", "pwd"],
            id="all-in-order",
        ),
        pytest.param("exit", "<exit> Finished! </exit>", ["Finished!"], id="exit-tag"),
        pytest.param(
            "action",
            "<action><bash> ls </bash><bash>pwd</bash></action>",
            ["<bash> ls </bash><bash>pwd</bash>"],
            id="outer-tag-keeps-inner-xml",
        ),
        pytest.param(
            "bash",
            "<bash>\n  cd src &&\n  ls\n</bash>",
            ["cd src &&\n  ls"],
            id="multiline-keeps-inner-whitespace",
        ),
        pytest.param("bash", "<exit>done</exit>", [], id="other-tag-only"),
        pytest.param("bash", "<bash>ls", [], id="unclosed"),
        # pins current behaviour, not a wish
        pytest.param("bash", "<bash></bash>", [""], id="empty-body"),
    ],
)
def test_parse_xml(tag: str, text: str, expected: list[str]) -> None:
    """Contents come back in order of appearance, stripped at the edges only."""
    assert parse_xml(tag, text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        pytest.param(
            "<bash>ls</bash><bash>pwd</bash>", {"bash": ["ls", "pwd"]}, id="bash"
        ),
        pytest.param("<exit> done </exit>", {"exit": ["done"]}, id="exit"),
        pytest.param(
            "<unknown> Finished! </unknown>",
            {"error": ["Couldn't parse any action. Please retry!"]},
            id="no-tag-errors",
        ),
        pytest.param(
            "Sure!\n<action><bash>ls</bash></action>\nHope it helps.",
            {"bash": ["ls"]},
            id="ignores-chatter",
        ),
        pytest.param(
            "<bash>ls</bash><exit>done</exit>",
            {"error": [MIXED_ACTIONS_ERROR]},
            id="mixed-bash-first",
        ),
        pytest.param(
            "<exit>done</exit><bash>ls</bash>",
            {"error": [MIXED_ACTIONS_ERROR]},
            id="mixed-exit-first",
        ),
    ],
)
def test_parse_action(text: str, expected: dict[str, list[str]]) -> None:
    """One action type per turn; mixing or an unknown reply is an error."""
    assert parse_action(text) == expected


def fake_ollama(handler: Callable[[httpx2.Request], httpx2.Response]) -> AsyncClient:
    return AsyncClient(
        transport=httpx2.MockTransport(handler), base_url="http://ollama"
    )


async def test_llm_call_posts_history_and_returns_reply() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={"message": {"content": "<exit>hi</exit>"}})

    messages: list[Message] = [{"role": "user", "content": "hello"}]
    async with fake_ollama(handler) as client:
        reply = await llm_call(client, messages)

    assert reply == "<exit>hi</exit>"
    assert requests[0].url.path == "/api/chat"
    assert json.loads(requests[0].content) == {
        "model": MODEL,
        "stream": False,
        "messages": messages,
    }


@pytest.mark.parametrize(
    "reply,expected",
    [
        pytest.param(
            httpx2.Response(404, json={"error": "model 'x' not found"}),
            "model 'x' not found",
            id="ollama-error-field",
        ),
        pytest.param(
            httpx2.Response(502, text="Bad Gateway"), "Bad Gateway", id="plain-text-5xx"
        ),
        pytest.param(
            httpx2.ConnectError("All connection attempts failed"),
            "All connection attempts failed",
            id="unreachable",
        ),
        pytest.param(
            httpx2.Response(200, json={"unexpected": "shape"}),
            "Unexpected reply",
            id="malformed-reply",
        ),
    ],
)
async def test_llm_call_turns_failures_into_llm_error(
    reply: httpx2.Response | Exception, expected: str
) -> None:
    """Front ends catch one error type, and its message is readable for the user."""

    def handler(_: httpx2.Request) -> httpx2.Response:
        if isinstance(reply, Exception):
            raise reply
        return reply

    async with fake_ollama(handler) as client:
        with pytest.raises(LlmError, match=expected):
            await llm_call(client, [])


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
