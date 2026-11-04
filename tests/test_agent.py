"""Regression tests for the agent's XML action parsing and the LLM call."""

import json
from collections.abc import Callable

import httpx2
import pytest
from httpx2 import AsyncClient

from koding_agent.core import LlmError, Message, llm_call
from koding_agent.core.protocol import MIXED_ACTIONS_ERROR, parse_action, parse_xml


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
        reply = await llm_call(client, "some-model", messages)

    assert reply == "<exit>hi</exit>"
    assert requests[0].url.path == "/api/chat"
    assert json.loads(requests[0].content) == {
        "model": "some-model",  # the model passed in, not a global
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
            await llm_call(client, "some-model", [])
