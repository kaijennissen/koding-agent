"""The Ollama client: request shape, and failures raised as LlmError."""

import json
from collections.abc import Callable

import httpx2
import pytest
from httpx2 import AsyncClient

from koding_agent.core import LlmError, Message, llm_call


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
