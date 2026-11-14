"""Provider defaults and early configuration errors."""

import json
from collections.abc import Callable

import httpx2
import pytest
from httpx2 import AsyncClient, Response

from koding_agent.core import (
    LlmError,
    Message,
    ProviderConfig,
    open_provider,
    providers,
)
from koding_agent.core.providers import OllamaProvider, OpenrouterProvider


def test_ollama_base_url_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert OllamaProvider(None, None, None).base_url == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_HOST", "http://env:1")
    assert OllamaProvider(None, None, None).base_url == "http://env:1"
    assert OllamaProvider("http://flag:2", None, None).base_url == "http://flag:2"


def test_ollama_has_a_default_model() -> None:
    assert OllamaProvider(None, None, None).model


def test_openrouter_requires_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with pytest.raises(LlmError, match="--model"):
        OpenrouterProvider(None, None, None, None)


def test_openrouter_requires_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LlmError, match="OPENROUTER_API_KEY"):
        OpenrouterProvider(None, "vendor/model", None, None)


def test_openrouter_reads_a_custom_key_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_KEY", "secret")
    provider = OpenrouterProvider(None, "vendor/model", None, "MY_KEY")
    assert provider.client.headers["Authorization"] == "Bearer secret"


async def test_open_provider_rejects_unknown_names() -> None:
    with pytest.raises(LlmError, match="Unknown provider"):
        async with open_provider(ProviderConfig(provider="nope", model="m")):
            pass


def test_unexpected_setup_errors_become_llm_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_: object, **__: object) -> None:
        raise ValueError("bad url")

    monkeypatch.setattr(OllamaProvider, "__init__", boom)
    with pytest.raises(LlmError, match="bad url"):
        OllamaProvider.from_config(ProviderConfig(provider="ollama", model="m"))


# --- post_json: transport only, never interprets the response --------------------

type Handler = Callable[[httpx2.Request], Response]

MESSAGES: list[Message] = [{"role": "user", "content": "hi"}]


def fake_client(handler: Handler) -> AsyncClient:
    """An AsyncClient on a MockTransport that routes every request to `handler`."""
    return AsyncClient(transport=httpx2.MockTransport(handler), base_url="http://x")


def recording_client(reply: Response) -> tuple[AsyncClient, list[httpx2.Request]]:
    """A client answering every request with `reply`, plus the list of requests seen."""
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> Response:
        seen.append(request)
        return reply

    return fake_client(handler), seen


async def test_post_json_posts_to_the_endpoint() -> None:
    client, seen = recording_client(Response(200, json={}))
    async with client:
        await providers.post_json(client, "/api/chat", {"model": "m"})

    assert seen[0].method == "POST"
    assert seen[0].url.path == "/api/chat"


async def test_post_json_sends_the_payload_as_the_json_body() -> None:
    payload = {"model": "m", "stream": False, "messages": MESSAGES}
    client, seen = recording_client(Response(200, json={}))
    async with client:
        await providers.post_json(client, "/api/chat", payload)

    assert json.loads(seen[0].content) == payload


@pytest.mark.parametrize("status", [200, 404, 500])
async def test_post_json_returns_the_response_untouched(status: int) -> None:
    client, _ = recording_client(Response(status, json={"a": 1}))
    async with client:
        result = await providers.post_json(client, "/e", {"model": "m"})

    assert isinstance(result, httpx2.Response)
    assert result.status_code == status
    assert result.json() == {"a": 1}


@pytest.mark.parametrize(
    "exc",
    [
        httpx2.ConnectError("refused"),
        httpx2.ReadTimeout("slow"),
        httpx2.HTTPError("other"),
    ],
    ids=["connect", "timeout", "generic"],
)
async def test_post_json_wraps_transport_errors(exc: httpx2.HTTPError) -> None:
    def handler(_: httpx2.Request) -> Response:
        raise exc

    async with fake_client(handler) as client:
        with pytest.raises(LlmError, match="Could not reach the model") as info:
            await providers.post_json(client, "/e", {"model": "m"})

    assert info.value.__cause__ is exc


async def test_post_json_does_not_swallow_unrelated_errors() -> None:
    def handler(_: httpx2.Request) -> Response:
        raise RuntimeError("bug")

    async with fake_client(handler) as client:
        with pytest.raises(RuntimeError, match="bug"):
            await providers.post_json(client, "/e", {"model": "m"})


# --- raise_for_error: the error path ---------------------------------------------


@pytest.mark.parametrize("status", [200, 201, 204])
def test_raise_for_error_ignores_success(status: int) -> None:
    # The body claims an error on purpose: only the status may decide.
    providers.raise_for_error(Response(status, json={"error": "x"}))


@pytest.mark.parametrize("status", [400, 401, 404, 429, 500, 503])
def test_raise_for_error_raises_for_error_statuses(status: int) -> None:
    with pytest.raises(LlmError) as info:
        providers.raise_for_error(Response(status, json={"error": "x"}))

    assert f"({status})" in str(info.value)


@pytest.mark.parametrize(
    ("response", "detail"),
    [
        (Response(401, json={"error": "bad key"}), "bad key"),
        (
            Response(401, json={"error": {"message": "bad key", "code": 401}}),
            "bad key",
        ),
        (Response(502, text="Bad Gateway"), "Bad Gateway"),
    ],
    ids=["string", "dict-with-message", "non-json"],
)
def test_raise_for_error_reports_the_detail(response: Response, detail: str) -> None:
    with pytest.raises(LlmError) as info:
        providers.raise_for_error(response)

    # Equality, not `in`: a printed dict would still contain "bad key".
    expected = f"Model request failed ({response.status_code}): {detail}"
    assert str(info.value) == expected


@pytest.mark.parametrize(
    "response",
    [
        Response(500, json={"detail": "x"}),
        Response(500, json=["x"]),
        Response(500, content=b"null"),
        Response(500, content=b"<html>oops</html>"),
        Response(400, json={"error": {"code": 400}}),
    ],
    ids=["no-error-key", "list", "json-null", "html-body", "dict-without-message"],
)
def test_raise_for_error_falls_back_to_the_raw_text(response: Response) -> None:
    with pytest.raises(LlmError) as info:  # and nothing but LlmError
        providers.raise_for_error(response)

    expected = f"Model request failed ({response.status_code}): {response.text}"
    assert str(info.value) == expected


# --- reply_text: the success path ------------------------------------------------

PATH_OLLAMA: tuple[str | int, ...] = ("message", "content")
PATH_OPENROUTER: tuple[str | int, ...] = ("choices", 0, "message", "content")


@pytest.mark.parametrize(
    ("body", "path"),
    [
        ({"message": {"content": "hello"}}, PATH_OLLAMA),
        ({"choices": [{"message": {"content": "hello"}}]}, PATH_OPENROUTER),
    ],
    ids=["ollama", "openrouter"],
)
def test_reply_text_returns_the_text_at_the_path(
    body: dict[str, object], path: tuple[str | int, ...]
) -> None:
    assert providers.reply_text(Response(200, json=body), *path) == "hello"


def test_reply_text_passes_an_empty_reply_through() -> None:
    response = Response(200, json={"message": {"content": ""}})

    assert providers.reply_text(response, *PATH_OLLAMA) == ""


@pytest.mark.parametrize(
    "response",
    [
        Response(200, text="not json"),
        Response(200, json={"choices": [{"message": {}}]}),
        Response(200, json={"choices": []}),
        Response(200, json={"choices": None}),
        Response(200, json=["x"]),
    ],
    ids=["non-json", "missing-key", "empty-list", "null-container", "list-body"],
)
def test_reply_text_rejects_a_body_without_the_path(response: Response) -> None:
    # Always the OpenRouter path, so the index step is reached.
    with pytest.raises(LlmError, match="Unexpected reply from the model") as info:
        providers.reply_text(response, *PATH_OPENROUTER)

    assert response.text in str(info.value)


@pytest.mark.parametrize(
    "content", [None, 5, ["a"], {"a": 1}], ids=["null", "int", "list", "dict"]
)
def test_reply_text_rejects_non_string_content(content: object) -> None:
    response = Response(200, json={"choices": [{"message": {"content": content}}]})

    with pytest.raises(LlmError, match="Unexpected reply from the model"):
        providers.reply_text(response, *PATH_OPENROUTER)


@pytest.mark.parametrize(
    ("response", "cause"),
    [
        (Response(200, json={}), KeyError),
        (Response(200, json={"choices": []}), IndexError),
        (Response(200, json={"choices": None}), TypeError),
        (Response(200, text="nope"), ValueError),
    ],
    ids=["key", "index", "type", "value"],
)
def test_reply_text_chains_the_cause(
    response: Response, cause: type[Exception]
) -> None:
    with pytest.raises(LlmError) as info:
        providers.reply_text(response, *PATH_OPENROUTER)

    assert isinstance(info.value.__cause__, cause)
