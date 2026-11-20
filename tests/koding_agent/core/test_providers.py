"""Provider configuration, request wiring and the shared HTTP helpers."""

import json
import re
from collections.abc import Callable
from typing import Any

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
from koding_agent.core.providers import (
    HttpProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
)

type Handler = Callable[[httpx2.Request], Response]
type AnyProvider = OllamaProvider | OpenAICompatibleProvider
type MakeProvider = Callable[[AsyncClient], AnyProvider]

MESSAGES: list[Message] = [{"role": "user", "content": "hi"}]
OLLAMA_REPLY: dict[str, object] = {"message": {"content": "hello"}}
OPENAI_REPLY: dict[str, object] = {"choices": [{"message": {"content": "hello"}}]}

# (provider name, its API key variable): the vendors' documented values, written
# out here on purpose. They are the spec the presets must meet.
PRESETS = [
    pytest.param("openai", "OPENAI_API_KEY", id="openai"),
    pytest.param("openrouter", "OPENROUTER_API_KEY", id="openrouter"),
]


def exactly(message: str) -> str:
    """A `match` pattern for the whole message, not a substring of it."""
    return f"^{re.escape(message)}$"


def config(
    provider: str,
    *,
    model: str | None = "m",
    base_url: str | None = None,
    timeout: int | None = None,
    api_key_env_var: str | None = None,
) -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        timeout=timeout,
        api_key_env_var=api_key_env_var,
    )


def build(provider: str, **fields: Any) -> AnyProvider:
    """Build a provider through its `from_config`, as the CLI does."""
    cfg = config(provider, **fields)
    if provider == "ollama":
        return OllamaProvider.from_config(cfg)
    return OpenAICompatibleProvider.from_config(cfg)


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Known credentials and no host override, whatever the real environment."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("MY_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> list[httpx2.Request]:
    """Requests made by providers built from config, answered by a fake server.

    Every `AsyncClient` the providers module creates gets a MockTransport, so
    headers and URLs can be checked on the request that was actually sent.
    """
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> Response:
        requests.append(request)
        if request.url.path.endswith("/api/chat"):
            return Response(200, json=OLLAMA_REPLY)
        if request.url.path.endswith("/chat/completions"):
            return Response(200, json=OPENAI_REPLY)
        return Response(404)

    def client_with_fake_server(**kwargs: Any) -> AsyncClient:
        return AsyncClient(transport=httpx2.MockTransport(handler), **kwargs)

    monkeypatch.setattr(providers, "AsyncClient", client_with_fake_server)
    return requests


async def chat(provider: str, **fields: Any) -> str:
    """One round trip through `open_provider`, as a front end would do it."""
    async with open_provider(config(provider, **fields)) as p:
        return await p.complete(MESSAGES)


# --- configuration: defaults, precedence and early errors ------------------------


@pytest.mark.parametrize(
    ("env_host", "flag", "expected"),
    [
        (None, None, "http://localhost:11434"),
        ("http://env:1", None, "http://env:1"),
        ("http://env:1", "http://flag:2", "http://flag:2"),
        (None, "http://flag:2", "http://flag:2"),
    ],
    ids=["default", "env", "flag-over-env", "flag"],
)
def test_ollama_base_url_precedence(
    monkeypatch: pytest.MonkeyPatch,
    env_host: str | None,
    flag: str | None,
    expected: str,
) -> None:
    if env_host:
        monkeypatch.setenv("OLLAMA_HOST", env_host)

    assert build("ollama", base_url=flag).base_url == expected


@pytest.mark.parametrize(
    ("name", "flag", "expected"),
    [
        ("openai", None, "https://api.openai.com/v1"),
        ("openrouter", None, "https://eu.openrouter.ai/api/v1"),
        ("openai", "http://flag:3/v1", "http://flag:3/v1"),
        ("openrouter", "http://flag:3/v1", "http://flag:3/v1"),
    ],
    ids=["openai", "openrouter", "openai-flag", "openrouter-flag"],
)
def test_openai_compatible_base_url_is_the_api_root(
    name: str, flag: str | None, expected: str
) -> None:
    assert build(name, base_url=flag).base_url == expected


@pytest.mark.parametrize("name", ["ollama", "openai", "openrouter"])
@pytest.mark.parametrize(
    ("timeout", "expected"), [(None, 120), (0, 0), (5, 5)], ids=["default", "0", "5"]
)
def test_timeout_defaults_to_120_seconds(
    name: str, timeout: int | None, expected: int
) -> None:
    assert build(name, timeout=timeout).timeout == expected


@pytest.mark.parametrize("name", ["openai", "openrouter"])
def test_openai_compatible_requires_a_model(name: str) -> None:
    with pytest.raises(
        LlmError, match=exactly(f"{name} has no default model; pass --model")
    ):
        build(name, model=None)


@pytest.mark.parametrize(
    ("name", "custom_var", "var", "value"),
    [
        ("openai", None, "OPENAI_API_KEY", None),
        ("openrouter", None, "OPENROUTER_API_KEY", None),
        ("openai", None, "OPENAI_API_KEY", ""),
        ("openai", "MY_KEY", "MY_KEY", None),
    ],
    ids=["openai", "openrouter", "empty", "custom-variable"],
)
def test_openai_compatible_requires_the_key(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    custom_var: str | None,
    var: str,
    value: str | None,
) -> None:
    monkeypatch.delenv(var, raising=False)
    if value is not None:
        monkeypatch.setenv(var, value)

    with pytest.raises(LlmError, match=exactly(f"${var} is not set")):
        build(name, api_key_env_var=custom_var)


def test_openai_compatible_rejects_an_unregistered_provider_name() -> None:
    with pytest.raises(
        LlmError, match=exactly("Unknown OpenAI-compatible provider: nope")
    ):
        build("nope")


# --- open_provider ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["openai", "openrouter"])
async def test_open_provider_serves_the_openai_compatible_presets(name: str) -> None:
    async with open_provider(ProviderConfig(provider=name, model="v/m")) as provider:
        assert provider.model == "v/m"


async def test_open_provider_rejects_unknown_names() -> None:
    with pytest.raises(LlmError, match="Unknown provider: nope"):
        async with open_provider(ProviderConfig(provider="nope", model="m")):
            pass


async def test_unexpected_setup_errors_become_llm_errors() -> None:
    # A port that is not a number makes the real HTTP client refuse to be built.
    cfg = ProviderConfig(provider="ollama", model="m", base_url="http://x:abc")

    with pytest.raises(
        LlmError, match=re.escape("Couldn't set up provider 'ollama': Invalid port")
    ):
        async with open_provider(cfg):
            pass


# --- requests built from config: URL, credentials, payload -----------------------


@pytest.mark.parametrize(
    ("name", "base_url", "url"),
    [
        ("ollama", "http://x", "http://x/api/chat"),
        ("ollama", "http://x/", "http://x/api/chat"),
        ("openai", "http://x/v1", "http://x/v1/chat/completions"),
        ("openai", "http://x/v1/", "http://x/v1/chat/completions"),
    ],
    ids=["ollama", "ollama-slash", "openai", "openai-slash"],
)
async def test_complete_uses_its_endpoint_and_reply_path(
    seen: list[httpx2.Request], name: str, base_url: str, url: str
) -> None:
    reply = await chat(name, base_url=base_url)

    assert (str(seen[0].url), reply) == (url, "hello")


@pytest.mark.parametrize(("name", "model"), [("ollama", "m"), ("openai", "v/m")])
async def test_complete_sends_the_model_and_history(
    seen: list[httpx2.Request], name: str, model: str
) -> None:
    await chat(name, model=model)

    assert json.loads(seen[0].content) == {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": "hi"}],
    }


@pytest.mark.parametrize(("name", "key_var"), PRESETS)
@pytest.mark.parametrize("custom_var", [None, "MY_KEY"], ids=["default", "custom"])
async def test_openai_compatible_sends_the_key_as_a_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
    seen: list[httpx2.Request],
    name: str,
    key_var: str,
    custom_var: str | None,
) -> None:
    monkeypatch.setenv(custom_var or key_var, "secret")  # the others stay "k"

    await chat(name, api_key_env_var=custom_var)

    assert seen[0].headers["Authorization"] == "Bearer secret"


async def test_ollama_sends_no_credentials(seen: list[httpx2.Request]) -> None:
    await chat("ollama")

    assert "Authorization" not in seen[0].headers


# --- post_json: transport only, never interprets the response --------------------


def fake_client(handler: Handler) -> AsyncClient:
    """An AsyncClient on a MockTransport that routes every request to `handler`."""
    return AsyncClient(transport=httpx2.MockTransport(handler), base_url="http://x")


def recording_client(reply: Response) -> tuple[AsyncClient, list[httpx2.Request]]:
    """A client answering every request with `reply`, plus the requests it saw."""
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> Response:
        requests.append(request)
        return reply

    return fake_client(handler), requests


async def test_post_json_posts_the_payload_as_json_to_the_endpoint() -> None:
    payload = {"model": "m", "stream": False, "messages": MESSAGES}
    client, requests = recording_client(Response(200, json={}))
    async with client:
        await providers.post_json(client=client, endpoint="/api/chat", json=payload)

    request = requests[0]
    assert (request.method, request.url.path, json.loads(request.content)) == (
        "POST",
        "/api/chat",
        payload,
    )


@pytest.mark.parametrize("status", [200, 500])
async def test_post_json_returns_the_response_untouched(status: int) -> None:
    client, _ = recording_client(Response(status, json={"a": 1}))
    async with client:
        result = await providers.post_json(
            client=client, endpoint="/e", json={"model": "m"}
        )

    assert (result.status_code, result.json()) == (status, {"a": 1})


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
            await providers.post_json(client=client, endpoint="/e", json={"model": "m"})

    assert info.value.__cause__ is exc


async def test_post_json_does_not_swallow_unrelated_errors() -> None:
    def handler(_: httpx2.Request) -> Response:
        raise RuntimeError("bug")

    async with fake_client(handler) as client:
        with pytest.raises(RuntimeError, match="bug"):
            await providers.post_json(client=client, endpoint="/e", json={"model": "m"})


# --- raise_for_error: the error path ---------------------------------------------


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (200, None),
        (399, None),
        (400, "Model request failed (400): x"),
        (599, "Model request failed (599): x"),
    ],
)
def test_raise_for_error_only_raises_for_error_statuses(
    status: int, message: str | None
) -> None:
    # The body claims an error on purpose: only the status may decide.
    response = Response(status, json={"error": "x"})
    if message is None:
        providers.raise_for_error(response)
    else:
        with pytest.raises(LlmError, match=exactly(message)):
            providers.raise_for_error(response)


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
    message = f"Model request failed ({response.status_code}): {detail}"

    with pytest.raises(LlmError, match=exactly(message)):
        providers.raise_for_error(response)


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
    message = f"Model request failed ({response.status_code}): {response.text}"

    with pytest.raises(LlmError, match=exactly(message)):  # and nothing but LlmError
        providers.raise_for_error(response)


# --- reply_text: the success path ------------------------------------------------

PATH_OLLAMA: tuple[str | int, ...] = ("message", "content")
PATH_OPENAI: tuple[str | int, ...] = ("choices", 0, "message", "content")


@pytest.mark.parametrize(
    ("body", "path", "text"),
    [
        ({"message": {"content": "hello"}}, PATH_OLLAMA, "hello"),
        ({"choices": [{"message": {"content": "hello"}}]}, PATH_OPENAI, "hello"),
        ({"message": {"content": ""}}, PATH_OLLAMA, ""),
    ],
    ids=["ollama", "openai-compatible", "empty-reply"],
)
def test_reply_text_returns_the_text_at_the_path(
    body: dict[str, object], path: tuple[str | int, ...], text: str
) -> None:
    assert providers.reply_text(Response(200, json=body), *path) == text


@pytest.mark.parametrize(
    ("response", "cause"),
    [
        (Response(200, text="not json"), ValueError),
        (Response(200, json={"choices": [{"message": {}}]}), KeyError),
        (Response(200, json={"choices": []}), IndexError),
        (Response(200, json={"choices": None}), TypeError),
        (Response(200, json=["x"]), TypeError),
    ],
    ids=["non-json", "missing-key", "empty-list", "null-container", "list-body"],
)
def test_reply_text_rejects_a_body_without_the_path(
    response: Response, cause: type[Exception]
) -> None:
    message = f"Unexpected reply from the model: {response.text}"

    with pytest.raises(LlmError, match=exactly(message)) as info:
        providers.reply_text(response, *PATH_OPENAI)

    assert isinstance(info.value.__cause__, cause)


@pytest.mark.parametrize(
    "content", [None, 5, ["a"], {"a": 1}], ids=["null", "int", "list", "dict"]
)
def test_reply_text_rejects_non_string_content(content: object) -> None:
    response = Response(200, json={"choices": [{"message": {"content": content}}]})
    message = f"Unexpected reply from the model: {response.text}"

    with pytest.raises(LlmError, match=exactly(message)):
        providers.reply_text(response, *PATH_OPENAI)


# --- wiring: each provider runs post -> check -> extract -------------------------


def make_ollama(client: AsyncClient) -> AnyProvider:
    return OllamaProvider(base_url="http://x", model="m", timeout=1, client=client)


def make_openai_compatible(client: AsyncClient) -> AnyProvider:
    return OpenAICompatibleProvider(
        base_url="http://x", model="v/m", timeout=1, client=client
    )


@pytest.mark.parametrize(
    "make", [make_ollama, make_openai_compatible], ids=["ollama", "openai-compatible"]
)
async def test_complete_checks_the_status_first(make: MakeProvider) -> None:
    """A 401 must say 'request failed', not 'unexpected reply'."""
    client, _ = recording_client(Response(401, json={"error": {"message": "bad key"}}))
    async with client:
        with pytest.raises(
            LlmError, match=exactly("Model request failed (401): bad key")
        ):
            await make(client).complete(MESSAGES)


# --- the subclass check ----------------------------------------------------------


@pytest.mark.parametrize("missing", ["_ENDPOINT", "_REPLY_PATH"])
def test_a_provider_must_define_its_class_variables(missing: str) -> None:
    attrs: dict[str, object] = {
        "_ENDPOINT": "/e",
        "_REPLY_PATH": ("a",),
        "from_config": classmethod(lambda cls, cfg: cls()),  # pyright: ignore
    }
    del attrs[missing]

    with pytest.raises(TypeError, match=missing):
        type("Broken", (HttpProvider,), attrs)


def test_an_abstract_intermediate_base_is_not_checked() -> None:
    type("Layer", (HttpProvider,), {})  # still abstract: no from_config
