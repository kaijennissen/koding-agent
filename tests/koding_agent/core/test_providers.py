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
from koding_agent.core.providers import (
    HttpProvider,
    OllamaProvider,
    OpenrouterProvider,
)


def ollama(base_url: str | None) -> OllamaProvider:
    return OllamaProvider.from_config(
        ProviderConfig(provider="ollama", model=None, base_url=base_url)
    )


def openrouter(*, model: str | None, key_var: str | None = None) -> OpenrouterProvider:
    return OpenrouterProvider.from_config(
        ProviderConfig(provider="openrouter", model=model, api_key_env_var=key_var)
    )


def test_ollama_base_url_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OllamaProvider.BASE_URL_ENV, raising=False)
    assert ollama(None).base_url == OllamaProvider.DEFAULT_BASE_URL
    monkeypatch.setenv(OllamaProvider.BASE_URL_ENV, "http://env:1")
    assert ollama(None).base_url == "http://env:1"
    assert ollama("http://flag:2").base_url == "http://flag:2"


def test_ollama_has_a_default_model() -> None:
    assert ollama(None).model


def test_openrouter_requires_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OpenrouterProvider.API_KEY_ENV, "k")
    with pytest.raises(LlmError, match="--model"):
        openrouter(model=None)


def test_openrouter_requires_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OpenrouterProvider.API_KEY_ENV, raising=False)
    with pytest.raises(LlmError, match=OpenrouterProvider.API_KEY_ENV):
        openrouter(model="vendor/model")


def test_openrouter_reads_a_custom_key_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_KEY", "secret")
    provider = openrouter(model="vendor/model", key_var="MY_KEY")
    assert provider.client.headers["Authorization"] == "Bearer secret"


async def test_open_provider_rejects_unknown_names() -> None:
    with pytest.raises(LlmError, match="Unknown provider"):
        async with open_provider(ProviderConfig(provider="nope", model="m")):
            pass


async def test_unexpected_setup_errors_become_llm_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_: object, **__: object) -> None:
        raise ValueError("bad url")

    monkeypatch.setattr(OllamaProvider, "__init__", boom)
    with pytest.raises(LlmError, match="bad url"):
        async with open_provider(ProviderConfig(provider="ollama", model="m")):
            pass


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
        await providers.post_json(
            client=client, endpoint="/api/chat", json={"model": "m"}
        )

    assert seen[0].method == "POST"
    assert seen[0].url.path == "/api/chat"


async def test_post_json_sends_the_payload_as_the_json_body() -> None:
    payload = {"model": "m", "stream": False, "messages": MESSAGES}
    client, seen = recording_client(Response(200, json={}))
    async with client:
        await providers.post_json(client=client, endpoint="/api/chat", json=payload)

    assert json.loads(seen[0].content) == payload


@pytest.mark.parametrize("status", [200, 404, 500])
async def test_post_json_returns_the_response_untouched(status: int) -> None:
    client, _ = recording_client(Response(status, json={"a": 1}))
    async with client:
        result = await providers.post_json(
            client=client, endpoint="/e", json={"model": "m"}
        )

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
            await providers.post_json(client=client, endpoint="/e", json={"model": "m"})

    assert info.value.__cause__ is exc


async def test_post_json_does_not_swallow_unrelated_errors() -> None:
    def handler(_: httpx2.Request) -> Response:
        raise RuntimeError("bug")

    async with fake_client(handler) as client:
        with pytest.raises(RuntimeError, match="bug"):
            await providers.post_json(client=client, endpoint="/e", json={"model": "m"})


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


# --- wiring: each provider runs post -> check -> extract with its own values -----

type AnyProvider = OllamaProvider | OpenrouterProvider
type MakeProvider = Callable[[AsyncClient], AnyProvider]

OLLAMA_REPLY: dict[str, object] = {"message": {"content": "hello"}}
OPENROUTER_REPLY: dict[str, object] = {"choices": [{"message": {"content": "hello"}}]}


def make_ollama(client: AsyncClient) -> AnyProvider:
    return OllamaProvider(base_url="http://x", model="m", timeout=1, client=client)


def make_openrouter(client: AsyncClient) -> AnyProvider:
    return OpenrouterProvider(
        base_url="http://x", model="v/m", timeout=1, client=client
    )


@pytest.mark.parametrize(
    ("make", "endpoint", "reply_body"),
    [
        (make_ollama, "/api/chat", OLLAMA_REPLY),
        (make_openrouter, "/api/v1/chat/completions", OPENROUTER_REPLY),
    ],
    ids=["ollama", "openrouter"],
)
async def test_complete_uses_its_endpoint_and_reply_path(
    make: MakeProvider, endpoint: str, reply_body: dict[str, object]
) -> None:
    client, seen = recording_client(Response(200, json=reply_body))
    async with client:
        reply = await make(client).complete(MESSAGES)

    assert seen[0].url.path == endpoint
    assert reply == "hello"


@pytest.mark.parametrize(
    ("make", "reply_body"),
    [(make_ollama, OLLAMA_REPLY), (make_openrouter, OPENROUTER_REPLY)],
    ids=["ollama", "openrouter"],
)
async def test_complete_sends_the_resolved_model_and_history(
    make: MakeProvider, reply_body: dict[str, object]
) -> None:
    client, seen = recording_client(Response(200, json=reply_body))
    async with client:
        provider = make(client)
        await provider.complete(MESSAGES)

    assert json.loads(seen[0].content) == {
        "model": provider.model,
        "stream": False,
        "messages": MESSAGES,
    }


@pytest.mark.parametrize(
    "make", [make_ollama, make_openrouter], ids=["ollama", "openrouter"]
)
async def test_complete_checks_the_status_first(make: MakeProvider) -> None:
    """A 401 must say 'request failed', not 'unexpected reply'."""
    client, _ = recording_client(Response(401, json={"error": {"message": "bad key"}}))
    async with client:
        with pytest.raises(LlmError) as info:
            await make(client).complete(MESSAGES)

    assert str(info.value) == "Model request failed (401): bad key"


# --- class variables and the subclass check --------------------------------------


def test_ollama_default_model_is_a_class_variable() -> None:
    assert ollama(None).model == OllamaProvider.DEFAULT_MODEL


def test_openrouter_has_no_default_model() -> None:
    assert OpenrouterProvider.DEFAULT_MODEL is None


@pytest.mark.parametrize("missing", ["_ENDPOINT", "_REPLY_PATH", "DEFAULT_BASE_URL"])
def test_a_provider_must_define_its_class_variables(missing: str) -> None:
    attrs: dict[str, object] = {
        "_ENDPOINT": "/e",
        "_REPLY_PATH": ("a",),
        "DEFAULT_BASE_URL": "http://x",
        "from_config": classmethod(lambda cls, cfg: cls()),  # pyright: ignore
    }
    del attrs[missing]

    with pytest.raises(TypeError, match=missing):
        type("Broken", (HttpProvider,), attrs)


def test_an_abstract_intermediate_base_is_not_checked() -> None:
    type("Layer", (HttpProvider,), {})  # still abstract: no from_config


def test_openrouter_base_url_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OpenrouterProvider.API_KEY_ENV, "k")
    default = openrouter(model="v/m")
    assert default.base_url == OpenrouterProvider.DEFAULT_BASE_URL

    cfg = ProviderConfig(provider="openrouter", model="v/m", base_url="http://flag:3")
    assert OpenrouterProvider.from_config(cfg).base_url == "http://flag:3"
