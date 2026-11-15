"""Provider defaults and early configuration errors."""

import pytest

from koding_agent.core import LlmError, ProviderConfig, open_provider
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
