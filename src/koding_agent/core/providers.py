"""Message and call types, and the Ollama client."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypedDict, cast

import httpx2
from httpx2 import AsyncClient

DEFAULT_MODEL = "qwen3.8:27b-mlx"


class Message(TypedDict):
    role: str
    content: str


type LlmCall = Callable[[list[Message]], Awaitable[str]]  # history -> reply


class LlmError(Exception):
    """The model gave no usable reply. The message is meant for the user.

    Front ends catch this one type and keep the session alive; the history is
    unchanged by the failed call, so the same run can simply be retried.
    """


@dataclass
class ProviderConfig:
    provider: str
    model: str
    base_url: str = "http://localhost:11434"


class ProviderRegistryFactory:
    _providers: dict = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(wrpd_cls):
            if name in cls._providers:
                raise ValueError(f"{name} is already registered.")
            cls._providers[name] = wrpd_cls
            return wrpd_cls

        return wrapper

    @classmethod
    def from_config(cls, cfg):
        if cfg.provider not in cls._providers:
            raise ValueError("No such provider: {cfg.provider}")

        return cls._providers[cfg.provider].from_config(cfg=cfg)


_PROVIDERS = {}


def register(key):
    def decorator(cls):
        _PROVIDERS[key] = cls

        def wrapper(*args, **kwargs):
            cls(*args, **kwargs)

        return cls

    return decorator


print(_PROVIDERS)


@ProviderRegistryFactory.register(name="ollama")
class OllamaProvider:
    def __init__(
        self,
        model: str | None = None,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.model = model

    def get_client(self):
        return AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def call_llm(self, client, messages: list[Message], model: str | None = None):
        return await call_ollama(
            client=client, model=self.model or model, messages=messages
        )

    @classmethod
    def from_config(cls, cfg):
        return OllamaProvider(model=cfg.model, base_url=cfg.base_url)


print(_PROVIDERS)


async def call_ollama(client: AsyncClient, model: str, messages: list[Message]) -> str:
    """Ask Ollama for the next reply; every failure is raised as LlmError."""
    try:
        response = await client.post(
            "/api/chat", json={"model": model, "stream": False, "messages": messages}
        )
    except httpx2.HTTPError as e:
        raise LlmError(f"Could not reach the model: {e}") from e

    if response.is_error:
        try:
            detail = response.json()["error"]  # Ollama puts its reason here
        except ValueError, KeyError, TypeError:
            detail = response.text
        raise LlmError(f"Model request failed ({response.status_code}): {detail}")

    try:
        return cast(str, response.json()["message"]["content"])
    except (ValueError, KeyError, TypeError) as e:
        raise LlmError(f"Unexpected reply from the model: {response.text}") from e


async def call_openrouter(
    client: AsyncClient, model: str, messages: list[Message]
) -> str:
    """Ask Ollama for the next reply; every failure is raised as LlmError."""
    try:
        response = await client.post(
            "/api/v1/chat/completions",
            json={"model": model, "stream": False, "messages": messages},
        )
    except httpx2.HTTPError as e:
        raise LlmError(f"Could not reach the model: {e}") from e

    if response.is_error:
        try:
            detail = response.json()["error"]  # Ollama puts its reason here
        except ValueError, KeyError, TypeError:
            detail = response.text
        raise LlmError(f"Model request failed ({response.status_code}): {detail}")

    try:
        return cast(str, response.json()["choices"][0]["message"]["content"])
    except (ValueError, KeyError, TypeError) as e:
        raise LlmError(f"Unexpected reply from the model: {response.text}") from e


async def main():
    msgs = [{"role": "user", "content": "Why is the sky blue?"}]
    cfg = ProviderConfig(provider="ollama", model="qwen3.8:27b-dflash2-nvfp4")

    provider = ProviderRegistryFactory.from_config(cfg=cfg)
    async with provider.get_client() as client:
        msg = await provider.call_llm(client=client, messages=msgs)
    print(msg)
    model = "qwen3.8:27b-mlx"
    async with provider.get_client() as client:
        msg = await provider.call_llm(client=client, model=model, messages=msgs)
    print(msg)


if __name__ == "__main__":
    asyncio.run(main())
