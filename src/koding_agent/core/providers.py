"""Message and call types, and the Ollama client."""

import asyncio
from collections.abc import Awaitable, Callable
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


_PROVIDERS = {}


def register(key):
    def decorator(cls):
        _PROVIDERS[key] = cls

        def wrapper(*args, **kwargs):
            cls()

        return cls

    return decorator


print(_PROVIDERS)


@register("ollama")
class OllamaProvider:
    def __init__(self, base_url: str = "http://localhost:11434", timeout=120):
        self.base_url = base_url
        self.timeout = timeout

    def get_client(self):
        return AsyncClient(base_url=self.base_url, timeout=self.timeout)

    async def call_llm(self, client, model: str, messages: list[Message]):
        return await call_ollama(client=client, model=model, messages=messages)


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
    model = "qwen3.8:27b-dflash2-nvfp4"
    provider = OllamaProvider()
    async with provider.get_client() as client:
        msg = await provider.call_llm(client=client, model=model, messages=msgs)
    print(msg)
    async with provider.get_client() as client:
        msg = await provider.call_llm(client=client, model=model, messages=msgs)
    print(msg)


if __name__ == "__main__":
    asyncio.run(main())
