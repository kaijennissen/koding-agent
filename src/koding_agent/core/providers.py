"""Message and call types, and the Ollama client."""

import asyncio
import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol, cast

import httpx2
from httpx2 import AsyncClient

from koding_agent.core.llm import LlmError, Message


@dataclass
class ProviderConfig:
    provider: str
    model: str
    base_url: str | None = None
    timeout: int | None = None


type LlmCall = Callable[[list[Message]], Awaitable[str]]  # history -> reply


class Provider(Protocol):
    @classmethod
    def from_config(cls, cfg: ProviderConfig) -> Provider: ...
    async def complete(self, messages: list[Message]) -> str: ...
    async def aclose(self): ...


# 1) base class
# 2) pass everything the class knows


class _BaseProvider:
    """Shared config handling for all providers"""

    @classmethod
    def from_config(cls, cfg: ProviderConfig) -> _BaseProvider:
        try:
            return cls()
        except Exception as err:
            raise LlmError(f"Couldn't instantiate the class: {err}") from err


async def post_json(
    client: AsyncClient, endpoint: str, json: dict[str, str | bool | list[Message]]
):
    try:
        return await client.post(endpoint, json=json)
    except httpx2.HTTPError as e:
        raise LlmError(f"Could not reach the model: {e}") from e


class OllamaProvider:
    _DEFAULT_BASE_URL: str = "http://localhost:11434"
    _DEFAULT_MODEL: str = "qwen3.8:27b-mlx"
    _DEFAULT_TIMEOUT: int = 120

    def __init__(
        self,
        base_url: str | None,
        model: str | None,
        timeout: int | None,
        client: AsyncClient | None = None,
    ):
        self.base_url = base_url or self._DEFAULT_BASE_URL
        self.model = model or self._DEFAULT_MODEL
        self.timeout = timeout or self._DEFAULT_TIMEOUT
        self.client = client or AsyncClient(
            base_url=self.base_url, timeout=self.timeout
        )  # makes testing easier

    async def complete(self, messages: list[Message]) -> str:
        """Ask Ollama for the next reply; every failure is raised as LlmError."""

        response = await post_json(
            client=self.client,
            endpoint="/api/chat",
            json={"model": self.model, "stream": False, "messages": messages},
        )

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

    async def aclose(self):
        await self.client.aclose()

    @classmethod
    def from_config(cls, cfg: ProviderConfig) -> OllamaProvider:
        try:
            return cls(base_url=cfg.base_url, model=cfg.model, timeout=cfg.timeout)
        except Exception as err:
            raise LlmError(f"Couldn't instantiat the class: {err}") from err


class OpenrouterProvider:
    _DEFAULT_BASE_URL: str = "https://eu.openrouter.ai"
    _DEFAULT_MODEL: str = "qwen3.8:27b-mlx"
    _DEFAULT_TIMEOUT: int = 10
    _API_KEY_ENV_VAR: str = "OPENROUTER_API_KEY"

    def __init__(
        self,
        base_url: str | None,
        model: str | None,
        timeout: int | None,
        client: AsyncClient | None = None,
    ):
        token = os.getenv(self._API_KEY_ENV_VAR)
        self.base_url = base_url or self._DEFAULT_BASE_URL
        self.model = model or self._DEFAULT_MODEL
        self.timeout = timeout or self._DEFAULT_TIMEOUT
        self.client = client or AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {token}"},
        )

    async def complete(self, messages: list[Message]) -> str:
        """Ask OpenRouter for the next reply; every failure is raised as LlmError."""
        try:
            response = await self.client.post(
                "/api/v1/chat/completions",
                json={"model": self.model, "stream": False, "messages": messages},
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

    async def aclose(self):
        await self.client.aclose()

    @classmethod
    def from_config(cls, cfg: ProviderConfig) -> OpenrouterProvider:
        # TODO: needs to be fixed in both Providers
        try:
            return cls(base_url=cfg.base_url, model=cfg.model, timeout=cfg.timeout)
        except Exception as err:
            raise LlmError(f"Couldn't instantiat the class: {err}") from err


_REGISTRY: dict[str, type[Provider]] = {
    "ollama": OllamaProvider,
    "openrouter": OpenrouterProvider,
}


@asynccontextmanager
async def open_provider(cfg: ProviderConfig):
    try:
        factory = _REGISTRY[cfg.provider].from_config
    except KeyError as err:
        raise LlmError(
            f"Unknown provider: {cfg.provider}; Choose one of {_REGISTRY.keys()}"
        ) from err

    provider = factory(cfg)
    # needs to be here, because otherwise an error during instansiation that doesn't provide a provider errors in finally due to the nonexisting provider variable.

    try:
        yield provider

    finally:
        await provider.aclose()


async def main():
    msgs: list[Message] = [{"role": "user", "content": "Why is the sky blue?"}]
    cfg = ProviderConfig(provider="ollama", model="qwen3.8:27b-dflash2-nvfp4")

    async with open_provider(cfg) as provider:
        response = await provider.complete(msgs)

    print(response)

    cfg = ProviderConfig(provider="ollama", model="qwen3.8:27b-mlx")
    async with open_provider(cfg) as provider:
        response = await provider.complete(msgs)

    print(response)


if __name__ == "__main__":
    asyncio.run(main())
