"""Message and call types, and the Ollama client."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

import httpx2
from httpx2 import AsyncClient
from llm import LlmError

DEFAULT_MODEL = "qwen3.8:27b-mlx"

class Message(TypedDict):
    role: str
    content: str


@dataclass
class ProviderConfig:
    provider: str
    model:str |None=None
    base_url:str | None=None
    timeout:int |None=None


type LlmCall = Callable[[list[Message]], Awaitable[str]]  # history -> reply

class Provider(Protocol):
    def complete():
        ...
    def aclose():
        ...


class OllamaProvider:
    def __init__(self,base_url:str="http://localhost:11434",model:str="qwen3.8:27b-mlx",timeout:int=120):
        self.base_url=base_url
        self.model=model
        self.client=AsyncClient(base_url=self.base_url, timeout=timeout)

    async def complete(self,messages: list[Message] ) -> str:
            """Ask Ollama for the next reply; every failure is raised as LlmError."""
            try:
                response = await self.client.post(
                    "/api/chat", json={"model": self.model, "stream": False, "messages": messages}
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

    async def aclose(self):
        self.client.aclose()

    @classmethod
    def from_config(cls,cfg:ProviderConfig)->Provider:
        try:
            return cls(model=cfg.model)
        except Exception as err:
            raise LlmError(f"Couldn't instantiat the class: {err}") from err

class OpenrouterProvider:
    def __init__(self,base_url:str="https://eu.openrouter.ai",model:str="qwen3.8:27b-mlx",timeout:int=120):
        self.base_url=base_url
        self.model=model
        self.client=AsyncClient(base_url=self.base_url, timeout=timeout)

    async def complete(self,messages: list[Message] ) -> str:
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
        self.client.aclose()

    @classmethod
    def from_config(cls,cfg:ProviderConfig)->Provider:
        # TODO: needs to be fixed in both Providers
        try:
            return cls(model=cfg.model)
        except Exception as err:
            raise LlmError(f"Couldn't instantiat the class: {err}") from err

_REGISTRY:dict[str,Provider]={"ollama":OllamaProvider}

@asynccontextmanager
async def open_provider(cfg:ProviderConfig):
    try:
        factory=_REGISTRY[cfg.provider].from_config
    except KeyError as err:
        raise LlmError(f"Unknown provider: {cfg.provider}; Choose one of {_REGISTRY.keys()}") from err

    provider=factory(cfg) # needs to be here, because otherwise an error during instansiation that doesn't provide a provider errors in finally due to the nonexisting provider variable.

    try:
        yield provider

    finally:
        await provider.aclose()






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

    async with open_provider(cfg) as provider:
        response=await provider.complete(msgs)

    print(response)


    cfg = ProviderConfig(provider="ollama", model="qwen3.8:27b-mlx")
    async with open_provider(cfg) as provider:
        response=await provider.complete(msgs)

    print(response)


if __name__ == "__main__":
    asyncio.run(main())
