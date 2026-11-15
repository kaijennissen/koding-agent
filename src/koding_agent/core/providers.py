"""Model providers: config, the shared HTTP skeleton, and the registry."""

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, Self, cast

import httpx2
from httpx2 import AsyncClient, Response

from koding_agent.core.llm import LlmError, Message


@dataclass(kw_only=True)
class ProviderConfig:
    provider: str
    model: str | None
    base_url: str | None = None
    timeout: int | None = None
    api_key_env_var: str | None = None


class Provider(Protocol):
    """What consumers (the agent loop, front ends) need from a provider."""

    model: str

    async def complete(self, messages: list[Message]) -> str: ...
    async def aclose(self) -> None: ...


type ProviderFactory = Callable[[ProviderConfig], Provider]

DEFAULT_TIMEOUT = 120  # seconds; replies are not streamed


async def post_json(
    *,
    client: AsyncClient, endpoint: str, json: dict[str, str | bool | list[Message]]
) -> Response:
    try:
        return await client.post(endpoint, json=json)
    except httpx2.HTTPError as e:
        raise LlmError(f"Could not reach the model: {e}") from e


def _error_detail(response: Response) -> str:
    """The provider's reason for a failed request, else the raw body text."""
    try:
        error: Any = response.json()["error"]
    except (ValueError, KeyError, TypeError):
        return response.text
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        message: Any = cast(dict[str, Any], error).get("message")
        if isinstance(message, str):
            return message
    return response.text


def raise_for_error(response: Response) -> None:
    """Raise LlmError for an HTTP error status; do nothing otherwise."""
    if response.is_error:
        raise LlmError(
            f"Model request failed ({response.status_code}): {_error_detail(response)}"
        )


def reply_text(response: Response, *path: str | int) -> str:
    """Decode the body, follow `path` into it and return the reply text."""
    try:
        data: Any = response.json()
        for key in path:
            data = data[key]
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise LlmError(f"Unexpected reply from the model: {response.text}") from e
    if not isinstance(data, str):
        raise LlmError(f"Unexpected reply from the model: {response.text}")
    return data


class HttpProvider(ABC):
    """Shared skeleton of providers that POST JSON and read one reply text.

    A subclass sets `_ENDPOINT` and `_REPLY_PATH` and implements `from_config`,
    which resolves defaults, environment and credentials into explicit values.
    """

    _ENDPOINT: ClassVar[str]
    _REPLY_PATH: ClassVar[tuple[str | int, ...]]
    DEFAULT_BASE_URL: ClassVar[str]
    DEFAULT_MODEL: ClassVar[str | None] = None  # None: the user must pass a model

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls.from_config, "__isabstractmethod__", False):
            return  # an intermediate base: its concrete subclasses are checked
        for name in ("_ENDPOINT", "_REPLY_PATH", "DEFAULT_BASE_URL"):
            if not hasattr(cls, name):
                raise TypeError(f"{cls.__name__} must define {name}")

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: int,
        headers: dict[str, str] | None = None,
        client: AsyncClient | None = None,  # injectable, makes testing easier
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.client = client or AsyncClient(
            base_url=base_url, timeout=timeout, headers=headers
        )

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: ProviderConfig) -> Self:
        """Build a provider from the user's config; raise LlmError if unusable."""

    @classmethod
    def _model(cls, cfg: ProviderConfig) -> str:
        """The model to use: the user's, else the default; LlmError if neither."""
        model = cfg.model or cls.DEFAULT_MODEL
        if not model:
            raise LlmError(f"{cls.__name__} has no default model; pass --model")
        return model

    async def complete(self, messages: list[Message]) -> str:
        """Ask the model for the next reply; every failure is raised as LlmError."""
        response = await post_json(
            client=self.client,
            endpoint=self._ENDPOINT,
            json={"model": self.model, "stream": False, "messages": messages},
        )
        raise_for_error(response)
        return reply_text(response, *self._REPLY_PATH)

    async def aclose(self) -> None:
        await self.client.aclose()


class OllamaProvider(HttpProvider):
    _ENDPOINT = "/api/chat"
    _REPLY_PATH = ("message", "content")
    DEFAULT_BASE_URL = "http://localhost:11434"
    DEFAULT_MODEL = "qwen3.8:27b-mlx"
    BASE_URL_ENV = "OLLAMA_HOST"

    @classmethod
    def from_config(cls, cfg: ProviderConfig) -> Self:
        return cls(
            base_url=cfg.base_url
            or os.environ.get(cls.BASE_URL_ENV)
            or cls.DEFAULT_BASE_URL,
            model=cls._model(cfg),
            timeout=DEFAULT_TIMEOUT if cfg.timeout is None else cfg.timeout,
        )


class OpenrouterProvider(HttpProvider):
    _ENDPOINT = "/api/v1/chat/completions"
    _REPLY_PATH = ("choices", 0, "message", "content")
    DEFAULT_BASE_URL = "https://eu.openrouter.ai"
    API_KEY_ENV = "OPENROUTER_API_KEY"  # no DEFAULT_MODEL: a model is required

    @classmethod
    def from_config(cls, cfg: ProviderConfig) -> Self:
        model = cls._model(cfg)
        key_var = cfg.api_key_env_var or cls.API_KEY_ENV
        if not (token := os.getenv(key_var)):
            raise LlmError(f"${key_var} is not set")
        return cls(
            base_url=cfg.base_url or cls.DEFAULT_BASE_URL,
            model=model,
            timeout=DEFAULT_TIMEOUT if cfg.timeout is None else cfg.timeout,
            headers={"Authorization": f"Bearer {token}"},
        )


_REGISTRY: dict[str, ProviderFactory] = {
    "ollama": OllamaProvider.from_config,
    "openrouter": OpenrouterProvider.from_config,
}


def provider_names() -> list[str]:
    return sorted(_REGISTRY)


@asynccontextmanager
async def open_provider(cfg: ProviderConfig):
    try:
        factory = _REGISTRY[cfg.provider]
    except KeyError as err:
        raise LlmError(
            f"Unknown provider: {cfg.provider}; choose one of {provider_names()}"
        ) from err

    # Set up outside the try/finally below: a failed setup has nothing to close.
    # Unexpected setup errors become LlmError so front ends can report them.
    try:
        provider = factory(cfg)
    except LlmError:
        raise
    except Exception as err:
        raise LlmError(f"Couldn't set up provider '{cfg.provider}': {err}") from err

    try:
        yield provider
    finally:
        await provider.aclose()
