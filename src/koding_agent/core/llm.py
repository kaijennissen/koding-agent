"""Message and call types, and the Ollama client."""

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
