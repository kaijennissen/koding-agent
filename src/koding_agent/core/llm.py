"""Message and call types, and the error front ends report."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class LlmReply:
    """One model answer. Tool calls are empty until tools arrive."""

    text: str
    tool_calls: tuple[()] = ()
    usage: Usage | None = None


type LlmCall = Callable[[list[Message]], Awaitable[LlmReply]]  # history -> reply


class LlmError(Exception):
    """The model gave no usable reply. The message is meant for the user.

    Front ends catch this one type and keep the session alive; the history is
    unchanged by the failed call, so the same run can simply be retried.
    """
