"""Message and call types, and the error front ends report."""

from collections.abc import Awaitable, Callable
from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


type LlmCall = Callable[[list[Message]], Awaitable[str]]  # history -> reply


class LlmError(Exception):
    """The model gave no usable reply. The message is meant for the user.

    Front ends catch this one type and keep the session alive; the history is
    unchanged by the failed call, so the same run can simply be retried.
    """
