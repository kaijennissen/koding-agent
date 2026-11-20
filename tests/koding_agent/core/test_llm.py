"""LlmReply: the immutable result of one model call."""

import dataclasses

import pytest

from koding_agent.core import LlmReply, Usage


@pytest.mark.parametrize(
    ("field", "expected"),
    [("text", "hi"), ("tool_calls", ()), ("usage", None)],
    ids=["text", "tool_calls-empty", "usage-none"],
)
def test_reply_defaults(field: str, expected: object) -> None:
    assert getattr(LlmReply(text="hi"), field) == expected


@pytest.mark.parametrize(
    ("prompt_tokens", "completion_tokens"),
    [(3, 5), (0, 0), (1200, 350)],
    ids=["small", "zero", "large"],
)
def test_reply_carries_usage(prompt_tokens: int, completion_tokens: int) -> None:
    usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    reply = LlmReply(text="hi", usage=usage)
    assert reply.usage == usage
    assert reply.usage is not None
    assert reply.usage.prompt_tokens == prompt_tokens
    assert reply.usage.completion_tokens == completion_tokens


@pytest.mark.parametrize(
    ("field", "new_value"),
    [("text", "changed"), ("tool_calls", ("call",)), ("usage", Usage(1, 1))],
    ids=["text", "tool_calls", "usage"],
)
def test_reply_is_frozen(field: str, new_value: object) -> None:
    reply = LlmReply(text="hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(reply, field, new_value)
