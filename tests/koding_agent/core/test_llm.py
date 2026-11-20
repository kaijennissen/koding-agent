"""LlmReply: the immutable result of one model call."""

import dataclasses

import pytest

from koding_agent.core import LlmReply, Usage


def test_reply_defaults_to_text_with_no_tool_calls_and_no_usage() -> None:
    reply = LlmReply(text="hi")
    assert reply.text == "hi"
    assert reply.tool_calls == ()
    assert reply.usage is None


def test_reply_carries_optional_usage() -> None:
    usage = Usage(prompt_tokens=3, completion_tokens=5)
    reply = LlmReply(text="hi", usage=usage)
    assert reply.usage == usage
    assert reply.usage is not None
    assert reply.usage.prompt_tokens == 3
    assert reply.usage.completion_tokens == 5


def test_reply_is_frozen() -> None:
    reply = LlmReply(text="hi")
    with pytest.raises(dataclasses.FrozenInstanceError):
        reply.text = "changed"  # type: ignore[misc]
