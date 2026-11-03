"""Test doubles: a scripted model for the LlmCall seam, and canned tool results."""

from koding_agent.agent import BashResult, LlmCall, Message


def replies(*script: str | Exception) -> LlmCall:
    """A fake model: answers with the scripted texts in order, raises scripted errors."""
    it = iter(script)

    async def call(_messages: list[Message]) -> str:
        reply = next(it)
        if isinstance(reply, Exception):
            raise reply
        return reply

    return call


def ok(cmd: str) -> BashResult:
    return BashResult.completed(cmd, stdout=f"{cmd} output", stderr="", returncode=0)
