"""Test doubles for the core's two seams, LlmCall and RunTool: no network, no subprocess."""

from koding_agent.core.agent import LlmCall, Message
from koding_agent.core.tools import BashResult


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


async def ok_tool(cmd: str) -> BashResult:
    """A fake tool runner: every command succeeds at once."""
    return ok(cmd)
