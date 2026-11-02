"""A minimal coding agent: an act -> observe -> decide loop over an LLM.

Each turn: ask the model for one or more actions, execute them (sequentially),
feed the results back. A turn is either one <exit> or one-or-more <bash>
actions; mixing the two is not allowed. The loop ends when the model
emits <exit> or the turn budget runs out.
"""

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import TypedDict, cast

import httpx2

from koding_agent.core.events import (
    OnEvent,
    TurnStart,
)
from koding_agent.core.tool_types import BashCall
from koding_agent.core.tools import (
    RunTool,
    _ignore,
    create_uuid,
    emit,
    format_observation,
    run_tool,
)

# OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
OLLAMA_ENDPOINT = "http://host.docker.internal:11434/api/chat"
MODEL = "qwen3.8:27b-mlx"

DEFAULT_MAX_TURNS = 25  # safety net; the agent is meant to stop itself with <exit>

MIXED_ACTIONS_ERROR = (
    "Invalid action: a turn may contain either a single <exit> tag "
    "or one or more <bash> tags, but not both. Re-issue the action "
    "with a single action type."
)


class Message(TypedDict):
    role: str
    content: str


type LlmCall = Callable[[list[Message]], Awaitable[str]]  # history -> reply


class LlmError(Exception):
    """The model gave no usable reply. The message is meant for the user.

    Front ends catch this one type and keep the session alive; the history is
    unchanged by the failed call, so the same run can simply be retried.
    """


async def llm_call(client: httpx2.AsyncClient, messages: list[Message]) -> str:
    """Ask Ollama for the next reply; every failure is raised as LlmError."""
    try:
        response = await client.post(
            "/api/chat", json={"model": MODEL, "stream": False, "messages": messages}
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


def parse_xml(tag: str, string: str) -> list[str]:
    """All contents of <tag>...</tag>, in order of appearance, whitespace-stripped."""
    pattern = rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    return [m.strip() for m in re.findall(pattern, string, re.DOTALL)]


def parse_action(text: str) -> dict[str, list[str]]:
    """Read the model's reply and return the action(s) it asked for.

    A turn is either a single <exit> or one-or-more <bash> actions. Every
    action in the reply must be of the same type; a reply that mixes <exit>
    with <bash> is invalid and is reported as an error. A reply with no
    recognised tag at all is also an error, so the model can retry.
    """
    exit_bodies = parse_xml("exit", text)
    bash_bodies = parse_xml("bash", text)
    if exit_bodies and bash_bodies:
        # Mixing the two action types is not allowed. Signal the mistake as
        # an error so the loop can feed it back and the model can retry,
        # rather than silently terminating.
        return {"error": [MIXED_ACTIONS_ERROR]}
    if exit_bodies:
        return {"exit": exit_bodies}
    if bash_bodies:
        return {"bash": bash_bodies}
    return {"error": ["Couldn't parse any action. Please retry!"]}


async def run_agent(
    messages: list[Message],
    max_turns: int,
    call: LlmCall,
    tool_fn: RunTool,
    on_event: OnEvent = _ignore,
) -> tuple[str, list[Message]]:
    for turn in range(max_turns):
        emit(on_event, TurnStart(turn, max_turns))

        response = await call(messages)
        # print(response)  # the model speaking: the one moment worth watching
        # not yet appended to messages such that a failure in executing the bash calls doesn't leave the agent in a partial state.
        assistant_msg: Message = {"role": "assistant", "content": response}

        action = parse_action(response)
        if "exit" in action:
            messages += [assistant_msg]
            return action["exit"][0], messages
        if "error" in action:
            # Invalid reply (mixed or no action); tell the model why and let it retry.
            messages.extend(
                [assistant_msg, {"role": "user", "content": action["error"][0]}]
            )
            continue

        if "bash" in action:
            # One or more commands, executed in turn; every result is fed
            # back together in a single observation message.
            tool_calls = [BashCall(id=create_uuid(), cmd=cmd) for cmd in action["bash"]]
            results = await asyncio.gather(
                *(
                    run_tool(tool_call, tool_fn=tool_fn, on_event=on_event)
                    for tool_call in tool_calls
                )
            )

            observations = [format_observation(result) for result in results]
            messages.extend(
                [
                    assistant_msg,
                    {"role": "user", "content": "\n\n".join(observations)},
                ]
            )

    return f"Ran out of turns ({max_turns}) without an <exit> action.", messages
