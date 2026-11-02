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
from koding_agent.core.tools import (
    BashCall,
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
MAX_OUTPUT_CHARS = 2000  # tool output is context we pay for on every later turn
COMMAND_TIMEOUT = 120  # seconds; a wall-clock limit per command, not per loop


SYSTEM_PROMPT = f"""You are an autonomous coding agent. You solve the user's task in turns: you act, observe the results, then decide what to do next.
Always explore your environment first. You're running on a computer and have access to a file-system as well as a bash-shell.

How the harness works:
- A turn is one reply from you and one reply from the harness. Every turn re-sends the whole conversation, so fewer turns and smaller outputs are cheaper.
- Several <bash> tags in one action run in order, and all results come back together in the next message. Use this for commands that don't depend on each other's output, like reading several files or running the tests and the linter. If a command needs the output of another, put them in separate turns. A failing command does not stop the ones after it.
- Each command runs in a fresh shell. `cd` and `export` do not carry over to the next command. Use absolute paths, or `cd dir && cmd` within one command.
- Limits: a command is killed after {COMMAND_TIMEOUT}s, and output is cut after {MAX_OUTPUT_CHARS} characters (a marker tells you when). So use `head`, `grep` and `wc -l` instead of dumping large files, and don't start servers, watchers or interactive programs.
- A reply that is not a valid <action> costs a turn. The harness sends back an error and you retry.
- <exit> ends your turn and shows your answer to the user, who may reply. If you need clarification, ask it there.

Every reply you send is a single <action> element. It contains either exactly one <exit> tag, or one or more <bash> tags. Do not mix <exit> with <bash> in the same action. Output only the <action> element, with nothing before or after it.

The two actions:
- <bash>: run a shell command. Put the command inside the tag. You may issue several <bash> tags in one action; each is executed in turn and every result is sent back to you as the next message, then you continue.
- <exit>: the task is done. This ends the loop. Put your final answer to the user inside the tag. An <exit> action contains no <bash> tags.

Examples:

# Run a command
<action>
<bash>ls -alh /path/to/dir</bash>
</action>

# Run another command
<action>
<bash>wc -l path/to/file.py</bash>
</action>

# Run several independent commands at once (each result comes back together)
<action>
<bash>ls -alh src</bash>
<bash>wc -l src/*.py</bash>
</action>

# Run a command with a pipe
<action>
<bash>grep -rn "ValueError" src | head -n 20</bash>
</action>

# Finish: report a fix
<action>
<exit>Fixed the ValueError in main.py by guarding against the empty list.</exit>
</action>

# Finish: answer a question
<action>
<exit>The project has 12 Python modules, the largest being agent.py at 240 lines.</exit>
</action>

# Finish: nothing to do
<action>
<exit>No changes were needed; the tests already pass.</exit>
</action>
"""


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
