"""A minimal coding agent: an act -> observe -> decide loop over an LLM.

Each turn: ask the model for one or more actions, execute them (concurrently),
feed the results back. A turn is either one <exit>, one-or-more <bash>, or
one-or-more <skill> actions; mixing types is not allowed. The loop ends when
the model emits <exit> or the turn budget runs out.
"""

import asyncio
import uuid
from functools import partial

from koding_agent.banner import GREY, RESET
from koding_agent.events import (
    GREEN,
    RED,
    YELLOW,
    Event,
    OnEvent,
    ToolCancelled,
    ToolEnd,
    ToolStart,
    TurnStart,
    emit,
    ignore_event,
    print_event,
    run_tool,
)
from koding_agent.llm import DEFAULT_MODEL, LlmCall, LlmError, Message, llm_call
from koding_agent.protocol import (
    MIXED_ACTIONS_ERROR,
    SKILLS_PROMPT,
    Context,
    build_system_prompt,
    load_context,
    parse_action,
    parse_xml,
)
from koding_agent.tools import ToolCall, ToolResult
from koding_agent.tools.bash import (
    DEFAULT_COMMAND_TIMEOUT,
    MAX_OUTPUT_CHARS,
    BashCall,
    BashResult,
    format_bash_result,
    run_bash_tool,
    truncate,
)
from koding_agent.tools.skills import (
    Skill,
    SkillCall,
    SkillLibrary,
    SkillResult,
    format_skill_result,
    load_skills,
    parse_skill,
    run_skill_tool,
    split_frontmatter,
)

# Re-exported so `koding_agent.agent` stays the one-stop public import.
__all__ = [
    "GREEN",
    "GREY",
    "YELLOW",
    "MAX_OUTPUT_CHARS",
    "MIXED_ACTIONS_ERROR",
    "SKILLS_PROMPT",
    "DEFAULT_COMMAND_TIMEOUT",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MODEL",
    "RED",
    "RESET",
    "BashCall",
    "BashResult",
    "Context",
    "Event",
    "LlmCall",
    "LlmError",
    "Message",
    "OnEvent",
    "Skill",
    "SkillCall",
    "SkillLibrary",
    "SkillResult",
    "ToolCall",
    "ToolCancelled",
    "ToolEnd",
    "ToolResult",
    "ToolStart",
    "TurnStart",
    "build_system_prompt",
    "emit",
    "format_bash_result",
    "format_skill_result",
    "llm_call",
    "load_context",
    "load_skills",
    "parse_action",
    "parse_skill",
    "parse_xml",
    "print_event",
    "run_agent",
    "run_bash_tool",
    "run_skill_tool",
    "run_tool",
    "split_frontmatter",
    "truncate",
]

DEFAULT_MAX_TURNS = 25  # safety net; the agent is meant to stop itself with <exit>


def create_uuid():
    return str(uuid.uuid4()).split("-")[0]


async def run_agent(
    messages: list[Message],
    max_turns: int,
    call: LlmCall,
    skills: SkillLibrary,
    on_event: OnEvent = ignore_event,
    *,
    command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
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
        if "skill" in action:
            skill_calls = [
                SkillCall(id=create_uuid(), name=name) for name in action["skill"]
            ]
            skill_fn = partial(run_skill_tool, library=skills)
            skill_results = await asyncio.gather(
                *(
                    run_tool(skill_call, tool_fn=skill_fn, on_event=on_event)
                    for skill_call in skill_calls
                )
            )
            observations = [format_skill_result(result) for result in skill_results]
            messages.extend(
                [
                    assistant_msg,
                    {"role": "user", "content": "\n\n".join(observations)},
                ]
            )
        if "bash" in action:
            # One or more commands, executed in turn; every result is fed
            # back together in a single observation message.
            bash_calls = [BashCall(id=create_uuid(), cmd=cmd) for cmd in action["bash"]]
            bash_fn = partial(run_bash_tool, timeout=command_timeout)
            bash_results = await asyncio.gather(
                *(
                    run_tool(bash_call, tool_fn=bash_fn, on_event=on_event)
                    for bash_call in bash_calls
                )
            )
            observations = [format_bash_result(result) for result in bash_results]
            messages.extend(
                [
                    assistant_msg,
                    {"role": "user", "content": "\n\n".join(observations)},
                ]
            )

    return f"Ran out of turns ({max_turns}) without an <exit> action.", messages
