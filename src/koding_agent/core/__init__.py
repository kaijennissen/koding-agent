"""The agent core: loop, prompts, protocol, tools, events and the model client.

This facade re-exports the API that front ends and other supported callers
need; everything else is imported from its defining submodule. The core knows
nothing about the CLI, console, banner or TUI.
"""

from koding_agent.core.context import Context, load_context
from koding_agent.core.events import (
    Event,
    OnEvent,
    ToolCancelled,
    ToolEnd,
    ToolStart,
    TurnStart,
)
from koding_agent.core.limits import DEFAULT_COMMAND_TIMEOUT, DEFAULT_MAX_TURNS
from koding_agent.core.llm import DEFAULT_MODEL, LlmCall, LlmError, Message
from koding_agent.core.loop import run_agent
from koding_agent.core.prompts import build_system_prompt
from koding_agent.core.providers import call_openrouter as llm_call
from koding_agent.core.tools import ToolCall, ToolResult
from koding_agent.core.tools.bash import BashCall, BashResult
from koding_agent.core.tools.skills import (
    Skill,
    SkillCall,
    SkillLibrary,
    SkillResult,
    load_skills,
)

__all__ = [
    "DEFAULT_COMMAND_TIMEOUT",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MODEL",
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
    "llm_call",
    "load_context",
    "load_skills",
    "run_agent",
]
