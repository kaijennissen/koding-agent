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
from koding_agent.core.llm import LlmCall, LlmError, LlmReply, Message, Usage
from koding_agent.core.loop import OutOfTurns, run_agent
from koding_agent.core.prompts import build_system_prompt
from koding_agent.core.providers import ProviderConfig, open_provider, provider_names
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
    "BashCall",
    "BashResult",
    "Context",
    "Event",
    "LlmCall",
    "LlmError",
    "LlmReply",
    "Message",
    "Usage",
    "OnEvent",
    "OutOfTurns",
    "ProviderConfig",
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
    "load_context",
    "load_skills",
    "open_provider",
    "provider_names",
    "run_agent",
]
