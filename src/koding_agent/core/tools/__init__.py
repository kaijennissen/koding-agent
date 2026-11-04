"""Tools the agent can call, and the neutral contract the event layer builds on.

Concrete tools live in the submodules and know nothing about events; the
lifecycle reporting around them is in `koding_agent.core.events`.
"""

from collections.abc import Awaitable, Callable

from koding_agent.core.tools.bash import BashCall, BashResult
from koding_agent.core.tools.skills import SkillCall, SkillResult

__all__ = ["RunTool", "ToolCall", "ToolResult"]

type ToolCall = BashCall | SkillCall
type ToolResult = BashResult | SkillResult
type RunTool[C: ToolCall, R: ToolResult] = Callable[[C], Awaitable[R]]  # call -> result
