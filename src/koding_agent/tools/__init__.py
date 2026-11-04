"""Tools the agent can call."""

from collections.abc import Awaitable, Callable

from koding_agent.tools.bash import BashCall, BashResult
from koding_agent.tools.skills import SkillCall, SkillResult

type ToolCall = BashCall | SkillCall
type ToolResult = BashResult | SkillResult
type RunTool[C: ToolCall, R: ToolResult] = Callable[[C], Awaitable[R]]  # call -> result
