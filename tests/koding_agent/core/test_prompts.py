"""The system prompt: project context, the command timeout and the skills catalog."""

from pathlib import Path

import pytest

from koding_agent.core import DEFAULT_COMMAND_TIMEOUT, Context, build_system_prompt
from koding_agent.core.prompts import SKILLS_PROMPT


@pytest.mark.parametrize("with_context", [False, True])
def test_system_prompt_includes_context(tmp_path: Path, with_context: bool) -> None:
    path = tmp_path / "AGENTS.md"
    prompt = build_system_prompt(Context(path, "be nice") if with_context else None)
    assert prompt.startswith(build_system_prompt())
    assert (f"## {path}" in prompt and "be nice" in prompt) == with_context


@pytest.mark.parametrize(
    "command_timeout,shown",
    [(DEFAULT_COMMAND_TIMEOUT, "120s"), (7, "7s"), (2.5, "2.5s")],
)
def test_system_prompt_announces_the_command_timeout(
    command_timeout: float, shown: str
) -> None:
    """The model is told the limit the harness enforces, not a hard-coded one."""
    prompt = build_system_prompt(command_timeout=command_timeout)
    assert f"a command is killed after {shown}" in prompt


@pytest.mark.parametrize("catalog", ["", "- pdf: Work with PDFs"])
def test_system_prompt_lists_skills_only_when_there_are_some(catalog: str) -> None:
    prompt = build_system_prompt(None, catalog)
    assert prompt.startswith(build_system_prompt())
    assert (SKILLS_PROMPT in prompt and catalog in prompt) == bool(catalog)
