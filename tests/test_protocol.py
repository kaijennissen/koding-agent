"""Parsing the model's XML actions, and building the system prompt."""

from pathlib import Path

import pytest

from koding_agent.protocol import (
    MIXED_ACTIONS_ERROR,
    SKILLS_PROMPT,
    Context,
    build_system_prompt,
    load_context,
    parse_action,
    parse_xml,
)
from koding_agent.tools.bash import DEFAULT_COMMAND_TIMEOUT


@pytest.mark.parametrize(
    "tag,text,expected",
    [
        pytest.param("bash", "<bash> ls -alh </bash>", ["ls -alh"], id="strips-edges"),
        pytest.param(
            "bash",
            "<action><bash> ls </bash><bash>pwd</bash></action>",
            ["ls", "pwd"],
            id="all-in-order",
        ),
        pytest.param("exit", "<exit> Finished! </exit>", ["Finished!"], id="exit-tag"),
        pytest.param(
            "action",
            "<action><bash> ls </bash><bash>pwd</bash></action>",
            ["<bash> ls </bash><bash>pwd</bash>"],
            id="outer-tag-keeps-inner-xml",
        ),
        pytest.param(
            "bash",
            "<bash>\n  cd src &&\n  ls\n</bash>",
            ["cd src &&\n  ls"],
            id="multiline-keeps-inner-whitespace",
        ),
        pytest.param("bash", "<exit>done</exit>", [], id="other-tag-only"),
        pytest.param("bash", "<bash>ls", [], id="unclosed"),
        # pins current behaviour, not a wish
        pytest.param("bash", "<bash></bash>", [""], id="empty-body"),
    ],
)
def test_parse_xml(tag: str, text: str, expected: list[str]) -> None:
    """Contents come back in order of appearance, stripped at the edges only."""
    assert parse_xml(tag, text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        pytest.param(
            "<bash>ls</bash><bash>pwd</bash>", {"bash": ["ls", "pwd"]}, id="bash"
        ),
        pytest.param("<exit> done </exit>", {"exit": ["done"]}, id="exit"),
        pytest.param(
            "<unknown> Finished! </unknown>",
            {"error": ["Couldn't parse any action. Please retry!"]},
            id="no-tag-errors",
        ),
        pytest.param(
            "Sure!\n<action><bash>ls</bash></action>\nHope it helps.",
            {"bash": ["ls"]},
            id="ignores-chatter",
        ),
        pytest.param(
            "<bash>ls</bash><exit>done</exit>",
            {"error": [MIXED_ACTIONS_ERROR]},
            id="mixed-bash-first",
        ),
        pytest.param(
            "<exit>done</exit><bash>ls</bash>",
            {"error": [MIXED_ACTIONS_ERROR]},
            id="mixed-exit-first",
        ),
    ],
)
def test_parse_action(text: str, expected: dict[str, list[str]]) -> None:
    """One action type per turn; mixing or an unknown reply is an error."""
    assert parse_action(text) == expected


@pytest.mark.parametrize("name", ["AGENTS.md", "agents.md", "Agents.MD"])
def test_load_context_matches_any_casing(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("be nice\n")
    ctx = load_context(tmp_path)
    assert ctx == Context(tmp_path / name, "be nice")


@pytest.mark.parametrize("content", [None, "", "  \n"])
def test_load_context_absent_or_empty(tmp_path: Path, content: str | None) -> None:
    if content is not None:
        (tmp_path / "AGENTS.md").write_text(content)
    assert load_context(tmp_path) is None


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
