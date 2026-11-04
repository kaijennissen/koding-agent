"""Project context: finding the AGENTS.md the model is shown."""

from pathlib import Path

import pytest

from koding_agent.core import Context, load_context


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
