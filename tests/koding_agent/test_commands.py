"""Slash commands are recognised by shape: `/name [args]`, never paths or prose."""

import pytest

from koding_agent.commands import parse_slash


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("/new", ("new", ""), id="bare-command"),
        pytest.param("/model gpt-5", ("model", "gpt-5"), id="command-with-arg"),
        pytest.param("/model   a b  ", ("model", "a b"), id="args-are-stripped"),
        pytest.param("/etc/hosts is missing", None, id="path-is-not-command"),
        pytest.param("hello /new", None, id="slash-not-at-start"),
        pytest.param("/", None, id="lone-slash"),
        pytest.param("/ new", None, id="empty-name"),
        pytest.param("new", None, id="no-slash"),
        pytest.param(
            "/new\nmore text", ("new", "more text"), id="newline-separates-args"
        ),
        pytest.param("/new\tfoo", ("new", "foo"), id="tab-separates-args"),
        pytest.param(
            "/new /etc/hosts", ("new", "/etc/hosts"), id="path-in-args-is-fine"
        ),
        pytest.param("/new\n/etc/hosts", ("new", "/etc/hosts"), id="newline-then-path"),
        pytest.param("/model a\nb", ("model", "a\nb"), id="args-keep-inner-newlines"),
        pytest.param("//", None, id="double-slash"),
    ],
)
def test_parse_slash(text: str, expected: tuple[str, str] | None) -> None:
    assert parse_slash(text) == expected
