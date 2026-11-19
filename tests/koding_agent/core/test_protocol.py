"""Parsing the model's XML reply into actions."""

import pytest

from koding_agent.core.protocol import MIXED_ACTIONS_ERROR, parse_action, parse_elements


@pytest.mark.parametrize(
    "text,expected",
    [
        pytest.param(
            "<bash> ls -alh </bash>", [("bash", "ls -alh")], id="strips-edges"
        ),
        pytest.param(
            "<action><bash> ls </bash><bash>pwd</bash></action>",
            [("bash", "ls"), ("bash", "pwd")],
            id="all-in-order",
        ),
        pytest.param(
            "<exit> Finished! </exit>", [("exit", "Finished!")], id="exit-tag"
        ),
        pytest.param(
            "<bash>\n  cd src &&\n  ls\n</bash>",
            [("bash", "cd src &&\n  ls")],
            id="multiline-keeps-inner-whitespace",
        ),
        pytest.param("<skill>tdd</skill>", [("skill", "tdd")], id="skill-tag"),
        pytest.param("<unknown>x</unknown>", [], id="unknown-tag"),
        pytest.param("<bash>ls", [], id="unclosed"),
        # pins current behaviour, not a wish
        pytest.param("<bash></bash>", [("bash", "")], id="empty-body"),
        pytest.param(
            "<bash>ls</bash>\n<exit>done</exit>",
            [("bash", "ls"), ("exit", "done")],
            id="mixed-types-kept-in-order",
        ),
        # Tags that appear inside a body are content, not markup.
        pytest.param(
            "<action><exit>Use the `<bash>` tag, e.g. `<bash>ls</bash>`.</exit></action>",
            [("exit", "Use the `<bash>` tag, e.g. `<bash>ls</bash>`.")],
            id="exit-mentions-bash-tags",
        ),
        pytest.param(
            "<action><exit>Example:\n```\n<action>\n<bash>ls</bash>\n</action>\n```\n"
            "done</exit></action>",
            [
                (
                    "exit",
                    "Example:\n```\n<action>\n<bash>ls</bash>\n</action>\n```\ndone",
                )
            ],
            id="exit-contains-fenced-action-example",
        ),
        pytest.param(
            "<action><exit>The parser looks for `</exit>` and stops.</exit></action>",
            [("exit", "The parser looks for `</exit>` and stops.")],
            id="exit-mentions-own-closing-tag",
        ),
        pytest.param(
            "<action><bash>cat > a.md <<'EOF'\nuse <bash>ls</bash>\nEOF</bash></action>",
            [("bash", "cat > a.md <<'EOF'\nuse <bash>ls</bash>\nEOF")],
            id="bash-heredoc-contains-bash-tags",
        ),
        pytest.param(
            "<action><bash>echo '<b>hi</b>' > a.html</bash></action>",
            [("bash", "echo '<b>hi</b>' > a.html")],
            id="bash-contains-html",
        ),
        pytest.param(
            "I will use <bash> next.\n<action><bash>ls</bash></action>",
            [("bash", "ls")],
            id="chatter-before-action-mentions-tag",
        ),
        pytest.param(
            "```xml\n<action>\n<bash>ls</bash>\n</action>\n```",
            [("bash", "ls")],
            id="fenced-action",
        ),
        pytest.param(
            "<action><exit>## Result\n- `a`\n- **b**</exit></action>",
            [("exit", "## Result\n- `a`\n- **b**")],
            id="markdown-in-exit",
        ),
    ],
)
def test_parse_elements(text: str, expected: list[tuple[str, str]]) -> None:
    """Elements come back in order, stripped at the edges only; tags quoted inside a body stay content."""
    assert parse_elements(text) == expected


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
        pytest.param(
            "<action><exit>Use the `<bash>` tag, e.g. `<bash>ls</bash>`.</exit></action>",
            {"exit": ["Use the `<bash>` tag, e.g. `<bash>ls</bash>`."]},
            id="exit-quoting-bash-is-not-mixed",
        ),
        pytest.param(
            "<action><exit>The parser looks for `</exit>` and stops.</exit></action>",
            {"exit": ["The parser looks for `</exit>` and stops."]},
            id="exit-keeps-quoted-closing-tag",
        ),
    ],
)
def test_parse_action(text: str, expected: dict[str, list[str]]) -> None:
    """One action type per turn; mixing or an unknown reply is an error."""
    assert parse_action(text) == expected
