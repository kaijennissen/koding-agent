"""Parsing the model's XML reply into actions."""

import pytest

from koding_agent.core.protocol import MIXED_ACTIONS_ERROR, parse_action, parse_xml


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
