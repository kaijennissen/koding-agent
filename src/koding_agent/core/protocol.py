"""Parsing the model's reply into actions."""

import re

MIXED_ACTIONS_ERROR = (
    "Invalid action: a turn may contain a single <exit> tag, one or more "
    "<bash> tags, or one or more <skill> tags, but not a mix of them. "
    "Re-issue the action with a single action type."
)

_ACTION_TAGS = ("bash", "exit", "skill")
_ACTION = re.compile(r"<action>(.*)</action>", re.DOTALL)
_OPEN = re.compile(rf"<({'|'.join(_ACTION_TAGS)})>")
# An element only ends at a closing tag that is followed by the next element or
# the end of the action. A tag mentioned inside the body (prose, a code block,
# a heredoc) is followed by more text, so it is content, not markup.
_ELEMENT_END = rf"(?=\s*(?:<(?:{'|'.join(_ACTION_TAGS)})>|$))"


def parse_elements(text: str) -> list[tuple[str, str]]:
    """The (tag, body) pairs of the action elements in a reply, in order.

    Only the outermost <action>...</action> is read (the whole reply if there is
    none), and elements are read one after another from its first action tag, so
    tags that merely appear inside a body are left alone. Bodies are stripped.
    """
    match = _ACTION.search(text)
    rest = match.group(1) if match else text
    elements: list[tuple[str, str]] = []
    while opening := _OPEN.search(rest):
        tag = opening.group(1)
        closing = re.compile(rf"</{tag}>{_ELEMENT_END}").search(rest, opening.end())
        if closing is None:
            break
        elements.append((tag, rest[opening.end() : closing.start()].strip()))
        rest = rest[closing.end() :]
    return elements


def parse_action(text: str) -> dict[str, list[str]]:
    """Read the model's reply and return the action(s) it asked for.

    A turn is either a single <exit> or one-or-more <bash> actions. Every
    action in the reply must be of the same type; a reply that mixes <exit>
    with <bash> is invalid and is reported as an error. A reply with no
    recognised tag at all is also an error, so the model can retry.
    """
    elements = parse_elements(text)
    exit_bodies = [body for tag, body in elements if tag == "exit"]
    bash_bodies = [body for tag, body in elements if tag == "bash"]
    skill_bodies = [body for tag, body in elements if tag == "skill"]
    num_bodies = len(
        [bodies for bodies in [exit_bodies, bash_bodies, skill_bodies] if bodies]
    )
    if num_bodies > 1:
        # Mixing the three action types is not allowed. Signal the mistake as
        # an error so the loop can feed it back and the model can retry,
        # rather than silently terminating.
        return {"error": [MIXED_ACTIONS_ERROR]}
    if exit_bodies:
        return {"exit": exit_bodies}
    if bash_bodies:
        return {"bash": bash_bodies}
    if skill_bodies:
        return {"skill": skill_bodies}
    return {"error": ["Couldn't parse any action. Please retry!"]}
