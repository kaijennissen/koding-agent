"""Parsing the model's reply into actions."""

import re

MIXED_ACTIONS_ERROR = (
    "Invalid action: a turn may contain a single <exit> tag, one or more "
    "<bash> tags, or one or more <skill> tags, but not a mix of them. "
    "Re-issue the action with a single action type."
)


def parse_xml(tag: str, string: str) -> list[str]:
    """All contents of <tag>...</tag>, in order of appearance, whitespace-stripped."""
    pattern = rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    return [m.strip() for m in re.findall(pattern, string, re.DOTALL)]


def parse_action(text: str) -> dict[str, list[str]]:
    """Read the model's reply and return the action(s) it asked for.

    A turn is either a single <exit> or one-or-more <bash> actions. Every
    action in the reply must be of the same type; a reply that mixes <exit>
    with <bash> is invalid and is reported as an error. A reply with no
    recognised tag at all is also an error, so the model can retry.
    """
    exit_bodies = parse_xml("exit", text)
    bash_bodies = parse_xml("bash", text)
    skill_bodies = parse_xml("skill", text)
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
