"""Shared slash-command parsing and alias resolution for the TUI and the plain REPL.

Pure functions only: no Textual, no state.
"""

import re

ALIASES = {"clear": "new"}

_SLASH = re.compile(r"/([^\s/]+)(?:\s+(.*))?", re.DOTALL)


def parse_slash(text: str) -> tuple[str, str] | None:
    """Split `/name [args]` into `(name, args)`, or None if it isn't a command.

    A command starts with `/` followed by a non-empty name. Any whitespace
    (space, tab, newline) separates the name from the args, which may span
    lines. A first word that contains another `/` is a path (`/etc/hosts`),
    not a command.
    """
    match = _SLASH.fullmatch(text)
    if match is None:
        return None
    return match[1], (match[2] or "").strip()


def canonical(name: str) -> str:
    """Resolve a command alias to its command name; other names pass through."""
    return ALIASES.get(name, name)
