"""Start-up banner and input prompt for koding agent.

The "kai" wordmark is the logo, one brand colour per letter. The name
next to it is plain bold: "koding agent" has no "i", so there is no letter
to carry the lime. Colour is
dropped when stdout is not a terminal or when NO_COLOR is set
(https://no-color.org).
"""

import os
import sys

from koding_agent import __version__
from koding_agent.core.colours import (
    BLUE,
    BOLD,
    DIM,
    LIME,
    MAGENTA,
    RESET,
)


def use_color() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def render_banner(model: str, color: bool, context: str | None = None) -> str:
    def paint(text: str, *codes: str) -> str:
        return f"{''.join(codes)}{text}{RESET}" if color else text

    # Logo glyphs: k | a | i
    k_top, a_top, i_top = "█▄▀", "▄▀█", "█"
    k_bot, a_bot, i_bot = "█ █", "█▀█", "█"

    name = paint("koding agent", BOLD)
    meta = paint(f"v{__version__} · {model} · type 'exit' to quit", DIM)

    def logo(k: str, a: str, i: str) -> str:
        return f"{paint(k, MAGENTA)} {paint(a, BLUE)} {paint(i, LIME)}"

    line1 = f"{logo(k_top, a_top, i_top)}   {name}"
    line2 = f"{logo(k_bot, a_bot, i_bot)}   {meta}"
    line3 = f"{' ' * 8}{paint(f'context: {context or "-"}', DIM)}"
    return f"{line1}\n{line2}\n{line3}\n"


def print_banner(model: str, context: str | None = None) -> None:
    print(render_banner(model, color=use_color(), context=context))


def render_prompt(color: bool, readline_active: bool) -> str:
    """Input prompt: three chevrons in the k/a/i brand colours.

    When readline drives input() it counts escape codes as visible text and
    misplaces the cursor while editing; wrapping each code in \\001...\\002
    marks it as zero-width.
    """
    if not color:
        return "❯❯❯ "

    def code(seq: str) -> str:
        return f"\001{seq}\002" if readline_active else seq

    return f"{code(MAGENTA)}❯{code(BLUE)}❯{code(LIME)}❯{code(RESET)} "


def prompt_text() -> str:
    return render_prompt(color=use_color(), readline_active="readline" in sys.modules)
