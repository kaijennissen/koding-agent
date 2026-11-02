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


def rgb(hex_color: str) -> str:
    """24-bit (truecolor) foreground escape for a '#RRGGBB' colour."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"\033[38;2;{r};{g};{b}m"


MAGENTA = rgb("#FF00D4")  # k
BLUE = rgb("#006FFF")  # a
LIME = rgb("#A9FF17")  # i
DIM = "\033[2m"
GREY = rgb("#888888")  # secondary info (tool calls)
BOLD = "\033[1m"
RESET = "\033[0m"


def use_color() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def render_banner(model: str, color: bool) -> str:
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
    return f"{line1}\n{line2}\n"


def print_banner(model: str) -> None:
    print(render_banner(model, color=use_color()))


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
