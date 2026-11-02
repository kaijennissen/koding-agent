"""Start-up banner and input prompt for koding agent.

The "kai" wordmark is the logo, one brand colour per letter. The name
next to it is plain bold: "koding agent" has no "i", so there is no letter
to carry the lime. Colour is
dropped when stdout is not a terminal or when NO_COLOR is set
(https://no-color.org).
"""


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
RED = rgb("#FF5555")
GREEN = rgb("#50FA7B")
YELLOW = rgb("#FFCC00")
