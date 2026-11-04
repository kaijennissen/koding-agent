"""Limits the harness enforces and the system prompt tells the model about.

Defined once: the prompt reads these values, and the Bash tool applies them.
"""

# DEFAULT_* values can be overridden per run (CLI flags); the rest are fixed.
DEFAULT_MAX_TURNS = 25  # safety net; the agent is meant to stop itself with <exit>
MAX_OUTPUT_CHARS = 2000  # tool output is context we pay for on every later turn
DEFAULT_COMMAND_TIMEOUT = (
    120  # default seconds; a wall-clock limit per command, not per loop
)
