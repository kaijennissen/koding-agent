"""Project context: the AGENTS.md the model is shown."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Context:
    path: Path
    text: str


def load_context(root: Path | None = None) -> Context | None:
    """Find AGENTS.md (any casing) directly in `root`; None if absent or empty."""
    root = root or Path.cwd()
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name.lower() == "agents.md":
            text = path.read_text(encoding="utf-8").strip()
            return Context(path.resolve(), text) if text else None
    return None
