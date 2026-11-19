"""A coding agent built from scratch."""

from importlib.metadata import version

# Single source of truth is pyproject.toml; read it from installed metadata.
__version__ = version("koding-agent")

__all__ = ["__version__"]
