# AGENTS.md

Small coding agent in Python that loops against a local Ollama model. Code is in `src/koding_agent`, tests in `tests`.

- Use `uv` for everything: `uv run pytest`, `uv run ruff check`, `uv run pyright`.
- Pre-commit runs ruff, pyright and a lockfile check. Make sure they pass before you commit.
- Keep it small. Don't add dependencies without asking.
- Never run the agent on the host, only in a container (see README).
