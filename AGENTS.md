# AGENTS.md

Small coding agent in Python. It runs an XML action loop against a model: a local
Ollama model by default, or OpenRouter. Code is in `src/koding_agent`, tests in `tests`.

- Never run the agent on the host, only in a container (see "Running in a container" in the README).
- Use `uv` for everything: `uv run pytest`, `uv run ruff check`, `uv run ruff format`, `uv run pyright`.
- Pre-commit runs ruff, pyright and a lockfile check. Make sure `uv run pre-commit run` passes before you commit.
- Keep `koding_agent.core` free of front-end code. Front ends (`cli`, `console`, `banner`, `tui`) depend on core, never the reverse.
- Keep it small. Don't add dependencies without asking.
- `./test.sh ollama` and `./test.sh openrouter` are manual curl checks of the provider endpoints.
