<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/headline-dark.svg">
    <img src="docs/images/headline-light.svg" alt="koding agent" height="48">
  </picture>
</h1>

A coding agent built from scratch: a small act → observe → decide loop that asks a
local LLM (served by Ollama) for a shell command, runs it, and feeds the output back
until the model declares the task done.

![koding agent start-up banner](docs/images/startup-banner.png)

> [!CAUTION]
> **Never run this agent directly on your host system.**
> The agent executes model-written shell commands with `shell=True` and no
> filtering. A wrong or malicious command can delete files, read credentials, or
> exfiltrate data. Run it **only inside a container** (at the very least Docker;
> see [Run the agent](#run-the-agent)), on a throwaway clone of the repository,
> with nothing secret mounted or reachable.

## Status

This is a learning project that has grown into a *koding agent that could, in
theory, do real work*. It is small enough to read in one sitting (~250 lines of
Python), but it is already used to develop itself: features such as parallel
(batched) tool calls were implemented by the agent working on its own repository.

What it does today:

- **XML action loop** – every reply is one `<action>` containing either `<bash>`
  (run a command, get the output back) or `<exit>` (final answer).
- **Three ways to run it** – a terminal UI (the default), a headless one-shot
  (`--prompt`, answer printed pipe-clean), and a plain line-based REPL (`--plain`).
  See [Modes](#modes).
- **Bounded runs** – a turn budget (`--max-turns`), a timeout per command (`--command-timeout`, default 120 s),
  and truncated tool output (first 2000 chars of stdout and of stderr) so a runaway
  turn can't flood the context.
- **Configurable model** – `--model` picks the Ollama model (default `qwen3.8:27b-mlx`).
- **Parallel tool calls** – a turn may batch several independent `<bash>` actions;
  mixing `<bash>` and `<exit>` is rejected with an error fed back to the model.
  (Lives on the `multiple_tool_calls` branch; not yet merged into `main`.)
- **Hardened container** – read-only root filesystem, all capabilities dropped,
  non-root user, resource limits, and an independent clone as the only writable
  project state. Design: [`thoughts/shared/plans/001_hardened_agent_container_design.md`](thoughts/shared/plans/001_hardened_agent_container_design.md).
- **Quality gates** – ruff, pyright (strict), pytest, pre-commit, and CI.

Known limits: one model backend (Ollama), no file-edit tool beyond what bash
offers (arguably not a limit, see below), no persistence between sessions, and
the container still has network access to reach Ollama. Docker narrows the
blast radius; it is not a perfect sandbox. Review the agent's diff before you
commit or publish anything.

> **Is bash-only really a limitation?** Maybe not. In
> [*An Empirical Study of Harness Design for Coding Agents*](https://arxiv.org/abs/2609.20804)
> (Fan et al., 2026), models with good shell proficiency did as well or better with
> a bash-only action space than with predefined file/search/edit tools, at
> substantially lower cost, especially on command-line-centric tasks. Weaker models
> did worse without the structured tools, and results were mixed for some models
> (one dropped sharply on SWE-Bench but gained on Terminal-Bench), so the benefit
> depends on the model. The study did not evaluate the model used here.

The [`diary.md`](diary.md) logs pitfalls and design decisions made along the way.

## Develop

Development uses uv with Python 3.14.4; `uv sync` builds the environment. The
agent reaches Ollama at `http://host.docker.internal:11434`, the host as seen from
a container, so develop **inside a container** as well. To run it directly on the
host instead, point it elsewhere with a full URL:
`export OLLAMA_HOST=http://localhost:11434`.

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
uv run pyright          # type-check
uv run pytest           # test
```

## Layout

```text
src/koding_agent/
  core/             # the agent: knows nothing about any UI
    __init__.py     # curated public API (re-exports only)
    loop.py         # run_agent and turn orchestration
    context.py      # AGENTS.md project context
    prompts.py      # system prompt templates and construction
    protocol.py     # action parsing and protocol errors
    events.py       # event types, emission and tool lifecycle
    llm.py          # message/call types and the Ollama client
    limits.py       # timeout and output limits shared by prompts and Bash tool
    tools/          # bash.py, skills.py
  console.py        # CLI event printing and terminal colours
  banner.py
  cli.py
  tui/
```

Front ends (`cli`, `console`, `tui`) depend on `koding_agent.core`; nothing in
core imports them, and modules inside core import each other directly, never via
the `koding_agent.core` facade.

**Breaking change:** `koding_agent.agent` is removed, with no compatibility shim.
Import the supported API from `koding_agent.core` (`run_agent`, `LlmError`,
`build_system_prompt`, event types, ...), internal helpers from their core
submodule (e.g. `koding_agent.core.protocol.parse_action`,
`koding_agent.core.tools.bash.run_bash_tool`), and `print_event` and the colour
constants from `koding_agent.console`.

## Modes

```sh
koding-agent                    # TUI
koding-agent "fix the bug"      # TUI, with this as the first message
koding-agent -p "fix the bug"   # headless: run once, print the answer
koding-agent --plain            # line-based REPL
```

`-t/--max-turns N` applies to all modes. Combining `-p`, `--plain` and a positional
prompt is an error.

- **TUI** – conversation on the left, a live list of tool calls on the right
  (yellow = running, green = exit 0, red = failed or timed out, grey = cancelled).
  `Enter` sends, `Ctrl+J` adds a newline, `Tab` shows or hides the tools panel,
  `Ctrl+C` quits. While a turn is running, `Enter` is refused and your text stays
  in the prompt. If the model can't be reached the error is shown and the turn is
  dropped from the history, so you can simply send again.
- **Headless (`-p`)** – no terminal needed. Turn and tool progress goes to stdout,
  followed by the answer. A model failure exits non-zero.
- **`--plain`** – the original REPL. Type `exit` to leave.

The TUI needs an interactive terminal on both stdin and stdout. Without one (a pipe,
cron or CI, `docker run` without `-t`, `ssh host cmd` without `-t`) it exits with a
hint instead of falling back silently; use `-p` there.

## Run the agent

The agent runs in a hardened container. It works on an independent clone of
**any git repository** mounted at `/workspace`; the code driving the run is the
read-only copy of this harness baked into the image at `/app`. The agent never
touches your real working tree.

```sh
# 1. Build the image with the current harness (run from this repository).
docker build --tag agent_sandbox --file Dockerfile .

# 2. Pick the target repository and create a fresh, independent clone of it.
#    REPO can be a remote URL or a local path (local: only committed work is cloned).
REPO=https://github.com/<owner>/<repo>.git
WORKSPACE=/tmp/agent-workspace
rm -rf "$WORKSPACE"
git clone --no-hardlinks "$REPO" "$WORKSPACE"

# 3. Run one agent invocation against the clone.
docker run --rm \
    -it \
    --read-only \
    --tmpfs /tmp:size=64m \
    --cap-drop ALL \
    --security-opt no-new-privileges \
    --pids-limit 100 \
    --memory 4g \
    --cpus 2 \
    --add-host=host.docker.internal:host-gateway \
    --volume "$WORKSPACE":/workspace \
    --name agent-sandbox \
    agent_sandbox \
    /opt/app/bin/koding-agent --max-turns 25 --prompt "..."

# 4. Review the agent's changes on the host.
git -C "$WORKSPACE" status
git -C "$WORKSPACE" diff
```

To work on this repository itself, use `REPO=.`.

Notes:

- Omit `--prompt` to open the TUI (it needs `-it`, as in the command above), or add
  `--plain` for the line-based REPL. Omit the trailing command to get an
  interactive shell in the same container instead.
- The image ships Python, `uv`, `curl`, `jq` and `vim`. If the target repository
  needs other toolchains, add them to the `Dockerfile`; the agent can't install
  system packages (non-root, read-only root filesystem).
- Only clone repositories you are willing to let the agent modify, and review the
  diff before pushing anything. Don't put credentials (SSH keys, tokens, `.env`
  files) into the clone.
- On a Linux host the clone must be writable by UID 1000, the container's `agent`
  user.
