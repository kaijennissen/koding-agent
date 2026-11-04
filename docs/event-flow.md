# How events travel between layers

The core (the `koding_agent.core` package) knows nothing about any UI. It
reports progress through one callback, `on_event`, which must return
immediately. Each front end supplies its own callback: `print_event`
(`koding_agent.console`) for the CLI, `AgentApp.forward_event` for the TUI.

Where things live: the event types, `emit` and `run_tool` are in
`core/events.py`; `run_agent` is in `core/loop.py`; the concrete tools
(`core/tools/bash.py`, `core/tools/skills.py`) know nothing about events.
`events.py` depends on the tools only through the neutral `ToolCall` /
`ToolResult` / `RunTool` types in `core/tools/__init__.py`, and the loop wires
tools and events together. Dependencies point one way:
`cli`/`console`/`tui` → `core` facade → `loop` → `events` → `tools`.
Import-boundary tests in `tests/test_core_structure.py` guard this.

## Sequence: one turn, one tool call (TUI)

```mermaid
sequenceDiagram
    box Core (agent task)
        participant Agent as run_agent
        participant Tool as run_tool
    end
    box TUI
        participant Fwd as forward_event
        participant Q as message queue
        participant UI as handlers / widgets
    end

    Agent->>Fwd: TurnStart
    Fwd-)Q: TurnStarted
    Note over Agent: llm_call → parse_action → BashCall(s)
    Agent->>Tool: run_tool(call) (one per call, via gather)
    Tool->>Fwd: ToolStart
    Fwd-)Q: ToolStarted
    alt command finishes (any exit code or timeout)
        Tool->>Fwd: ToolEnd(result)
        Fwd-)Q: ToolFinished
    else agent task cancelled
        Tool->>Fwd: ToolCancelled
        Fwd-)Q: ToolAborted
    end
    Tool-->>Agent: result
    Q-)UI: on_turn_started / on_tool_*
    Note over UI: update status bar, mount ToolRow, set_status
```

Solid arrows (`->>`) are plain function calls on the agent's task, made through
`emit()`, which swallows exceptions so a broken front end can't crash the agent.
Open arrows (`-)`) are fire-and-forget. `post_message` only enqueues, and the
handlers run later on Textual's own loop. That queue is the boundary between
the two layers.

## State: one ToolRow

```mermaid
stateDiagram-v2
    [*] --> running: ToolStarted
    running --> ok: ToolFinished, exit 0
    running --> failed: ToolFinished, exit ≠ 0 or timeout
    running --> cancelled: ToolAborted
    ok --> [*]
    failed --> [*]
    cancelled --> [*]
```

`status_for(result)` alone decides between `ok` and `failed`. Each state is a
CSS class on the row.
