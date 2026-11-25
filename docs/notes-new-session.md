# Notes: `/new` session flow

`run_tool` emits `ToolCancelled` when its task is cancelled, so cancelling a turn
leaves events behind in the message queue. The reset has to run after them.

## 1. Submit routing

```mermaid
flowchart TD
    A[Enter / first_prompt] --> B{empty after strip?}
    B -- yes --> B1[clear prompt, return]
    B -- no --> C{parse_slash}
    C -- "None<br/>(plain text, incl. /etc/hosts…)" --> G{busy?}
    G -- yes --> G1[toast 'busy', keep draft]
    G -- no --> H[send to model, start worker]
    C -- "(name, args)" --> D[clear prompt]
    D --> E[run_command name]
    E --> F{"canonical(name) == 'new'?<br/>(aliases: clear)"}
    F -- yes --> S[start_new_session]
    F -- no --> E1["error: Unknown command: /name"]

    K[Ctrl+N] -- "draft kept" --> N[action_new_session]
    N --> S
```

Commands are dispatched before the busy check, so they work mid-turn. `args` is
parsed but not passed on yet.

## 2. `/new` while a tool runs

```mermaid
sequenceDiagram
    participant U as User
    participant A as AgentApp (UI loop)
    participant W as run_turn worker (group "turn")
    participant Q as Message queue

    Note over W: run_agent awaiting a bash tool
    U->>A: /new  (or Ctrl+N)
    A->>W: cancel() for each worker in group "turn"
    W-->>Q: ToolAborted (from run_tool's ToolCancelled)
    W-->>Q: TurnFinished (from finally)
    A->>W: await worker.wait() (WorkerCancelled suppressed)
    A->>Q: post NewSession
    Note over Q: queue order: ToolAborted, TurnFinished, NewSession
    Q->>A: ToolAborted → rows[id] still exists, row marked cancelled
    Q->>A: TurnFinished → busy=False, status cleared
    Q->>A: NewSession → reset_session
    Note over A: messages = fresh_history(), busy=False, status cleared<br/>rows.clear(), tool list cleared, details = NO_SELECTION<br/>.message widgets removed (banner stays)<br/>prompt focused
```

## 3. Why `NewSession`

```diff
 start_new_session
   cancel workers in group "turn"
   await worker.wait()
-  reset_session()                         # runs now, leftovers still queued
+  post_message(NewSession)                # reset runs after the leftovers
 ...
 queue: ToolAborted
-  on_tool_aborted → self.rows[id]  → KeyError (row already cleared)
+  on_tool_aborted → row still present → fine
```

## Edge cases

- **Cancel during the model call:** `run_tool` isn't active, so no `ToolAborted`;
  only `TurnFinished` is posted. The reset works the same way.
- **Cancel before the worker's first step:** the worker never runs its `try`/`finally`,
  so no `TurnFinished` is posted at all. That is why `reset_session` itself sets
  `busy=False` and clears the status. `run_turn` is passed to `run_worker` as a
  callable (not `run_turn()`), so a worker cancelled before it starts leaves no
  never-awaited coroutine behind.

## Plain REPL

No concurrency, so none of the above applies. `/new` (alias `/clear`) just rebinds
`messages = start_session()`, which prints the banner. Unknown commands are printed
in red and not sent. Both share `parse_slash` and `canonical` from `commands.py`.

## Later

- **#84 `/model`:** needs `args` passed through `run_command`, plus an idle-only
  guard there.
- **#85:** commands are dispatched before the busy check, so they are never queued
  as steering messages.
