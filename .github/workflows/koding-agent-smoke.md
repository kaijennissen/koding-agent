---
description: |
  Smoke test for the koding-agent engine: the agent inspects the repository
  and records a noop safe output. Run it manually after changing the engine
  definition in shared/koding-agent.md.

on:
  workflow_dispatch:

engine:
  id: koding-agent
model: copilot/claude-haiku-5.5

imports:
  - shared/koding-agent.md

permissions:
  contents: read
  copilot-requests: write

safe-outputs:
  noop:

max-turns: 10
timeout-minutes: 10
---

# koding-agent smoke test

1. List the files in the repository root and read the first lines of `README.md`.
2. Record the result with a safe output:
   `safeoutputs noop '{"message":"<one sentence: what this repository is>"}'`
3. Then exit with a short summary.
