---
runtimes:
  python:
    version: "3.14"
pre-agent-steps:
  - name: Install koding-agent
    run: |
      # Installed before AWF starts: the sandbox has no access to PyPI or GitHub
      # source archives. GH_AW_ENGINE_VERSION is a git ref of koding-agent.
      python3 -m pip install --quiet --user --disable-pip-version-check \
        "koding-agent @ git+https://github.com/kaijennissen/koding-agent@${GH_AW_ENGINE_VERSION}"
      "$HOME/.local/bin/koding-agent" --version
# Keep real credentials out of the sandbox: the AWF API proxy injects them.
excluded-env:
  - COPILOT_GITHUB_TOKEN
  - OPENAI_API_KEY
  - CODEX_API_KEY
  - ANTHROPIC_API_KEY
engine:
  id: koding-agent
  detection-engine: copilot
  # A git ref (branch, tag or commit SHA) of github.com/kaijennissen/koding-agent.
  version: "main"
  display-name: koding-agent
  description: Small bash-only coding agent with an XML action loop, run headless
  experimental: true
  mcp: false
  provider:
    name: github
  behaviors:
    secret-strategy: universal-llm-consumer
    capabilities:
      max-turns: true
    network:
      defaults:
        - host.docker.internal
      provider-domains:
        copilot: api.githubcopilot.com
        openai: api.openai.com
    execution:
      command-name: koding-agent
      step-name: Execute koding-agent
      model-env-var: GH_AW_KODING_MODEL
      provider-env-mode: universal-llm-consumer
      write-timestamp: true
    harness-script: |
      const { spawnSync } = require("child_process");
      const { readFileSync } = require("fs");
      const { homedir } = require("os");
      const { join } = require("path");
      const {
        fetchAWFReflect,
        deriveBaseUrlFromModelsURL,
        normalizeReflectProviderName,
        REFLECT_PROVIDER_ALIASES,
      } = require("./awf_reflect.cjs");

      const log = message => process.stderr.write(`[koding-agent] ${message}\n`);

      const main = async () => {
        const [command, ...args] = process.argv.slice(2);
        if (!command) throw new Error("koding-agent command is required");
        if (process.env.AWF_REFLECT_ENABLED !== "1") {
          throw new Error("koding-agent requires the AWF sandbox");
        }

        // The proxy speaks OpenAI chat-completions for Copilot and OpenAI;
        // koding-agent has no Anthropic Messages client.
        const provider = process.env.GH_AW_LLM_PROVIDER;
        if (!["github", "openai"].includes(provider)) {
          throw new Error(`koding-agent supports copilot/ and openai/ models, not ${provider}`);
        }
        const selected = process.env.GH_AW_KODING_MODEL || "";
        const separator = selected.indexOf("/");
        if (separator <= 0 || separator === selected.length - 1) {
          throw new Error("model must use provider/model format, e.g. copilot/claude-haiku-5.5");
        }
        const model = selected.slice(separator + 1);

        const turns = process.env.GH_AW_MAX_TURNS || "50";
        if (!/^[1-9][0-9]*$/.test(turns)) throw new Error("GH_AW_MAX_TURNS must be a positive integer");
        const promptFile = process.env.GH_AW_PROMPT;
        if (!promptFile) throw new Error("GH_AW_PROMPT is not set");
        const prompt = readFileSync(promptFile, "utf8");

        const reflected = await fetchAWFReflect({ logger: log });
        if (!reflected.ok || !reflected.reflectData) {
          throw new Error(`cannot discover the LLM endpoint from /reflect: ${reflected.reason || "empty response"}`);
        }
        const aliases = REFLECT_PROVIDER_ALIASES[provider];
        const endpoint = reflected.reflectData.endpoints?.find(
          entry =>
            entry?.configured === true &&
            aliases.has(normalizeReflectProviderName(entry.provider)) &&
            typeof entry.models_url === "string"
        );
        if (!endpoint) throw new Error(`no configured /reflect endpoint for provider ${provider}`);
        // ".../v1/models" -> ".../v1": the API root koding-agent's --base-url expects.
        const baseUrl = deriveBaseUrlFromModelsURL(endpoint.models_url);
        log(`provider=${provider} model=${model} base-url=${baseUrl} max-turns=${turns}`);

        // The proxy injects the real credential; koding-agent only needs a non-empty key.
        const env = {
          ...process.env,
          PATH: `${join(homedir(), ".local", "bin")}:${process.env.PATH || ""}`,
          GH_AW_KODING_API_KEY: "awf-proxy",
        };
        for (const name of ["COPILOT_GITHUB_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY"]) {
          delete env[name];
        }

        const result = spawnSync(
          command,
          [
            ...args,
            "--prompt", prompt,
            "--provider", "openai",
            "--base-url", baseUrl,
            "--api-key-env", "GH_AW_KODING_API_KEY",
            "--model", model,
            "--max-turns", turns,
            "--no-skills",
          ],
          { env, cwd: process.env.GITHUB_WORKSPACE, stdio: "inherit" }
        );
        if (result.error) throw result.error;
        if (result.status !== 0) {
          const error = new Error(`koding-agent failed: ${result.signal || `exit code ${result.status}`}`);
          error.exitCode = result.status || 1;
          throw error;
        }
      };

      main().catch(error => {
        log(error instanceof Error ? error.message : String(error));
        process.exitCode = error.exitCode || 1;
      });
---

## koding-agent execution notes

You act through one `<action>` per reply: `<bash>` commands, or `<exit>` with your
final answer. Commands run from the repository root.

- **Safe outputs are shell commands:** `safeoutputs <tool_name> '<json>'`, run inside
  `<bash>`, with the complete JSON object as one single-quoted argument. For example:

  <action>
  <bash>safeoutputs noop '{"message":"Nothing to change."}'</bash>
  </action>

- **Record the safe output before you exit.** `<exit>` ends the run; text in the
  exit answer is not published anywhere.
- If you cannot finish the task, run
  `safeoutputs report_incomplete '{"reason":"..."}'` with the actual blocker
  instead of claiming success.

<!--
# koding-agent engine

Engine definition for [koding-agent](https://github.com/kaijennissen/koding-agent).
Import it and select a `provider/model`:

```yaml
engine:
  id: koding-agent
model: copilot/claude-haiku-5.5
imports:
  - shared/koding-agent.md
permissions:
  copilot-requests: write
```

Supported providers are `copilot` and `openai`: the harness discovers the
matching AWF API-proxy endpoint through `/reflect` and runs
`koding-agent --provider openai --base-url <proxy API root>` with a placeholder
key. Real credentials stay out of the agent process. Anthropic models are not
supported (koding-agent has no Anthropic Messages client).

koding-agent is installed with pip from a git ref (`engine.version`) in a
pre-agent step, before AWF starts. It has no MCP client: tools and safe outputs
are reached through gh-aw's shell CLI proxies. There is no log parser yet, so
the step summary shows raw output only. `max-turns` maps to `--max-turns`; a
run that ends without `<exit>` exits non-zero.

Models: the model must be in the provider's live catalog and have AI-credits
pricing, otherwise the AWF proxy rejects it (`unknown_model_ai_credits`). Copilot
Claude models (`claude-haiku-5.5`, `claude-sonnet-5`, `claude-sonnet-5.5`) passed
the smoke test. `gpt-6-luna` answered one turn, then OpenAI's prompt filter
rejected the conversation ("flagged as potentially violating our usage policy").
-->
