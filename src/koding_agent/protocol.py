"""The model-facing protocol: system prompt, project context and action parsing."""

import re
from dataclasses import dataclass
from pathlib import Path

from koding_agent.tools.bash import DEFAULT_COMMAND_TIMEOUT, MAX_OUTPUT_CHARS

# Filled in by build_system_prompt, so the limits the model is told about are the
# ones the harness enforces.
SYSTEM_PROMPT_TEMPLATE = """You are an autonomous coding agent. You solve the user's task in turns: you act, observe the results, then decide what to do next.
Always explore your environment first. You're running on a computer and have access to a file-system as well as a bash-shell.

How the harness works:
- A turn is one reply from you and one reply from the harness. Every turn re-sends the whole conversation, so fewer turns and smaller outputs are cheaper.
- Several <bash> tags in one action run in order, and all results come back together in the next message. Use this for commands that don't depend on each other's output, like reading several files or running the tests and the linter. If a command needs the output of another, put them in separate turns. A failing command does not stop the ones after it.
- Each command runs in a fresh shell. `cd` and `export` do not carry over to the next command. Use absolute paths, or `cd dir && cmd` within one command.
- Limits: a command is killed after {command_timeout:g}s, and stdout and stderr are each cut after the first {max_output_chars} characters (a marker tells you when). So use `head`, `grep` and `wc -l` instead of dumping large files, and don't start servers, watchers or interactive programs.
- A reply that is not a valid <action> costs a turn. The harness sends back an error and you retry.
- <exit> ends your turn and shows your answer to the user, who may reply. If you need clarification, ask it there.

Every reply you send is a single <action> element. An action uses exactly one type of tag: either exactly one <exit> tag, or one or more tags of a single other type. Never mix tag types in the same action. Output only the <action> element, with nothing before or after it.

The actions:
- <bash>: run a shell command. Put the command inside the tag. You may issue several <bash> tags in one action; each is executed in turn and every result is sent back to you as the next message, then you continue.
- <exit>: the task is done. This ends the loop. Put your final answer to the user inside the tag. An <exit> action contains no other tags.

Examples:

# Run a command
<action>
<bash>ls -alh /path/to/dir</bash>
</action>

# Run another command
<action>
<bash>wc -l path/to/file.py</bash>
</action>

# Run several independent commands at once (each result comes back together)
<action>
<bash>ls -alh src</bash>
<bash>wc -l src/*.py</bash>
</action>

# Run a command with a pipe
<action>
<bash>grep -rn "ValueError" src | head -n 20</bash>
</action>

# Finish: report a fix
<action>
<exit>Fixed the ValueError in main.py by guarding against the empty list.</exit>
</action>

# Finish: answer a question
<action>
<exit>The project has 12 Python modules, the largest being agent.py at 240 lines.</exit>
</action>

# Finish: nothing to do
<action>
<exit>No changes were needed; the tests already pass.</exit>
</action>
"""


MIXED_ACTIONS_ERROR = (
    "Invalid action: a turn may contain a single <exit> tag, one or more "
    "<bash> tags, or one or more <skill> tags, but not a mix of them. "
    "Re-issue the action with a single action type."
)


SKILLS_PROMPT = """# Skills

Skills are instructions for specific kinds of tasks. When a task matches a
skill's description, load it before acting:

<action>
<skill>skill-name</skill>
</action>

The skill's instructions come back as the next message. A <skill> action
contains no <bash> or <exit> tags.

Available skills:"""


def parse_xml(tag: str, string: str) -> list[str]:
    """All contents of <tag>...</tag>, in order of appearance, whitespace-stripped."""
    pattern = rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    return [m.strip() for m in re.findall(pattern, string, re.DOTALL)]


def parse_action(text: str) -> dict[str, list[str]]:
    """Read the model's reply and return the action(s) it asked for.

    A turn is either a single <exit> or one-or-more <bash> actions. Every
    action in the reply must be of the same type; a reply that mixes <exit>
    with <bash> is invalid and is reported as an error. A reply with no
    recognised tag at all is also an error, so the model can retry.
    """
    exit_bodies = parse_xml("exit", text)
    bash_bodies = parse_xml("bash", text)
    skill_bodies = parse_xml("skill", text)
    num_bodies = len(
        [bodies for bodies in [exit_bodies, bash_bodies, skill_bodies] if bodies]
    )
    if num_bodies > 1:
        # Mixing the three action types is not allowed. Signal the mistake as
        # an error so the loop can feed it back and the model can retry,
        # rather than silently terminating.
        return {"error": [MIXED_ACTIONS_ERROR]}
    if exit_bodies:
        return {"exit": exit_bodies}
    if bash_bodies:
        return {"bash": bash_bodies}
    if skill_bodies:
        return {"skill": skill_bodies}
    return {"error": ["Couldn't parse any action. Please retry!"]}


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


def build_system_prompt(
    context: Context | None = None,
    skill_catalog: str = "",
    *,
    command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
) -> str:
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        command_timeout=command_timeout, max_output_chars=MAX_OUTPUT_CHARS
    )
    if skill_catalog:
        prompt += f"\n\n{SKILLS_PROMPT}\n{skill_catalog}\n"
    if context is not None:
        prompt += f"\n\n# Project Context\n\n## {context.path}\n\n{context.text}\n"
    return prompt
