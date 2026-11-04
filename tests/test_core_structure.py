"""Structure of the `koding_agent.core` package: public API, import boundaries
and import order (each order in a fresh interpreter)."""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import koding_agent
import koding_agent.core as core

PACKAGE_DIR = Path(koding_agent.__file__).parent
CORE_DIR = PACKAGE_DIR / "core"
CORE = "koding_agent.core"
PRESENTATION = (
    "koding_agent.cli",
    "koding_agent.console",
    "koding_agent.banner",
    "koding_agent.tui",
    "koding_agent.__main__",
    "textual",
    "rich",
)

PUBLIC_API = {
    "DEFAULT_COMMAND_TIMEOUT",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MODEL",
    "BashCall",
    "BashResult",
    "Context",
    "Event",
    "LlmCall",
    "LlmError",
    "Message",
    "OnEvent",
    "Skill",
    "SkillCall",
    "SkillLibrary",
    "SkillResult",
    "ToolCall",
    "ToolCancelled",
    "ToolEnd",
    "ToolResult",
    "ToolStart",
    "TurnStart",
    "build_system_prompt",
    "llm_call",
    "load_context",
    "load_skills",
    "run_agent",
}


def module_name(path: Path) -> str:
    rel = path.relative_to(PACKAGE_DIR.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def core_modules() -> dict[str, Path]:
    return {module_name(p): p for p in sorted(CORE_DIR.rglob("*.py"))}


MODULES = core_modules()


def explicit_imports(name: str, path: Path) -> set[str]:
    """Modules named by import statements, relative imports resolved.

    Implicit parent-package initialization is not an import statement and is
    deliberately not counted. `from pkg import x` counts as `pkg.x` when that
    is a module, otherwise as `pkg`.
    """
    is_package = path.name == "__init__.py"
    package = name if is_package else name.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.rsplit(".", node.level - 1)[0]
                base = f"{anchor}.{base}" if base else anchor
            for alias in node.names:
                sub = f"{base}.{alias.name}"
                found.add(sub if sub in MODULES else base)
    return found


def find_cycle(graph: dict[str, set[str]]) -> list[str] | None:
    state: dict[str, int] = {}  # 1 = on stack, 2 = done
    stack: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        stack.append(node)
        for nxt in sorted(graph.get(node, ())):
            if state.get(nxt) == 1:
                return [*stack[stack.index(nxt) :], nxt]
            if nxt not in state and (cycle := visit(nxt)):
                return cycle
        stack.pop()
        state[node] = 2
        return None

    for node in sorted(graph):
        if node not in state and (cycle := visit(node)):
            return cycle
    return None


GRAPH = {name: explicit_imports(name, path) for name, path in MODULES.items()}


def test_expected_modules_exist() -> None:
    expected = {
        "context",
        "events",
        "limits",
        "llm",
        "loop",
        "prompts",
        "protocol",
        "tools",
        "tools.bash",
        "tools.skills",
    }
    assert {f"{CORE}.{m}" for m in expected} <= set(MODULES)


def test_old_agent_module_is_gone() -> None:
    assert not (PACKAGE_DIR / "agent.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("koding_agent.agent")


def test_public_api_is_curated() -> None:
    assert set(core.__all__) == PUBLIC_API
    for name in PUBLIC_API:
        assert hasattr(core, name), name


def test_facade_does_not_export_helpers() -> None:
    for helper in ("emit", "run_tool", "parse_action", "truncate", "run_bash_tool"):
        assert helper not in core.__all__
        assert not hasattr(core, helper)


def test_facade_does_not_define_implementation() -> None:
    tree = ast.parse((CORE_DIR / "__init__.py").read_text(encoding="utf-8"))
    defs = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    assert defs == []


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("run_agent", f"{CORE}.loop"),
        ("load_context", f"{CORE}.context"),
        ("build_system_prompt", f"{CORE}.prompts"),
        ("llm_call", f"{CORE}.llm"),
        ("LlmError", f"{CORE}.llm"),
        ("ToolStart", f"{CORE}.events"),
        ("BashCall", f"{CORE}.tools.bash"),
        ("SkillLibrary", f"{CORE}.tools.skills"),
        ("DEFAULT_COMMAND_TIMEOUT", f"{CORE}.limits"),
    ],
)
def test_defined_in_expected_module(name: str, module: str) -> None:
    obj = getattr(core, name)
    if callable(obj) and hasattr(obj, "__module__"):
        assert obj.__module__ == module
    else:
        assert getattr(importlib.import_module(module), name) == obj


def test_limits_are_shared_by_prompt_and_bash_tool() -> None:
    from koding_agent.core import limits, prompts
    from koding_agent.core.tools import bash

    assert prompts.MAX_OUTPUT_CHARS is limits.MAX_OUTPUT_CHARS
    assert bash.MAX_OUTPUT_CHARS is limits.MAX_OUTPUT_CHARS
    assert prompts.DEFAULT_COMMAND_TIMEOUT is limits.DEFAULT_COMMAND_TIMEOUT
    for path in MODULES.values():
        if path.name != "limits.py":
            text = path.read_text(encoding="utf-8")
            assert "MAX_OUTPUT_CHARS =" not in text
            assert "DEFAULT_COMMAND_TIMEOUT =" not in text


def test_core_imports_no_presentation_code() -> None:
    for name, imports in GRAPH.items():
        for imported in imports:
            for banned in PRESENTATION:
                assert imported != banned and not imported.startswith(banned + "."), (
                    f"{name} imports {imported}"
                )


def test_core_modules_do_not_import_the_facade() -> None:
    for name, imports in GRAPH.items():
        if name == CORE:
            continue
        assert CORE not in imports, f"{name} imports the core facade"


def test_core_has_no_import_cycles() -> None:
    internal = {
        name: {i for i in imports if i in MODULES} for name, imports in GRAPH.items()
    }
    assert find_cycle(internal) is None


def test_cycle_detector_finds_cycles() -> None:
    assert find_cycle({"a": {"b"}, "b": {"c"}, "c": {"a"}}) == ["a", "b", "c", "a"]
    assert find_cycle({"a": {"b"}, "b": set(), "c": {"b"}}) is None


def test_relative_imports_are_resolved(tmp_path: Path) -> None:
    pkg = tmp_path / "core" / "tools"
    pkg.mkdir(parents=True)
    mod = pkg / "x.py"
    mod.write_text("from . import y\nfrom .. import z\nfrom ..w import v\n")
    # resolved against a package path that exists in MODULES-less context
    imports = explicit_imports("koding_agent.core.tools.x", mod)
    assert imports == {
        "koding_agent.core.tools",
        "koding_agent.core",
        "koding_agent.core.w",
    }


def run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


def assert_imports(*modules: str) -> None:
    code = "import importlib\n" + "".join(
        f"importlib.import_module({m!r})\n" for m in modules
    )
    result = run_python(code)
    assert result.returncode == 0, f"{modules}:\n{result.stderr}"


def test_core_import_pulls_in_no_presentation_modules() -> None:
    result = run_python(
        "import sys, koding_agent.core\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('textual', 'rich') "
        "or m in ('koding_agent.cli', 'koding_agent.console', "
        "'koding_agent.banner') or m.startswith('koding_agent.tui')]\n"
        "assert not bad, bad\n"
    )
    assert result.returncode == 0, result.stderr


ALL_CORE = sorted(MODULES)


@pytest.mark.parametrize("first", ALL_CORE)
def test_import_order_in_fresh_interpreter(first: str) -> None:
    rest = [m for m in ALL_CORE if m != first]
    assert_imports(first, *rest)
    assert_imports(first, *reversed(rest))
    assert_imports(first, *rest[len(rest) // 2 :], *rest[: len(rest) // 2])


@pytest.mark.parametrize(
    "module",
    ["koding_agent.cli", "koding_agent.console", "koding_agent.banner"]
    + ["koding_agent.tui.app"],
)
def test_front_end_modules_import_first(module: str) -> None:
    assert_imports(module, *ALL_CORE)
    assert_imports(*ALL_CORE, module)
