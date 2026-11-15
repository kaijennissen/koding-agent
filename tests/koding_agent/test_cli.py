"""Mode selection and argument rules of the command line."""

import pytest

from koding_agent import cli
from koding_agent.cli import (
    Mode,
    parse_args,
    select_mode,
)
from koding_agent.core import DEFAULT_COMMAND_TIMEOUT


@pytest.mark.parametrize(
    "argv,mode,tui_prompt,headless_prompt",
    [
        pytest.param([], "tui", None, None, id="no-args-tui"),
        pytest.param(["fix it"], "tui", "fix it", None, id="positional-tui"),
        pytest.param(["-p", "fix it"], "headless", None, "fix it", id="-p"),
        pytest.param(["--prompt", "fix it"], "headless", None, "fix it", id="--prompt"),
        pytest.param(["--plain"], "plain", None, None, id="--plain"),
    ],
)
def test_select_mode(
    argv: list[str], mode: Mode, tui_prompt: str | None, headless_prompt: str | None
) -> None:
    """Each mode reads its prompt from its own argument."""
    args = parse_args(argv)

    assert select_mode(args) == mode
    assert (args.prompt, args.headless_prompt) == (tui_prompt, headless_prompt)


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["-p", "x", "--plain"], id="-p+--plain"),
        pytest.param(["first", "-p", "x"], id="positional+-p"),
        pytest.param(["first", "--plain"], id="positional+--plain"),
    ],
)
def test_conflicting_modes_are_rejected(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv,expected",
    [
        pytest.param([], DEFAULT_COMMAND_TIMEOUT, id="default"),
        pytest.param(["--command-timeout", "7.5"], 7.5, id="--command-timeout"),
    ],
)
def test_command_timeout_flag(argv: list[str], expected: float) -> None:
    assert parse_args(argv).command_timeout == expected


@pytest.mark.parametrize(
    "argv,expected",
    [
        pytest.param([], None, id="default"),
        pytest.param(["--model", "llama3"], "llama3", id="--model"),
        pytest.param(["-m", "llama3"], "llama3", id="-m"),
    ],
)
def test_model_flag(argv: list[str], expected: str | None) -> None:
    assert parse_args(argv).model == expected


def test_provider_flags_default_to_none() -> None:
    """Unset flags stay None so the provider's own defaults apply."""
    args = parse_args([])
    assert (args.provider, args.base_url, args.api_key_env, args.timeout) == (
        "ollama",
        None,
        None,
        None,
    )


def test_provider_flags() -> None:
    args = parse_args(
        ["--provider", "openrouter", "--base-url", "http://x", "--api-key-env", "K"]
        + ["--timeout", "5"]
    )
    assert (args.provider, args.base_url, args.api_key_env, args.timeout) == (
        "openrouter",
        "http://x",
        "K",
        5,
    )


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(["--provider", "nope"])
    assert exc.value.code == 2


def test_tui_without_a_terminal_exits_with_a_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(SystemExit) as exc:
        cli.cli([])
    assert "-p" in str(exc.value.code)


@pytest.mark.parametrize(
    ("argv", "expected"), [([], False), (["-nc"], True), (["--no-context"], True)]
)
def test_no_context_flag(argv: list[str], expected: bool) -> None:
    assert parse_args(argv).no_context is expected


def test_provider_setup_error_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing model for openrouter is reported, not shown as a traceback."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with pytest.raises(SystemExit) as exc:
        cli.cli(["-p", "x", "--provider", "openrouter"])
    assert "--model" in str(exc.value.code)
