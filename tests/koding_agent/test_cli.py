"""Mode selection and argument rules of the command line."""

import re
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager

import pytest

from fakes import replies
from koding_agent import cli
from koding_agent.cli import (
    Mode,
    parse_args,
    run_plain,
    select_mode,
)
from koding_agent.core import (
    DEFAULT_COMMAND_TIMEOUT,
    LlmReply,
    Message,
    ProviderConfig,
    SkillLibrary,
)


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


@pytest.mark.parametrize("value", ["0", "-1"])
def test_timeout_must_be_positive(value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--timeout", value])


class FakeProvider:
    """A provider that answers from a script, standing in for `open_provider`."""

    def __init__(self, *script: str) -> None:
        self.model = "fake"
        self.complete = replies(*script)


@pytest.mark.parametrize(
    ("script", "exit_code"),
    [
        pytest.param(["<action><exit>done</exit></action>"], None, id="exit-0"),
        pytest.param(
            ["<action><bash>true</bash></action>"],
            "Ran out of turns (1) without an <exit> action.",
            id="out-of-turns-fails",
        ),
    ],
)
def test_headless_exit_status(
    monkeypatch: pytest.MonkeyPatch, script: list[str], exit_code: str | None
) -> None:
    """A one-shot run fails (non-zero exit) when it never reaches <exit>.

    sys.exit(str) exits with status 1; returning normally exits with 0.
    """

    @asynccontextmanager
    async def fake_open_provider(_cfg: object) -> AsyncGenerator[FakeProvider]:
        yield FakeProvider(*script)

    monkeypatch.setattr(cli, "open_provider", fake_open_provider)
    argv = ["-p", "task", "-t", "1", "--no-context", "--no-skills"]

    if exit_code is None:
        cli.cli(argv)
    else:
        with pytest.raises(SystemExit, match=re.escape(exit_code)):
            cli.cli(argv)


class RecordingProvider:
    """A provider that records the history of each call and always exits."""

    def __init__(self) -> None:
        self.model = "fake"
        self.calls: list[list[Message]] = []

    async def complete(self, messages: list[Message]) -> LlmReply:
        self.calls.append([Message(**m) for m in messages])
        return LlmReply(text="<action><exit>hi</exit></action>")


def roles_and_contents(messages: list[Message]) -> list[tuple[str, str]]:
    return [(m["role"], m["content"]) for m in messages]


class TestPlainSlashCommands:
    @pytest.fixture
    def provider(self, monkeypatch: pytest.MonkeyPatch) -> RecordingProvider:
        fake = RecordingProvider()

        @asynccontextmanager
        async def fake_open_provider(_cfg: object) -> AsyncGenerator[RecordingProvider]:
            yield fake

        monkeypatch.setattr(cli, "open_provider", fake_open_provider)
        return fake

    @staticmethod
    async def run(monkeypatch: pytest.MonkeyPatch, *lines: str) -> None:
        """Run the plain REPL, typing `lines` at the prompt."""
        typed: Iterator[str] = iter(lines)
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(typed))
        await run_plain(
            cfg=ProviderConfig(provider="ollama", model=None),
            max_turns=5,
            command_timeout=1.0,
            skills=SkillLibrary({}),
            context=None,
        )

    @pytest.mark.parametrize("command", ["/new", "/clear"])
    async def test_new_starts_a_fresh_history(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: RecordingProvider,
        command: str,
    ) -> None:
        """/new and its alias /clear drop the conversation but keep the system prompt."""
        await self.run(monkeypatch, "a", command, "b", "exit")

        system = provider.calls[0][0]["content"]
        assert roles_and_contents(provider.calls[-1]) == [
            ("system", system),
            ("user", "b"),
        ]

    async def test_unknown_command_is_not_sent_and_reports_an_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider: RecordingProvider,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        await self.run(monkeypatch, "/nope", "exit")

        assert provider.calls == []
        assert "Unknown command: /nope" in capsys.readouterr().out

    async def test_unknown_command_keeps_the_history(
        self, monkeypatch: pytest.MonkeyPatch, provider: RecordingProvider
    ) -> None:
        await self.run(monkeypatch, "a", "/nope", "b", "exit")

        system = provider.calls[0][0]["content"]
        assert roles_and_contents(provider.calls[-1]) == [
            ("system", system),
            ("user", "a"),
            ("assistant", "<action><exit>hi</exit></action>"),
            ("user", "b"),
        ]

    async def test_a_path_is_sent_as_a_prompt(
        self, monkeypatch: pytest.MonkeyPatch, provider: RecordingProvider
    ) -> None:
        """A first word containing another slash is a path, not a command."""
        await self.run(monkeypatch, "/etc/hosts is missing", "exit")

        assert len(provider.calls) == 1
        assert provider.calls[0][-1]["content"] == "/etc/hosts is missing"
