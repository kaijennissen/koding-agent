"""The Textual front end, driven by a fake model and fake tools through Textual's Pilot.

Fixture chain: `script` -> `llm` -> `app` -> `pilot`. A test changes what the
model says with `@pytest.mark.parametrize("script", [...])`; a class that needs
a special model overrides `llm`.
"""

import asyncio
from collections.abc import AsyncIterator

import pytest
from textual.app import App, ComposeResult
from textual.color import Color
from textual.containers import VerticalScroll
from textual.pilot import Pilot
from textual.widgets import Static

from fakes import ok, replies
from koding_agent.core import (
    BashCall,
    BashResult,
    Event,
    LlmCall,
    LlmError,
    LlmReply,
    Message,
    SkillCall,
    SkillResult,
    ToolCall,
    ToolCancelled,
    ToolEnd,
    ToolStart,
    TurnStart,
)
from koding_agent.tui.app import (
    NO_SELECTION,
    AgentApp,
    PromptArea,
    Status,
    ToolRow,
    describe,
)

MODEL = "some-model"
EXIT_HI = "<action><exit>hi</exit></action>"
BASH_TRUE = "<action><bash>true</bash></action>"

LS = BashCall(id="a", cmd="ls")
FAILED = BashResult.completed("ls", "", "boom", 1)


# --- fixtures and helpers ------------------------------------------------


@pytest.fixture
def script() -> list[str | Exception]:
    return [EXIT_HI]


@pytest.fixture
def llm(script: list[str | Exception]) -> LlmCall:
    return replies(*script)


@pytest.fixture
def app(llm: LlmCall) -> AgentApp:
    return AgentApp(llm, model=MODEL)


@pytest.fixture
async def pilot(app: AgentApp) -> AsyncIterator[Pilot[None]]:
    async with app.run_test() as pilot:
        yield pilot


async def settle(app: AgentApp, pilot: Pilot[None]) -> None:
    """Wait for the running turn, then let the messages it posted be handled."""
    await app.workers.wait_for_complete()  # pyright: ignore[reportUnknownMemberType]
    await pilot.pause()


async def submit(app: AgentApp, pilot: Pilot[None], text: str) -> None:
    await pilot.press(*text, "enter")
    await settle(app, pilot)


def texts(app: App[None], selector: str) -> list[str]:
    return [str(w.render()) for w in app.query(selector)]


def status_line(app: AgentApp) -> str:
    return str(app.query_one("#status", Static).render())


def prompt_text(app: AgentApp) -> str:
    return app.query_one("#prompt", PromptArea).text


def row_for(app: AgentApp, id: str) -> ToolRow:
    (row,) = [r for r in app.query(ToolRow) if r.tool.id == id]
    return row


# --- layout and prompt ---------------------------------------------------


class TestLayout:
    async def test_banner_opens_the_conversation_and_prompt_has_focus(
        self, app: AgentApp, pilot: Pilot[None]
    ) -> None:
        banner = app.query_one("#banner", Static)
        assert MODEL in str(banner.render())
        assert app.query_one("#conversation", VerticalScroll).children[0] is banner
        assert app.query_one("#prompt", PromptArea).has_focus

    @pytest.mark.parametrize(
        "screen_width,tools_width",
        [
            pytest.param(80, 32, id="narrow-uses-min"),
            pytest.param(200, 60, id="30-percent"),
            pytest.param(400, 80, id="wide-uses-max"),
        ],
    )
    async def test_tools_panel_scales_with_the_screen(
        self, app: AgentApp, screen_width: int, tools_width: int
    ) -> None:
        async with app.run_test(size=(screen_width, 30)):
            assert app.query_one("#tools").outer_size.width == tools_width

    async def test_tab_toggles_tools_panel(
        self, app: AgentApp, pilot: Pilot[None]
    ) -> None:
        """Tab is a priority binding, so it wins over the focused TextArea."""
        tools = app.query_one("#tools")
        displayed: list[bool] = []
        for _ in range(2):
            await pilot.press("tab")
            displayed.append(tools.display)
        assert displayed == [False, True]


class TestPrompt:
    @pytest.mark.parametrize(
        "keys,echo",
        [
            pytest.param([*"hello", "enter"], "> hello", id="single-line"),
            pytest.param(["a", "ctrl+j", "b", "enter"], "> a\nb", id="multi-line"),
        ],
    )
    async def test_enter_submits_the_whole_text_and_clears_the_prompt(
        self, app: AgentApp, pilot: Pilot[None], keys: list[str], echo: str
    ) -> None:
        await pilot.press(*keys)
        await settle(app, pilot)
        assert texts(app, ".message.user") == [echo]
        assert prompt_text(app) == ""

    @pytest.mark.parametrize(
        "keys,left_in_prompt",
        [
            pytest.param(["space", "enter"], "", id="blank-enter"),
            pytest.param(["a", "ctrl+j", "b"], "a\nb", id="ctrl+j-is-a-newline"),
        ],
    )
    async def test_does_not_submit(
        self, app: AgentApp, pilot: Pilot[None], keys: list[str], left_in_prompt: str
    ) -> None:
        await pilot.press(*keys)
        await settle(app, pilot)
        assert texts(app, ".message") == []
        assert prompt_text(app) == left_in_prompt


# --- tool rows -----------------------------------------------------------


class RowApp(App[None]):
    """Hosts a single ToolRow, to test the widget on its own."""

    def __init__(self, row: ToolRow) -> None:
        super().__init__()
        self.row = row

    def compose(self) -> ComposeResult:
        yield self.row


class TestToolRow:
    @pytest.fixture
    def cmd(self) -> str:
        return "ls"

    @pytest.fixture
    def tool(self, cmd: str) -> ToolCall:
        return BashCall(id="t1", cmd=cmd)

    @pytest.fixture
    async def row(self, tool: ToolCall) -> AsyncIterator[ToolRow]:
        app = RowApp(ToolRow(tool))
        async with app.run_test():
            yield app.row

    @staticmethod
    def status_classes(row: ToolRow) -> set[str]:
        return set(row.classes) & {"running", "ok", "failed", "cancelled"}

    async def test_starts_running(self, row: ToolRow) -> None:
        assert self.status_classes(row) == {"running"}

    @pytest.mark.parametrize("status", ["ok", "failed", "cancelled"])
    async def test_set_status_replaces_the_css_class(
        self, row: ToolRow, status: Status
    ) -> None:
        """The colour lives in CSS, so exactly one status class may be set."""
        row.set_status(status)
        assert self.status_classes(row) == {status}

    async def test_each_status_has_its_own_bullet_colour(self, row: ToolRow) -> None:
        """A missing CSS rule would leave the class set but the colour unchanged."""
        bullet = row.query_one(".bullet")
        colours = {"running": bullet.styles.color}
        statuses: list[Status] = ["ok", "failed", "cancelled"]
        for status in statuses:
            row.set_status(status)
            colours[status] = bullet.styles.color
        assert colours["running"] == Color.parse("yellow")
        assert len(set(colours.values())) == 4

    @pytest.mark.parametrize(
        "cmd,shown",
        [
            pytest.param("echo a\necho b", "echo a echo b", id="one-line"),
            pytest.param("echo [b]hi[/b]", "echo [b]hi[/b]", id="no-markup"),
        ],
    )
    async def test_shows_the_command(self, row: ToolRow, shown: str) -> None:
        assert str(row.query_one(".cmd", Static).render()) == shown

    @pytest.mark.parametrize(
        "tool,kind,subject",
        [
            pytest.param(BashCall(id="t1", cmd="ls"), "Bash", "ls", id="bash"),
            pytest.param(SkillCall(id="t1", name="pdf"), "Skill", "pdf", id="skill"),
        ],
    )
    async def test_shows_kind_and_subject(
        self, row: ToolRow, kind: str, subject: str
    ) -> None:
        assert str(row.query_one(".kind", Static).render()) == kind
        assert str(row.query_one(".cmd", Static).render()) == subject

    async def test_tooltip_shows_the_full_command(self, row: ToolRow) -> None:
        assert row.tooltip == "Bash: ls"

    async def test_kind_is_muted(self, row: ToolRow) -> None:
        """The kind is secondary information: dimmer and italic, not bold."""
        kind, cmd = row.query_one(".kind"), row.query_one(".cmd")
        assert kind.styles.color != cmd.styles.color
        assert kind.styles.text_style.italic
        assert not kind.styles.text_style.bold


class TestDescribe:
    """The details text is a pure function, so it is tested without an app."""

    def test_running_bash_shows_the_full_multiline_command(self) -> None:
        text = describe(BashCall(id="a", cmd="echo a\necho b"), "running", None)
        assert text == "Bash · running\n$ echo a\necho b"

    def test_finished_bash_shows_exit_code_and_output(self) -> None:
        text = describe(LS, "failed", BashResult.completed("ls", "out\n", "boom", 1))
        assert "exit code: 1" in text
        assert "── stdout ──\nout" in text
        assert "── stderr ──\nboom" in text

    def test_timeout_is_reported(self) -> None:
        assert "time limit" in describe(LS, "failed", BashResult.timeout("ls"))

    def test_unknown_skill_lists_the_available_ones(self) -> None:
        result = SkillResult(name="x", skill=None, available=("pdf", "show-me"))
        text = describe(SkillCall(id="s", name="x"), "failed", result)
        assert "unknown skill; available: pdf, show-me" in text


class TestToolInspection:
    """Clicking or arrowing through the tool list shows the details below it."""

    @staticmethod
    def details(app: AgentApp) -> str:
        return str(app.query_one("#tool-details", Static).render())

    async def start_two(self, app: AgentApp, pilot: Pilot[None]) -> None:
        app.forward_event(ToolStart(LS))
        app.forward_event(ToolEnd(LS, FAILED))
        app.forward_event(ToolStart(BashCall(id="b", cmd="pwd")))
        await pilot.pause()

    async def test_nothing_is_selected_at_first(
        self, app: AgentApp, pilot: Pilot[None]
    ) -> None:
        await self.start_two(app, pilot)
        assert self.details(app) == NO_SELECTION

    async def test_click_shows_the_details(
        self, app: AgentApp, pilot: Pilot[None]
    ) -> None:
        await self.start_two(app, pilot)
        await pilot.click(row_for(app, "a"))
        await pilot.pause()
        assert "$ ls" in self.details(app)
        assert "boom" in self.details(app)

    async def test_ctrl_t_starts_on_the_latest_and_arrows_navigate(
        self, app: AgentApp, pilot: Pilot[None]
    ) -> None:
        await self.start_two(app, pilot)
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert self.details(app).startswith("Bash · running\n$ pwd")
        await pilot.press("up")
        await pilot.pause()
        assert "$ ls" in self.details(app)

    async def test_escape_returns_to_the_prompt(
        self, app: AgentApp, pilot: Pilot[None]
    ) -> None:
        await self.start_two(app, pilot)
        await pilot.press("ctrl+t", "escape")
        assert app.query_one("#prompt", PromptArea).has_focus

    async def test_details_update_when_the_shown_tool_finishes(
        self, app: AgentApp, pilot: Pilot[None]
    ) -> None:
        await self.start_two(app, pilot)
        await pilot.press("ctrl+t")
        app.forward_event(ToolEnd(BashCall(id="b", cmd="pwd"), ok("pwd")))
        await pilot.pause()
        assert "exit code: 0" in self.details(app)
        assert "pwd output" in self.details(app)


class TestCoreEvents:
    """forward_event turns core events into rows and status text."""

    async def test_tool_start_mounts_a_running_row_in_the_tools_panel(
        self, app: AgentApp, pilot: Pilot[None]
    ) -> None:
        app.forward_event(ToolStart(LS))
        await pilot.pause()
        (row,) = app.query_one("#tools").query(ToolRow)
        assert row.status == "running"
        assert row.tool == LS

    @pytest.mark.parametrize(
        "end,status",
        [
            pytest.param(ToolEnd(LS, ok("ls")), "ok", id="end-ok"),
            pytest.param(ToolEnd(LS, FAILED), "failed", id="end-failed"),
            pytest.param(ToolCancelled(LS), "cancelled", id="cancelled"),
        ],
    )
    async def test_end_event_sets_the_row_status(
        self, app: AgentApp, pilot: Pilot[None], end: Event, status: Status
    ) -> None:
        app.forward_event(ToolStart(LS))
        app.forward_event(end)
        await pilot.pause()
        assert row_for(app, "a").status == status

    async def test_parallel_tools_are_matched_by_id(
        self, app: AgentApp, pilot: Pilot[None]
    ) -> None:
        b = BashCall(id="b", cmd="false")
        app.forward_event(ToolStart(LS))
        app.forward_event(ToolStart(b))
        app.forward_event(ToolEnd(b, FAILED))  # b finishes first
        await pilot.pause()
        assert (row_for(app, "a").status, row_for(app, "b").status) == (
            "running",
            "failed",
        )

    async def test_turn_start_updates_the_status_line(
        self, app: AgentApp, pilot: Pilot[None]
    ) -> None:
        app.forward_event(TurnStart(0, 5))
        await pilot.pause()
        assert status_line(app) == "Working… turn 1/5"


# --- running turns -------------------------------------------------------


class TestTurn:
    """A submitted message runs the real run_agent with the fake model."""

    @pytest.mark.parametrize(
        "script,shown,roles",
        [
            pytest.param(
                [EXIT_HI],
                (".message.assistant", "hi"),
                ["system", "user", "assistant"],
                id="answer",
            ),
            pytest.param(
                [LlmError("boom")],
                (".message.error", "boom"),
                ["system", "user"],  # kept, so the user can retry
                id="llm-error",
            ),
        ],
    )
    async def test_outcome_is_shown_and_the_turn_cleaned_up(
        self,
        app: AgentApp,
        pilot: Pilot[None],
        shown: tuple[str, str],
        roles: list[str],
    ) -> None:
        await submit(app, pilot, "yo")
        selector, text = shown
        assert texts(app, selector) == [text]
        assert [m["role"] for m in app.messages] == roles
        assert not app.busy
        assert status_line(app) == ""

    @pytest.mark.parametrize("script", [[BASH_TRUE, EXIT_HI]])
    async def test_tool_turn_leaves_a_finished_row(
        self, app: AgentApp, pilot: Pilot[None]
    ) -> None:
        await submit(app, pilot, "go")
        (row,) = app.query(ToolRow)
        assert row.status == "ok"
        assert texts(app, ".message.assistant") == ["hi"]

    @pytest.mark.parametrize("script", [[EXIT_HI, EXIT_HI]])
    async def test_next_turn_continues_the_history(
        self, app: AgentApp, pilot: Pilot[None]
    ) -> None:
        await submit(app, pilot, "a")
        await submit(app, pilot, "b")
        assert [m["content"] for m in app.messages[1:]] == ["a", EXIT_HI, "b", EXIT_HI]

    async def test_first_prompt_is_submitted_on_start(self) -> None:
        app = AgentApp(replies(EXIT_HI), model=MODEL, first_prompt="hello")
        async with app.run_test() as pilot:
            await settle(app, pilot)
            assert texts(app, ".message.user") == ["> hello"]
            assert texts(app, ".message.assistant") == ["hi"]


class TestWhileBusy:
    """The model blocks until the test releases it, so the turn stays running."""

    @pytest.fixture
    def release(self) -> asyncio.Event:
        return asyncio.Event()

    @pytest.fixture
    def llm(self, release: asyncio.Event) -> LlmCall:
        async def blocked(_messages: list[Message]) -> LlmReply:
            await release.wait()
            return LlmReply(text=EXIT_HI)

        return blocked

    async def test_submit_keeps_the_text_and_starts_no_second_turn(
        self, app: AgentApp, pilot: Pilot[None], release: asyncio.Event
    ) -> None:
        await pilot.press(*"one", "enter")
        await pilot.press(*"two", "enter")
        await pilot.pause()
        assert app.busy
        assert prompt_text(app) == "two"
        assert texts(app, ".message.user") == ["> one"]

        release.set()
        await settle(app, pilot)
        assert not app.busy
        assert texts(app, ".message.assistant") == ["hi"]
