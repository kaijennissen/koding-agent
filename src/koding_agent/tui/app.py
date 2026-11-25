"""Textual front end: conversation on the left, tool calls on the right, prompt below.

The app owns the conversation history and runs the core's `run_agent` as a
worker on Textual's event loop. The core reports progress through `on_event`;
`forward_event` turns each event into a Textual message, and the handlers
update the widgets.
"""

import contextlib
from typing import Literal, assert_never

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Footer, ListItem, ListView, Static, TextArea
from textual.worker import WorkerCancelled

from koding_agent.banner import render_banner
from koding_agent.commands import canonical, parse_slash
from koding_agent.core import (
    DEFAULT_COMMAND_TIMEOUT,
    DEFAULT_MAX_TURNS,
    BashCall,
    BashResult,
    Context,
    Event,
    LlmCall,
    LlmError,
    OutOfTurns,
    SkillCall,
    SkillLibrary,
    SkillResult,
    ToolCall,
    ToolCancelled,
    ToolEnd,
    ToolResult,
    ToolStart,
    TurnStart,
    initial_messages,
    run_agent,
)
from koding_agent.core import (
    Message as ChatMessage,
)

type Status = Literal["running", "ok", "failed", "cancelled"]


class ToolRow(ListItem):
    """One tool call: a coloured bullet and the command on a single line.

    It is a ListItem, so the ToolPanel's ListView handles clicks and Up/Down.
    The status is a CSS class (`running`, `ok`, `failed`, `cancelled`), so the
    colours live in CSS and tests can assert on the class.
    """

    DEFAULT_CSS = """
    ToolRow { layout: horizontal; height: 1; }
    ToolRow .bullet { width: 2; }
    ToolRow .kind { width: auto; margin-right: 1; color: $text-muted; text-style: italic; }
    ToolRow .cmd { width: 1fr; text-wrap: nowrap; text-overflow: ellipsis; }
    ToolRow.running .bullet { color: yellow; }
    ToolRow.ok .bullet { color: $success; }
    ToolRow.failed .bullet { color: $error; }
    ToolRow.cancelled .bullet { color: $text-muted; }
    """

    def __init__(self, tool: ToolCall) -> None:
        super().__init__(classes="running")
        self.tool = tool
        self.status: Status = "running"
        self.result: ToolResult | None = None  # set when the tool finishes

    def compose(self) -> ComposeResult:
        yield Static("●", classes="bullet")
        yield Static(self.tool.kind, classes="kind", markup=False)
        yield Static(self.tool.subject, classes="cmd", markup=False)

    def on_mount(self) -> None:
        self.tooltip = self.tool.display  # hover shows the full command

    def set_status(self, status: Status) -> None:
        self.remove_class(self.status)
        self.add_class(status)
        self.status = status


def describe(tool: ToolCall, status: Status, result: ToolResult | None) -> str:
    """The details text for one tool call: full command, outcome and output."""
    lines = [f"{tool.kind} · {status}"]
    match tool:
        case BashCall(cmd=cmd):
            lines.append(f"$ {cmd}")  # the real command, newlines kept
        case SkillCall(name=name):
            lines.append(f"skill: {name}")
    match result:
        case None:
            pass
        case BashResult(timed_out=True):
            lines.append("killed: hit the time limit")
        case BashResult(returncode=code, stdout=stdout, stderr=stderr):
            lines.append(f"exit code: {code}")
            if stdout:
                lines += ["", "── stdout ──", stdout.rstrip()]
            if stderr:
                lines += ["", "── stderr ──", stderr.rstrip()]
        case SkillResult(skill=skill, available=available):
            if skill is None:
                lines.append(f"unknown skill; available: {', '.join(available) or '-'}")
            else:
                lines.append(f"loaded from {skill.dir}")
    return "\n".join(lines)


NO_SELECTION = "Click a tool call, or press Ctrl+T and use ↑/↓."


class ToolPanel(Vertical):
    """Right-hand column: tool calls on top, details of the highlighted one below.

    Tab hides it; Ctrl+T focuses the list, Escape goes back to the prompt.
    """

    DEFAULT_CSS = """
    ToolPanel #tool-list { height: 1fr; background: transparent; }
    ToolPanel #details { height: 2fr; border-top: solid $secondary; }
    """

    def compose(self) -> ComposeResult:
        yield ListView(id="tool-list", initial_index=None)
        with VerticalScroll(id="details"):
            yield Static(NO_SELECTION, id="tool-details", markup=False)

    def on_mount(self) -> None:
        self.border_title = "Tools"

    async def add(self, row: ToolRow) -> None:
        await self.query_one("#tool-list", ListView).append(row)

    def show(self, row: ToolRow | None) -> None:
        text = describe(row.tool, row.status, row.result) if row else NO_SELECTION
        self.query_one("#tool-details", Static).update(Text(text))  # Text: no markup

    def refresh_row(self, row: ToolRow) -> None:
        """Called after a row changed; redraw the details if it is the shown one."""
        if row.highlighted:
            self.show(row)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        self.show(event.item if isinstance(event.item, ToolRow) else None)


class PromptArea(TextArea):
    """Multi-line prompt: Enter submits, Ctrl+J (or Shift+Enter) adds a newline.

    Shift+Enter only reaches us in terminals with an enhanced keyboard
    protocol; elsewhere it looks like Enter, so Ctrl+J is the dependable key.
    """

    BINDINGS = [
        Binding("enter", "submit", "Send", priority=True),
        Binding("ctrl+j,shift+enter", "newline", "Newline", priority=True),
    ]

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def action_submit(self) -> None:
        self.post_message(self.Submitted(self.text))

    def action_newline(self) -> None:
        self.insert("\n")


class ToolStarted(Message):
    def __init__(self, tool: ToolCall) -> None:
        super().__init__()
        self.tool = tool


class ToolFinished(Message):
    def __init__(self, tool: ToolCall, result: ToolResult) -> None:
        super().__init__()
        self.tool = tool
        self.result = result


class ToolAborted(Message):
    def __init__(self, tool: ToolCall) -> None:
        super().__init__()
        self.tool = tool


class TurnStarted(Message):
    def __init__(self, turn: int, max_turns: int) -> None:
        super().__init__()
        self.turn = turn
        self.max_turns = max_turns


class TurnFinished(Message):
    """Posted after a turn ends, so it is handled after that turn's other messages."""


class NewSession(Message):
    """Posted once the cancelled turn has finished, so the reset is handled last."""


class AgentApp(App[None]):
    """One conversation with the agent; at most one turn runs at a time.

    The model call and the skill library are passed in, so tests can use fakes.
    """

    CSS = """
    #main { height: 1fr; }
    #conversation { width: 1fr; border: round $primary; padding: 0 1; }
    #tools { width: 30%; min-width: 32; max-width: 80; border: round $secondary; padding: 0 1; }
    #tools.collapsed { display: none; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    #prompt { height: auto; min-height: 3; max-height: 10; }
    .message { margin: 1 0 0 0; }
    .message.user { background: $boost; padding: 0 1; }
    .message.error { color: $error; }
    """

    BINDINGS = [
        Binding("tab", "toggle_tools", "Tools", priority=True),
        Binding("ctrl+t", "focus_tools", "Inspect", priority=True),
        Binding("escape", "focus_prompt", "Back", show=False),
        Binding("ctrl+c", "quit", "Exit", priority=True),
    ]

    def __init__(
        self,
        call: LlmCall,
        model: str,  # display only; the model actually called is baked into `call`
        skills: SkillLibrary = SkillLibrary({}),
        max_turns: int = DEFAULT_MAX_TURNS,
        command_timeout: float = DEFAULT_COMMAND_TIMEOUT,
        first_prompt: str | None = None,
        context: Context | None = None,
    ) -> None:
        super().__init__()
        self.context = context
        self.first_prompt = first_prompt
        self.call = call
        self.model = model
        self.max_turns = max_turns
        self.command_timeout = command_timeout
        self.skills = skills
        self.messages: list[ChatMessage] = self.fresh_history()
        self.busy = False
        self.rows: dict[str, ToolRow] = {}  # tool.id -> row

    def fresh_history(self) -> list[ChatMessage]:
        return initial_messages(
            self.context, self.skills.catalog(), command_timeout=self.command_timeout
        )

    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            conversation = VerticalScroll(id="conversation")
            conversation.border_title = "Conversation"
            with conversation:
                yield Static(
                    Text.from_ansi(
                        render_banner(
                            self.model,
                            color=True,
                            context=self.context.path.name if self.context else None,
                            skills=self.skills.skills,
                        )
                    ),
                    id="banner",
                )
            yield ToolPanel(id="tools")
        yield Static("", id="status")
        prompt = PromptArea(placeholder="Send a message", id="prompt", soft_wrap=True)
        prompt.border_title = "Message"
        yield prompt
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#prompt", PromptArea).focus()
        if self.first_prompt:  # goes through the normal submit path
            self.post_message(PromptArea.Submitted(self.first_prompt))

    def action_toggle_tools(self) -> None:
        self.query_one("#tools").toggle_class("collapsed")

    def action_focus_tools(self) -> None:
        """Show the panel and focus the list; start on the latest tool call."""
        self.query_one("#tools").remove_class("collapsed")
        tool_list = self.query_one("#tool-list", ListView)
        if tool_list.index is None and len(tool_list):
            tool_list.index = len(tool_list) - 1
        tool_list.focus()

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt", PromptArea).focus()

    def forward_event(self, event: Event) -> None:
        """The core's OnEvent callback: translate to a message and return at once."""
        match event:
            case ToolStart(tool):
                self.post_message(ToolStarted(tool))
            case ToolEnd(tool, result):
                self.post_message(ToolFinished(tool, result))
            case ToolCancelled(tool):
                self.post_message(ToolAborted(tool))
            case TurnStart(turn, max_turns):
                self.post_message(TurnStarted(turn, max_turns))
            case _:
                assert_never(event)

    async def on_tool_started(self, message: ToolStarted) -> None:
        row = ToolRow(message.tool)
        self.rows[message.tool.id] = row
        await self.query_one("#tools", ToolPanel).add(row)

    def on_tool_finished(self, message: ToolFinished) -> None:
        # A missing id means an end without a start: a bug, so let it raise.
        row = self.rows[message.tool.id]
        row.result = message.result
        row.set_status(message.result.status)
        self.query_one("#tools", ToolPanel).refresh_row(row)

    def on_tool_aborted(self, message: ToolAborted) -> None:
        row = self.rows[message.tool.id]
        row.set_status("cancelled")
        self.query_one("#tools", ToolPanel).refresh_row(row)

    def on_turn_started(self, message: TurnStarted) -> None:
        self.query_one("#status", Static).update(
            f"Working… turn {message.turn + 1}/{message.max_turns}"
        )

    def on_turn_finished(self, message: TurnFinished) -> None:
        self.busy = False
        self.query_one("#status", Static).update("")

    async def add_message(self, text: str, role: str) -> None:
        conversation = self.query_one("#conversation", VerticalScroll)
        await conversation.mount(Static(text, classes=f"message {role}", markup=False))
        conversation.scroll_end(animate=False)

    async def on_prompt_area_submitted(self, event: PromptArea.Submitted) -> None:
        text = event.text.strip()
        prompt = self.query_one("#prompt", PromptArea)
        if not text:
            prompt.clear()
            return
        match parse_slash(text):
            case None:
                pass  # not a command: send it to the model
            case (name, _):
                prompt.clear()
                await self.run_command(name)
                return
        if self.busy:
            self.notify("Agent is busy…", severity="warning")  # keep the typed text
            return
        prompt.clear()
        await self.add_message(f"> {text}", "user")
        self.messages.append({"role": "user", "content": text})
        self.busy = True
        self.run_worker(self.run_turn, exclusive=True, group="turn")

    async def run_command(self, name: str) -> None:
        """Run the slash command `name` (as typed; aliases are resolved here)."""
        match canonical(name):
            case "new":
                await self.start_new_session()
            case _:
                await self.add_message(f"Unknown command: /{name}", "error")

    async def start_new_session(self) -> None:
        """Cancel the running turn, then post `NewSession` so the reset runs last."""
        for worker in [w for w in self.workers if w.group == "turn"]:
            worker.cancel()
            with contextlib.suppress(WorkerCancelled):
                await worker.wait()
        self.post_message(NewSession())

    async def on_new_session(self, message: NewSession) -> None:
        await self.reset_session()

    async def reset_session(self) -> None:
        """Start over: fresh history, empty tool panel, banner only, not busy.

        Runs after the turn was cancelled, so it also clears what a turn that
        never got to post `TurnFinished` left behind.
        """
        self.messages = self.fresh_history()
        self.busy = False
        self.query_one("#status", Static).update("")
        self.rows.clear()
        await self.query_one("#tool-list", ListView).clear()
        self.query_one("#tools", ToolPanel).show(None)
        conversation = self.query_one("#conversation", VerticalScroll)
        await conversation.remove_children(".message")
        self.query_one("#prompt", PromptArea).focus()

    async def run_turn(self) -> None:
        """Worker body: run the agent on the history and show its answer or error."""
        try:
            answer, _ = await run_agent(
                messages=self.messages,
                max_turns=self.max_turns,
                call=self.call,
                skills=self.skills,
                on_event=self.forward_event,
                command_timeout=self.command_timeout,
            )
            await self.add_message(answer, "assistant")
        except (LlmError, OutOfTurns) as e:
            # Keep the session: the history is still valid (LlmError appended
            # nothing; OutOfTurns kept the turns that ran), so the user can retry
            # or continue. Anything else is a bug and should crash loudly.
            await self.add_message(str(e), "error")
        finally:
            self.post_message(TurnFinished())
