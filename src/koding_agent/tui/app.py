"""Textual front end: conversation on the left, tool calls on the right, prompt below.

The app owns the conversation history and runs the core's `run_agent` as a
worker on Textual's event loop. The core reports progress through `on_event`;
`forward_event` turns each event into a Textual message, and the handlers
update the widgets.
"""

from typing import Literal, assert_never

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Footer, Static, TextArea

from koding_agent.banner import render_banner
from koding_agent.core.agent import (
    DEFAULT_MAX_TURNS,
<<<<<<< HEAD
    SYSTEM_PROMPT,
=======
    BashResult,
    Context,
    Event,
>>>>>>> origin/main
    LlmCall,
    LlmError,
    run_agent,
)
from koding_agent.core.agent import (
    Message as ChatMessage,
)
from koding_agent.core.events import (
    Event,
    ToolCancelled,
    ToolEnd,
    ToolStart,
    TurnStart,
<<<<<<< HEAD
=======
    build_system_prompt,
    run_agent,
>>>>>>> origin/main
)
from koding_agent.core.tool_types import BashResult, ToolCall
from koding_agent.core.tools import RunTool

DEFAULT_MODEL = "unknown-model"

type Status = Literal["running", "ok", "failed", "cancelled"]


def status_for(result: BashResult) -> Status:
    """The one place that decides what counts as a failed tool call."""
    if result.timed_out or result.returncode != 0:
        return "failed"
    return "ok"


class ToolRow(Horizontal):
    """One tool call: a coloured bullet and the command on a single line.

    The status is a CSS class (`running`, `ok`, `failed`, `cancelled`), so the
    colours live in CSS and tests can assert on the class.
    """

    DEFAULT_CSS = """
    ToolRow { height: 1; }
    ToolRow .bullet { width: 2; }
    ToolRow .cmd { width: 1fr; text-wrap: nowrap; text-overflow: ellipsis; }
    ToolRow.running .bullet { color: yellow; }
    ToolRow.ok .bullet { color: $success; }
    ToolRow.failed .bullet { color: $error; }
    ToolRow.cancelled .bullet { color: $text-muted; }
    """

    def __init__(self, call: ToolCall) -> None:
        super().__init__(classes="running")
        self.call = call
        self.status: Status = "running"

    def compose(self) -> ComposeResult:
        yield Static("●", classes="bullet")
        yield Static(self.call.cmd.replace("\n", " "), classes="cmd", markup=False)

    def set_status(self, status: Status) -> None:
        self.remove_class(self.status)
        self.add_class(status)
        self.status = status


class ToolPanel(VerticalScroll):
    """Right-hand column with one ToolRow per tool call; Tab hides it."""

    def on_mount(self) -> None:
        self.border_title = "Tools"


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
    def __init__(self, call: ToolCall) -> None:
        super().__init__()
        self.call = call


class ToolFinished(Message):
    def __init__(self, call: ToolCall, result: BashResult) -> None:
        super().__init__()
        self.call = call
        self.result = result


class ToolAborted(Message):
    def __init__(self, call: ToolCall) -> None:
        super().__init__()
        self.call = call


class TurnStarted(Message):
    def __init__(self, turn: int, max_turns: int) -> None:
        super().__init__()
        self.turn = turn
        self.max_turns = max_turns


class TurnFinished(Message):
    """Posted after a turn ends, so it is handled after that turn's other messages."""


class AgentApp(App[None]):
    """One conversation with the agent; at most one turn runs at a time.

    The model call and the tool runner are passed in, so tests can use fakes.
    """

    CSS = """
    #main { height: 1fr; }
    #conversation { width: 1fr; border: round $primary; padding: 0 1; }
    #tools { width: 32; border: round $secondary; padding: 0 1; }
    #tools.collapsed { display: none; }
    #status { height: 1; padding: 0 1; color: $text-muted; }
    #prompt { height: auto; min-height: 3; max-height: 10; }
    .message { margin: 1 0 0 0; }
    .message.user { background: $boost; padding: 0 1; }
    .message.error { color: $error; }
    """

    BINDINGS = [
        Binding("tab", "toggle_tools", "Tools", priority=True),
        Binding("ctrl+c", "quit", "Exit", priority=True),
    ]

    def __init__(
        self,
        call: LlmCall,
        tool_fn: RunTool,
        model: str = DEFAULT_MODEL,
        max_turns: int = DEFAULT_MAX_TURNS,
        first_prompt: str | None = None,
        context: Context | None = None,
    ) -> None:
        super().__init__()
        self.context = context
        self.first_prompt = first_prompt
        self.call = call
        self.tool_fn = tool_fn
        self.model = model
        self.max_turns = max_turns
        self.messages: list[ChatMessage] = [
            {"role": "system", "content": build_system_prompt(context)}
        ]
        self.busy = False
        self.rows: dict[str, ToolRow] = {}  # call.id -> row

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

    def forward_event(self, event: Event) -> None:
        """The core's OnEvent callback: translate to a message and return at once."""
        match event:
            case ToolStart(call):
                self.post_message(ToolStarted(call))
            case ToolEnd(call, result):
                self.post_message(ToolFinished(call, result))
            case ToolCancelled(call):
                self.post_message(ToolAborted(call))
            case TurnStart(turn, max_turns):
                self.post_message(TurnStarted(turn, max_turns))
            case _:
                assert_never(event)

    async def on_tool_started(self, message: ToolStarted) -> None:
        row = ToolRow(message.call)
        self.rows[message.call.id] = row
        await self.query_one("#tools", ToolPanel).mount(row)

    def on_tool_finished(self, message: ToolFinished) -> None:
        # A missing id means an end without a start: a bug, so let it raise.
        self.rows[message.call.id].set_status(status_for(message.result))

    def on_tool_aborted(self, message: ToolAborted) -> None:
        self.rows[message.call.id].set_status("cancelled")

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
        if self.busy:
            self.notify("Agent is busy…", severity="warning")  # keep the typed text
            return
        prompt.clear()
        await self.add_message(f"> {text}", "user")
        self.messages.append({"role": "user", "content": text})
        self.busy = True
        self.run_worker(self.run_turn(), exclusive=True)

    async def run_turn(self) -> None:
        """Worker body: run the agent on the history and show its answer or error."""
        try:
            answer, _ = await run_agent(
                self.messages,
                self.max_turns,
                self.call,
                self.tool_fn,
                on_event=self.forward_event,
            )
            await self.add_message(answer, "assistant")
        except LlmError as e:
            # Keep the session: the failed call appended nothing, so the history
            # is still valid and the user can retry. Only LlmError is expected;
            # anything else is a bug and should crash loudly.
            await self.add_message(str(e), "error")
        finally:
            self.post_message(TurnFinished())
