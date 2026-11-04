"""The bash tool: call, result, formatting and runner."""

import asyncio
import os
import signal
from dataclasses import dataclass
from typing import ClassVar, Literal, Self

MAX_OUTPUT_CHARS = 2000  # tool output is context we pay for on every later turn
DEFAULT_COMMAND_TIMEOUT = (
    120  # default seconds; a wall-clock limit per command, not per loop
)


@dataclass(frozen=True)
class BashResult:
    cmd: str
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool

    @classmethod
    def completed(
        cls, cmd: str, stdout: str, stderr: str, returncode: int | None
    ) -> Self:
        return cls(cmd, stdout, stderr, returncode, timed_out=False)

    @classmethod
    def timeout(cls, cmd: str, stdout: str = "", stderr: str = "") -> Self:
        return cls(cmd, stdout, stderr, returncode=None, timed_out=True)

    @property
    def status(self) -> Literal["ok", "failed"]:
        if self.timed_out or self.returncode != 0:
            return "failed"
        return "ok"


@dataclass(frozen=True, kw_only=True)
class BashCall:
    kind: ClassVar[str] = "Bash"
    id: str
    cmd: str

    @property
    def subject(self) -> str:
        return self.cmd.replace("\n", " ")

    @property
    def display(self) -> str:
        return f"{self.kind}: {self.subject}"


def format_bash_result(result: BashResult) -> str:
    """Turn a BashResult into the text the model reads."""
    if result.timed_out:
        return (
            f"Command `{result.cmd}` was killed because it hit the time limit. "
            "It did not finish, so its result and side effects are unknown, "
            "and any output it produced was discarded."
        )
    return (
        f"Command `{result.cmd}` finished with exit code {result.returncode}.\n"
        f'stdout: "{truncate(result.stdout)}"\n'
        f'stderr: "{truncate(result.stderr)}"'
    )


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Keep the first `limit` characters and say how much was cut."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[... truncated: showing {limit} of {len(text)} characters]"


async def run_bash_tool(tool: BashCall, *, timeout: float) -> BashResult:
    proc = await asyncio.create_subprocess_shell(
        tool.cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    proc_id = proc.pid

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return BashResult.completed(
            cmd=tool.cmd,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            returncode=proc.returncode,
        )

    except TimeoutError:
        return BashResult.timeout(tool.cmd)

    finally:
        try:
            os.killpg(proc_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()
