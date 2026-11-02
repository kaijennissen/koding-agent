"""Plain data types for tool calls and their results; imports nothing from the core."""

from dataclasses import dataclass
from typing import Self


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


@dataclass(frozen=True, kw_only=True)
class BashCall:
    id: str
    cmd: str


@dataclass(frozen=True, kw_only=True)
class SkillCall:
    id: str
    name: str


type ToolCall = BashCall
