"""The Skill tool: load task instructions from <dir>/*/SKILL.md files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str  # frontmatter stripped
    dir: Path  # absolute; the skill's relative paths resolve against it


@dataclass(frozen=True)
class SkillLibrary:
    skills: dict[str, Skill]

    def catalog(self) -> str:
        """One line per skill: name and description. Empty string when there are no skills."""
        return "\n".join(f"- {s.name}: {s.description}" for s in self.skills.values())

    def resolve(self, name: str) -> Skill | None:
        return self.skills.get(name)


@dataclass(frozen=True)
class SkillResult:
    name: str  # what the model asked for
    skill: Skill | None  # None = unknown name
    available: tuple[str, ...]  # for the error message

    @property
    def status(self) -> Literal["ok", "failed"]:
        return "ok" if self.skill is not None else "failed"


@dataclass(frozen=True, kw_only=True)
class SkillCall:
    kind: ClassVar[str] = "Skill"
    id: str
    name: str

    @property
    def subject(self) -> str:
        return self.name

    @property
    def display(self) -> str:
        return f"{self.kind}: {self.subject}"


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return (metadata dict, body string) for a ----delimited frontmatter file."""
    if not text.startswith("---\n"):
        return {}, text
    front, _, body = text.partition("\n---")  # first --- closes the header
    return yaml.safe_load(front) or {}, body.lstrip("\n")


def parse_skill(text: str, source: Path) -> Skill:
    """Turn SKILL.md text into a Skill. Raises ValueError if frontmatter is missing."""
    meta, body = split_frontmatter(text)
    missing = {"name", "description"} - set(meta)
    if missing:
        raise ValueError(f"{source} missing required frontmatter: {missing}")
    return Skill(meta["name"], meta["description"], body, dir=source.parent.resolve())


def load_skills(skills_dir: Path | None) -> SkillLibrary:
    """Read and parse every <skills_dir>/*/SKILL.md. Empty library if no dir is given."""
    if skills_dir is None:
        return SkillLibrary({})
    skills: dict[str, Skill] = {}
    for path in sorted(skills_dir.glob("*/SKILL.md")):
        skill = parse_skill(path.read_text(encoding="utf-8"), path)
        if skill.name in skills:
            raise ValueError(
                f"Duplicate skill name `{skill.name}`: {skills[skill.name].dir} and {skill.dir}"
            )
        skills[skill.name] = skill
    return SkillLibrary(skills)


def format_skill_result(result: SkillResult) -> str:
    """Turn a SkillResult into the text the model reads."""
    if result.skill is None:
        available = ", ".join(result.available) or "none"
        return f"Unknown skill `{result.name}`. Available: {available}"
    skill = result.skill
    return (
        f"Skill `{skill.name}` loaded. Its directory is {skill.dir}; relative paths "
        f"in this skill are relative to it. Read or run those files with <bash>, "
        f"using absolute paths.\n\n{skill.body}"
    )


async def run_skill_tool(call: SkillCall, *, library: SkillLibrary) -> SkillResult:
    return SkillResult(call.name, library.resolve(call.name), tuple(library.skills))
