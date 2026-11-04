"""Skills: parsing SKILL.md, loading a directory, the catalog, and what the model reads."""

from pathlib import Path

import pytest

from koding_agent.agent import (
    Skill,
    SkillCall,
    SkillLibrary,
    SkillResult,
    format_skill_result,
    load_skills,
    parse_skill,
    run_skill_tool,
)

PDF_MD = """---
name: pdf
description: Work with PDFs
---

Use pdftotext. See forms.md for forms.
"""

PDF = Skill("pdf", "Work with PDFs", "Use pdftotext.", Path("/skills/pdf"))
GIT = Skill("git", "Work with git", "Use git.", Path("/skills/git"))


def write_skill(root: Path, folder: str, text: str) -> Path:
    path = root / folder / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


class TestParseSkill:
    def test_splits_frontmatter_from_body(self, tmp_path: Path) -> None:
        source = tmp_path / "pdf" / "SKILL.md"

        skill = parse_skill(PDF_MD, source)

        assert skill == Skill(
            name="pdf",
            description="Work with PDFs",
            body="Use pdftotext. See forms.md for forms.\n",
            dir=(tmp_path / "pdf").resolve(),
        )

    @pytest.mark.parametrize(
        "text",
        [
            pytest.param("no frontmatter at all", id="no-frontmatter"),
            pytest.param("---\nname: pdf\n---\nbody", id="no-description"),
        ],
    )
    def test_missing_frontmatter_raises(self, text: str) -> None:
        with pytest.raises(ValueError, match="missing required frontmatter"):
            parse_skill(text, Path("x/SKILL.md"))


class TestLoadSkills:
    def test_no_dir_is_an_empty_library(self) -> None:
        assert load_skills(None) == SkillLibrary({})

    def test_loads_every_skill_md_one_level_deep(self, tmp_path: Path) -> None:
        write_skill(tmp_path, "pdf", PDF_MD)
        write_skill(tmp_path, "git", "---\nname: git\ndescription: Git\n---\nbody")
        (tmp_path / "README.md").write_text("not a skill")

        library = load_skills(tmp_path)

        assert list(library.skills) == ["git", "pdf"]  # sorted by folder

    def test_duplicate_names_raise(self, tmp_path: Path) -> None:
        write_skill(tmp_path, "a", PDF_MD)
        write_skill(tmp_path, "b", PDF_MD)

        with pytest.raises(ValueError, match="Duplicate skill name `pdf`"):
            load_skills(tmp_path)


class TestSkillLibrary:
    def test_catalog_has_one_line_per_skill(self) -> None:
        library = SkillLibrary({"pdf": PDF, "git": GIT})
        assert library.catalog() == "- pdf: Work with PDFs\n- git: Work with git"

    def test_empty_catalog_is_an_empty_string(self) -> None:
        assert SkillLibrary({}).catalog() == ""

    @pytest.mark.parametrize("name,skill", [("pdf", PDF), ("nope", None)])
    def test_resolve(self, name: str, skill: Skill | None) -> None:
        assert SkillLibrary({"pdf": PDF}).resolve(name) == skill


async def test_run_skill_tool_wraps_the_lookup() -> None:
    library = SkillLibrary({"pdf": PDF, "git": GIT})

    result = await run_skill_tool(SkillCall(id="1", name="pdf"), library=library)

    assert result == SkillResult("pdf", PDF, ("pdf", "git"))
    assert result.status == "ok"


class TestFormatSkillResult:
    def test_found_points_at_the_dir_and_includes_the_body(self) -> None:
        text = format_skill_result(SkillResult("pdf", PDF, ("pdf",)))

        assert "/skills/pdf" in text
        assert "<bash>" in text
        assert text.endswith("Use pdftotext.")

    @pytest.mark.parametrize(
        "available,listed",
        [(("pdf", "git"), "Available: pdf, git"), ((), "Available: none")],
    )
    def test_unknown_lists_what_is_available(
        self, available: tuple[str, ...], listed: str
    ) -> None:
        result = SkillResult("nope", None, available)

        assert result.status == "failed"
        assert format_skill_result(result) == f"Unknown skill `nope`. {listed}"
