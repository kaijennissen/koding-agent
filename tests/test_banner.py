"""The banner shows name, version and model, and stays plain without colour."""

from koding_agent import __version__
from koding_agent.banner import render_banner, render_prompt


def test_plain_banner_has_name_version_and_model() -> None:
    banner = render_banner("some-model", color=False)

    assert "koding agent" in banner
    assert f"v{__version__}" in banner
    assert "some-model" in banner
    assert "\033[" not in banner


def test_color_banner_uses_ansi_codes() -> None:
    assert "\033[" in render_banner("some-model", color=True)


def test_plain_prompt_is_three_chevrons() -> None:
    assert render_prompt(color=False, readline_active=False) == "❯❯❯ "


def test_color_prompt_wraps_codes_only_for_readline() -> None:
    raw = render_prompt(color=True, readline_active=False)
    wrapped = render_prompt(color=True, readline_active=True)

    assert raw.count("❯") == 3 and "\001" not in raw
    assert wrapped.count("\001") == wrapped.count("\002") == 4
    assert wrapped.replace("\001", "").replace("\002", "") == raw
