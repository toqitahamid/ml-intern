"""Every tool referenced in the skill body must exist in the MCP registry
or be a known Claude Code native / HF MCP tool, and no dropped tool names
must leak back into the skill."""

import re
from pathlib import Path

from plugin.mcp.ml_intern_tools import REGISTERED_TOOL_NAMES

SKILL_PATH = Path(__file__).resolve().parents[2] / "plugin" / "skills" / "ml-intern" / "SKILL.md"
SKILL_TEXT = SKILL_PATH.read_text()

TOKEN_RE = re.compile(r"\b([a-z][a-z0-9_]*)\b")


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text))


def test_every_registered_tool_is_mentioned():
    """The skill must teach the model to use every tool we register,
    otherwise the tool is dead weight in the plugin."""
    mentioned = _tokens(SKILL_TEXT) & REGISTERED_TOOL_NAMES
    unreferenced = REGISTERED_TOOL_NAMES - mentioned
    assert not unreferenced, (
        f"MCP tools never mentioned in the skill: {unreferenced}. "
        "Either reference them in the skill or drop them from the registry."
    )


def test_no_dropped_tool_names_in_skill():
    forbidden = {"plan_tool", "sandbox_create", "hf_jobs", "hub_repo_details"}
    hits = forbidden & _tokens(SKILL_TEXT)
    assert not hits, f"dropped tools still referenced in skill: {hits}"


def test_num_tools_template_resolved():
    assert "{{ num_tools }}" not in SKILL_TEXT
