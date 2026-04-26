"""Tests for reasoning_effort -> Claude Agent SDK option mapping.

Without these, the claude-code backend silently dropped the user's
configured effort, defaulting whatever the SDK picks for a given model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.config import Config
from agent.core.claude_code_backend import (
    _claude_code_effort,
    _claude_code_thinking,
)


def _session(effort: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        config=Config(model_name="claude-code/opus", reasoning_effort=effort)
    )


@pytest.mark.parametrize(
    "effort,expected",
    [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "max"),
        ("max", "max"),
        (None, None),
        ("garbage", None),
    ],
)
def test_effort_mapping(effort, expected):
    assert _claude_code_effort(_session(effort)) == expected


def test_thinking_adaptive_when_effort_set():
    assert _claude_code_thinking("max") == {"type": "adaptive"}
    assert _claude_code_thinking("low") == {"type": "adaptive"}


def test_thinking_disabled_when_effort_none():
    assert _claude_code_thinking(None) == {"type": "disabled"}
