"""Regression tests for the claude-code backend approval event contract.

The litellm path emits ``approval_required`` with
``data = {"tools": [...], "count": N}``; CLI yolo mode and headless
auto-approve both read ``event.data["tools"]``. The claude-code backend
must use the same payload shape, otherwise approvals never dispatch and
``can_use_tool`` hangs on its future. See agent_loop.py:1019 for the
litellm contract and agent/main.py:381,1231 for the consumers.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent.config import Config
from agent.core.claude_code_backend import _make_can_use_tool
from agent.core.session import Event


class _FakeSession:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.events: list[Event] = []
        self.pending_approval: dict | None = None
        self._cc_approvals: dict[str, asyncio.Future[bool]] = {}

    async def send_event(self, event: Event) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_approval_event_uses_tools_key_matching_litellm_contract():
    session = _FakeSession(Config(model_name="claude-code/sonnet", yolo_mode=False))
    can_use_tool = _make_can_use_tool(session)

    tool_input = {"name": "demo", "flavor": "cpu-basic"}

    async def resolve_when_event_emitted() -> None:
        # Wait for the backend to emit approval_required and register the future.
        for _ in range(200):
            if session.events and session._cc_approvals:
                break
            await asyncio.sleep(0.005)
        approval_id = next(iter(session._cc_approvals))
        session._cc_approvals[approval_id].set_result(True)

    resolver = asyncio.create_task(resolve_when_event_emitted())
    result = await can_use_tool(
        "mcp__ml_intern__sandbox_create", tool_input, ctx=SimpleNamespace()
    )
    await resolver

    # Future resolved True → backend returns an Allow result.
    assert getattr(result, "behavior", None) == "allow"

    # Exactly one approval_required event with the litellm-shaped payload.
    approval_events = [e for e in session.events if e.event_type == "approval_required"]
    assert len(approval_events) == 1
    data = approval_events[0].data
    assert "tools" in data, "must use 'tools' key, not 'tool_calls'"
    assert "tool_calls" not in data
    assert data["count"] == 1

    (tool_entry,) = data["tools"]
    assert tool_entry["tool"] == "sandbox_create"
    assert tool_entry["arguments"] == tool_input
    assert tool_entry["tool_call_id"].startswith("cc-")


@pytest.mark.asyncio
async def test_no_approval_event_when_tool_does_not_need_approval():
    session = _FakeSession(Config(model_name="claude-code/sonnet", yolo_mode=False))
    can_use_tool = _make_can_use_tool(session)

    # hf_docs is a read-only tool — _needs_approval returns False.
    result = await can_use_tool(
        "mcp__ml_intern__hf_docs", {"query": "anything"}, ctx=SimpleNamespace()
    )

    assert getattr(result, "behavior", None) == "allow"
    assert not [e for e in session.events if e.event_type == "approval_required"]
    assert session._cc_approvals == {}
