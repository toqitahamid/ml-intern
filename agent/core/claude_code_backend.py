"""
Claude Code backend for ml-intern.

Routes LLM calls through the Claude Agent SDK so usage bills against the
user's Claude Max subscription instead of the Anthropic API.

Design:
  - Claude Agent SDK runs the agentic loop (LLM call → tool → LLM call → ...).
  - ml-intern's ToolRouter tools are exposed to the SDK as an in-process MCP
    server (create_sdk_mcp_server + @tool).
  - Claude Code's built-in tools (Bash/Read/Edit/…) are disabled so the model
    can only use ml-intern's ML-focused tools.
  - SDK messages are translated into ml-intern's Event stream so the existing
    CLI/backend renderers keep working.

Trade-offs:
  - No approval flow yet (yolo only). The SDK can do it via `can_use_tool`;
    wire that in a follow-up.
  - No doom-loop detector; the SDK has its own termination heuristics.
  - ml-intern's ContextManager still stores the conversation for save/upload,
    but the SDK owns the live message history sent to the model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    tool,
)
from litellm import Message

from agent.core.session import Event, Session

logger = logging.getLogger(__name__)

MODEL_PREFIX = "claude-code/"

# If the SDK calls the same (tool_name, args_hash) this many times in one
# user turn, we interrupt — the same guardrail idea as the litellm backend's
# doom-loop detector, but scoped to a single .query() invocation.
_DOOM_LOOP_THRESHOLD = 4


def strip_prefix(model_name: str) -> str:
    """`claude-code/sonnet` → `sonnet`.  Plain names pass through."""
    if model_name.startswith(MODEL_PREFIX):
        return model_name[len(MODEL_PREFIX):] or "sonnet"
    return model_name


def _tool_prefix(tool_name: str) -> str:
    """Claude Code exposes MCP tools as `mcp__<server>__<tool>`."""
    return f"mcp__ml_intern__{tool_name}"


def _build_mcp_server(session: Session):
    """Wrap ml-intern's ToolRouter tools as Claude Agent SDK @tool handlers."""

    wrappers = []
    for spec in session.tool_router.tools.values():
        name = spec.name
        description = spec.description
        schema = spec.parameters or {"type": "object", "properties": {}}

        def make_handler(tool_name: str):
            async def handler(args: dict[str, Any]) -> dict[str, Any]:
                try:
                    output, ok = await session.tool_router.call_tool(
                        tool_name, args, session=session
                    )
                except Exception as e:
                    logger.exception("tool %s raised", tool_name)
                    return {
                        "content": [{"type": "text", "text": f"tool error: {e}"}],
                        "is_error": True,
                    }
                return {
                    "content": [{"type": "text", "text": output}],
                    "is_error": not ok,
                }
            return handler

        wrappers.append(
            tool(name, description, schema)(make_handler(name))
        )

    return create_sdk_mcp_server(
        name="ml_intern", version="0.1.0", tools=wrappers
    )


def _make_can_use_tool(session: Session):
    """Build a `can_use_tool` callback that defers to ml-intern's approval logic."""
    from agent.core.agent_loop import _needs_approval  # local import avoids cycle

    async def can_use_tool(
        tool_name: str, tool_input: dict[str, Any], ctx: ToolPermissionContext
    ):
        real_name = tool_name.removeprefix("mcp__ml_intern__")
        if not _needs_approval(real_name, tool_input, session.config):
            return PermissionResultAllow(behavior="allow", updated_input=tool_input)

        # Register a future so Handlers.exec_approval can resolve this.
        approval_id = f"cc-{uuid.uuid4().hex[:12]}"
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        approvals: dict[str, asyncio.Future[bool]] = getattr(
            session, "_cc_approvals", None
        ) or {}
        approvals[approval_id] = fut
        session._cc_approvals = approvals

        # Surface the request to the UI exactly like the litellm backend does.
        await session.send_event(Event(
            event_type="approval_required",
            data={
                "tool_calls": [{
                    "tool_call_id": approval_id,
                    "tool": real_name,
                    "arguments": tool_input,
                }],
            },
        ))
        # Mirror litellm backend's pending_approval slot so UIs that key off
        # `session.pending_approval` still light up.
        session.pending_approval = {
            "tool_calls": [],
            "claude_code_ids": list(approvals.keys()),
        }

        try:
            approved = await fut
        except asyncio.CancelledError:
            return PermissionResultDeny(
                behavior="deny", message="Interrupted before approval.",
            )
        finally:
            approvals.pop(approval_id, None)
            if not approvals:
                session.pending_approval = None

        if approved:
            return PermissionResultAllow(behavior="allow", updated_input=tool_input)
        return PermissionResultDeny(
            behavior="deny", message="User rejected tool execution.",
        )

    return can_use_tool


def _build_options(session: Session) -> ClaudeAgentOptions:
    mcp_server = _build_mcp_server(session)

    allowed = [_tool_prefix(name) for name in session.tool_router.tools.keys()]

    return ClaudeAgentOptions(
        model=strip_prefix(session.config.model_name),
        system_prompt=session.context_manager.system_prompt,
        mcp_servers={"ml_intern": mcp_server},
        tools=[],                     # disable all Claude Code built-in tools
        allowed_tools=allowed,        # whitelist our MCP tools only
        # Route every tool through our can_use_tool for ml-intern's approval rules.
        permission_mode="default",
        can_use_tool=_make_can_use_tool(session),
        setting_sources=[],           # ignore user's ~/.claude config
        include_partial_messages=True,
        max_turns=session.config.max_iterations
        if session.config.max_iterations > 0 else None,
        continue_conversation=False,  # we drive multi-turn via .query()
    )


async def _get_or_create_client(session: Session) -> ClaudeSDKClient:
    client = getattr(session, "_claude_code_client", None)
    if client is not None:
        return client
    client = ClaudeSDKClient(options=_build_options(session))
    await client.connect()
    session._claude_code_client = client
    return client


async def shutdown_client(session: Session) -> None:
    client = getattr(session, "_claude_code_client", None)
    if client is None:
        return
    try:
        await client.disconnect()
    except Exception:
        logger.warning("claude-code client disconnect failed", exc_info=True)
    session._claude_code_client = None


def _check_api_key_warning() -> str | None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return (
            "ANTHROPIC_API_KEY is set — the Claude Agent SDK may bill the API "
            "instead of your subscription. Unset it to force subscription mode."
        )
    return None


async def run_agent_claude_code(session: Session, text: str) -> str | None:
    """Drop-in replacement for Handlers.run_agent when backend=claude-code."""
    session.reset_cancel()

    warning = _check_api_key_warning()
    if warning:
        await session.send_event(Event(
            event_type="tool_log", data={"tool": "system", "log": warning}
        ))

    if text:
        session.context_manager.add_message(Message(role="user", content=text))

    await session.send_event(
        Event(event_type="processing", data={"message": "Processing user input"})
    )

    client = await _get_or_create_client(session)
    await client.query(text or "")

    final_text_parts: list[str] = []
    pending_tool_calls: dict[str, dict[str, Any]] = {}
    # Per-turn doom-loop guard.
    import hashlib
    sig_counts: dict[tuple[str, str], int] = {}
    doom_triggered = False

    async for msg in client.receive_response():
        if session.is_cancelled:
            try:
                await client.interrupt()
            except Exception:
                pass
            break

        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text:
                    final_text_parts.append(block.text)
                    if session.stream:
                        await session.send_event(Event(
                            event_type="assistant_chunk",
                            data={"content": block.text},
                        ))
                    else:
                        await session.send_event(Event(
                            event_type="assistant_message",
                            data={"content": block.text},
                        ))
                elif isinstance(block, ThinkingBlock):
                    # Surface thinking as a log so the UI can show it if desired.
                    await session.send_event(Event(
                        event_type="tool_log",
                        data={"tool": "thinking", "log": block.thinking or ""},
                    ))
                elif isinstance(block, ToolUseBlock):
                    pending_tool_calls[block.id] = {
                        "name": block.name,
                        "args": block.input,
                    }
                    display_name = block.name.removeprefix("mcp__ml_intern__")
                    await session.send_event(Event(
                        event_type="tool_call",
                        data={
                            "tool_call_id": block.id,
                            "tool": display_name,
                            "arguments": block.input,
                        },
                    ))
                    # Doom-loop: same (tool, args) too many times → interrupt.
                    args_hash = hashlib.md5(
                        json.dumps(block.input, sort_keys=True, default=str).encode()
                    ).hexdigest()[:12]
                    key = (block.name, args_hash)
                    sig_counts[key] = sig_counts.get(key, 0) + 1
                    if sig_counts[key] >= _DOOM_LOOP_THRESHOLD and not doom_triggered:
                        doom_triggered = True
                        await session.send_event(Event(
                            event_type="tool_log",
                            data={
                                "tool": "system",
                                "log": (
                                    f"Doom loop: {display_name} called "
                                    f"{sig_counts[key]}× with identical args — "
                                    "interrupting."
                                ),
                            },
                        ))
                        try:
                            await client.interrupt()
                        except Exception:
                            logger.warning("interrupt failed", exc_info=True)

        elif isinstance(msg, UserMessage):
            # User messages in this stream carry tool_result blocks the SDK
            # produced by executing our MCP tools.
            for block in msg.content if isinstance(msg.content, list) else []:
                if isinstance(block, ToolResultBlock):
                    entry = pending_tool_calls.pop(block.tool_use_id, None)
                    display_name = (entry or {}).get("name", "tool").removeprefix("mcp__ml_intern__")
                    content = block.content
                    if isinstance(content, list):
                        output = "\n".join(
                            c.get("text", "") if isinstance(c, dict) else str(c)
                            for c in content
                        )
                    else:
                        output = str(content or "")
                    await session.send_event(Event(
                        event_type="tool_output",
                        data={
                            "tool_call_id": block.tool_use_id,
                            "tool": display_name,
                            "output": output,
                            "is_error": bool(block.is_error),
                        },
                    ))

        elif isinstance(msg, SystemMessage):
            logger.debug("claude-code system msg: %s", msg)

        elif isinstance(msg, ResultMessage):
            # End of this turn.
            break

    final_text = "".join(final_text_parts) or None
    if final_text:
        session.context_manager.add_message(
            Message(role="assistant", content=final_text)
        )

    await session.send_event(Event(
        event_type="turn_complete",
        data={"final_response": final_text},
    ))

    session.increment_turn()
    await session.auto_save_if_needed()

    return final_text
