# ML Intern — Project Guide for Claude Code

An autonomous ML agent built on the Hugging Face ecosystem. Entry points: a CLI
(`ml-intern`) and a FastAPI web backend.

## Repo layout

- `agent/` — Python agent package (CLI + agentic loop)
  - `main.py` — CLI entry (`ml-intern` script, see `[project.scripts]` in `pyproject.toml`)
  - `core/` — agent loop, handlers, tool router, session, doom-loop detector
  - `context_manager/` — message history + auto-compaction (170k token budget)
  - `tools/` — built-in tools (HF docs/repos/datasets/jobs/papers, GitHub search, sandbox, planning)
  - `prompts/`, `utils/`, `config.py`
- `backend/` — FastAPI server (`main.py`, `routes/`, `session_manager.py`, `start.sh`)
- `frontend/` — web UI assets
- `configs/main_agent_config.json` — model + MCP server config (supports `${ENV_VAR}` substitution)
- `Dockerfile` — HF Space deploy target

## Setup

```bash
uv sync
uv tool install -e .        # installs `ml-intern` CLI globally
```

Required env (`.env` at repo root):

- `ANTHROPIC_API_KEY` — for Anthropic models
- `HF_TOKEN` — Hugging Face (CLI prompts if missing)
- `GITHUB_TOKEN` — GitHub code search

## Common commands

- Run agent CLI: `ml-intern` (interactive) or `ml-intern "prompt"` (headless)
- Run backend: `bash backend/start.sh`
- Tests: `uv run pytest` (dev extra: `uv sync --extra dev`)
- Eval suite: `uv sync --extra eval` (uses `inspect-ai`)

## Architecture notes

- Agent loop lives in `agent/core/` — `submission_loop` consumes `Operation`s and
  emits events on `event_queue`. Max 300 iterations per run.
- LLM calls go through `litellm.acompletion` — model-agnostic (Anthropic, HF
  Inference, etc.). Default model set in `configs/main_agent_config.json`.
- `ContextManager` auto-compacts near the 170k token ceiling.
- Tool execution is gated: jobs, sandbox, and destructive ops require approval
  (see `approval_required` event).
- MCP servers are configured in `configs/main_agent_config.json` under
  `mcpServers`; `${VAR}` tokens are substituted from `.env` at load time.

## Events emitted on `event_queue`

`processing`, `ready`, `assistant_chunk`, `assistant_message`,
`assistant_stream_end`, `tool_call`, `tool_output`, `tool_log`,
`tool_state_change`, `approval_required`, `turn_complete`, `error`,
`interrupted`, `compacted`, `undo_complete`, `shutdown`.

## Adding things

- **Built-in tool**: append a `ToolSpec` in `agent/core/tools.py::create_builtin_tools`.
- **MCP server**: add an entry under `mcpServers` in `configs/main_agent_config.json`.

## Backends: `litellm` vs `claude-code`

Two LLM backends, selectable at runtime:

- **`litellm` (default)** — routes via `litellm.acompletion`. Uses `ANTHROPIC_API_KEY`, `HF_TOKEN`, etc. Classic path in `agent/core/agent_loop.py`.
- **`claude-code`** — routes via `claude-agent-sdk` so usage bills against your Claude Max subscription instead of the API. Implemented in `agent/core/claude_code_backend.py`. The SDK drives the agentic loop; ml-intern's tools are exposed to it as an in-process MCP server and Claude Code's built-in tools (Bash/Read/Edit…) are disabled.

Selecting the backend:

```bash
# auto-detect from model name prefix
ml-intern --model claude-code/sonnet "your prompt"

# or explicitly
ml-intern --backend claude-code --model claude-code/opus "your prompt"
```

Interactive mode:

- `/backend` (no arg) — print current backend and available options
- `/backend claude-code` or `/backend litellm` — switch mid-session
- `/model claude-code/<name>` — also auto-switches backend to `claude-code`
- SUGGESTED_MODELS (shown by `/model`) includes claude-code entries

Gotchas:

- `unset ANTHROPIC_API_KEY` before running, or the SDK may prefer the API and bill it instead of the subscription. The backend logs a warning if it's set.
- The `claude` CLI must be installed and logged in (`claude login`).
- **Approval flow** is wired: ml-intern's `_needs_approval` check runs inside the SDK's `can_use_tool` callback. Tools that need approval (sandbox_create, CPU hf_jobs, file uploads, etc.) emit `approval_required` events just like the litellm path and block until `exec_approval` resolves the future. Abandonment on user-continue also resolves pending futures to False.
- **Doom-loop** is a per-turn in-loop guard: if the SDK calls the same tool with identical args ≥ `_DOOM_LOOP_THRESHOLD` (4) times in one turn, the backend calls `client.interrupt()`. This is simpler than the litellm detector (which spans turns via ContextManager) but catches the common spam case.
- `ContextManager` auto-compaction is bypassed — the SDK owns its own message history and compaction.

Dispatch point: top of `Handlers.run_agent` in `agent/core/agent_loop.py` branches on `session.config.backend`. Approval resolution: `Handlers.exec_approval` checks `session._cc_approvals` futures before falling through to the litellm path.

## Conventions

- Python ≥3.11, managed by `uv` (see `uv.lock`).
- Package name in `pyproject.toml` is `hf-agent`; installed CLI is `ml-intern`.
- Async throughout the agent core — prefer `async def` handlers.
- Don't bypass the approval flow for sandbox/job tools.
- Don't commit `.env` or tokens.
