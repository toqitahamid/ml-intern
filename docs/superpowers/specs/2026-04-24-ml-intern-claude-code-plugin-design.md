# ml-intern Claude Code Plugin — Design

**Date:** 2026-04-24
**Status:** Proposed

## Goal

Package ml-intern's ML-research capabilities as a Claude Code plugin so that
Claude Code sessions gain parity with the standalone `ml-intern` agent for
Hugging Face ecosystem work, without duplicating Claude Code's native
infrastructure.

## Non-goals

- Reimplementing ml-intern's agent loop, context compaction, doom-loop detector,
  or LLM routing inside Claude Code. Claude Code owns these.
- Porting tools that duplicate Claude Code natives (`local_tools`, `edit_utils`,
  `plan_tool`).
- Wrapping the FastAPI backend or session persistence layer.

## Approach

**Option A (tools-only plugin), chosen** over B (CLI wrapper) and C (full port).
A is the only approach that aligns with the plugin extension model: Claude Code
remains the agent and gains ml-intern's specialized capabilities.

## Plugin layout

```
ml-intern-plugin/
├── .claude-plugin/plugin.json
├── skills/ml-intern/SKILL.md
├── commands/ml-intern.md
├── mcp/ml_intern_tools.py
├── settings.json
└── .mcp.json
```

## Components

### 1. MCP server — `mcp/ml_intern_tools.py`

A stdio MCP server that imports ml-intern's existing tool modules and exposes
them as MCP tools. No reimplementation; each MCP tool is a thin adapter calling
the corresponding `agent/tools/*` function.

Tools exposed (all prefixed `ml_` to avoid collision with Claude Code natives):

| Tool | Source module |
| --- | --- |
| `ml_hf_docs_search`, `ml_hf_docs_fetch` | `docs_tools.py` |
| `ml_hf_dataset_search`, `ml_hf_dataset_info`, `ml_hf_dataset_sample` | `dataset_tools.py` |
| `ml_hf_repo_files`, `ml_hf_repo_git` | `hf_repo_files_tool.py`, `hf_repo_git_tool.py` |
| `ml_hf_papers_search` | `papers_tool.py` |
| `ml_hf_research` | `research_tool.py` |
| `ml_hf_private_repo_list`, `ml_hf_private_repo_read`, `ml_hf_private_repo_upload` | `private_hf_repo_tools.py` |
| `ml_hf_jobs_run`, `ml_hf_jobs_status`, `ml_hf_jobs_logs` | `jobs_tool.py` |
| `ml_sandbox_create`, `ml_sandbox_exec`, `ml_sandbox_close` | `sandbox_tool.py`, `sandbox_client.py` |
| `ml_github_find_examples`, `ml_github_list_repos`, `ml_github_read_file` | `github_*.py` |

Environment required: `HF_TOKEN`, `GITHUB_TOKEN` (same as ml-intern).

### 2. Skill — `skills/ml-intern/SKILL.md`

Auto-triggers on ML research tasks. Description includes trigger phrases:
"train a model", "fine-tune", "HF dataset", "run a job on HF", "find a paper on",
"ML research", "Hugging Face", "inference endpoint", "sandbox run".

Body contains:
- Condensed persona from `agent/prompts/system_prompt_v3.yaml`: research-first,
  validate-then-implement, autonomous, zero-error target.
- **Division of labor rule** — the core of the skill:
  - Use `ml_hf_*` and `ml_github_*` for HF/ML-specific operations.
  - Use Claude Code's native Read/Edit/Write/Bash for local files and shell.
  - Use `TaskCreate` for plan tracking, never a ported plan tool.
- **Approval etiquette** — call `ml_sandbox_create`, `ml_hf_jobs_run`, and
  `ml_hf_private_repo_upload` only when necessary; expect a permission prompt.

### 3. Slash command — `commands/ml-intern.md`

`/ml-intern <prompt>` forces skill activation for the current turn by prepending
the skill's persona block to the user message. Used when auto-triggering would
miss (ambiguous prompts, cross-domain tasks).

### 4. Permission config — `settings.json`

```json
{
  "permissions": {
    "ask": [
      "mcp__ml_intern_tools__ml_sandbox_create",
      "mcp__ml_intern_tools__ml_hf_jobs_run",
      "mcp__ml_intern_tools__ml_hf_private_repo_upload"
    ]
  }
}
```

Maps ml-intern's `_needs_approval` set onto Claude Code's native permission
prompts. Other ported tools auto-run.

### 5. MCP config — `.mcp.json`

```json
{
  "mcpServers": {
    "ml_intern_tools": {
      "type": "stdio",
      "command": "python",
      "args": ["mcp/ml_intern_tools.py"],
      "env": {
        "HF_TOKEN": "${HF_TOKEN}",
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "hf-mcp-server": {
      "transport": "http",
      "url": "https://huggingface.co/mcp?login"
    }
  }
}
```

Second entry copied verbatim from `configs/main_agent_config.json`.

## Data flow

1. User submits an ML task in Claude Code.
2. Skill auto-activates (description match) or user invokes `/ml-intern`.
3. Claude Code plans the work and calls tools:
   - HF/GitHub operations → `ml_*` tools via MCP.
   - Local file/shell operations → native Read/Edit/Write/Bash.
   - Progress tracking → TaskCreate.
4. Expensive or destructive tools (sandbox/jobs/upload) trigger Claude Code's
   native permission prompt.
5. Results stream into the conversation; Claude Code's loop continues.

## What is explicitly not ported

| Dropped | Reason |
| --- | --- |
| `agent/core/agent_loop.py` | Claude Code has its own loop |
| `agent/context_manager/` | Claude Code has its own compaction |
| Doom-loop detector | Claude Code has stop hooks |
| `local_tools`, `edit_utils` | Superseded by Read/Edit/Write |
| `plan_tool` | Superseded by TaskCreate |
| litellm / claude-code-backend | Claude Code handles routing |
| `backend/` FastAPI | Out of scope for a CC plugin |

## Testing

1. **Smoke test** — `/ml-intern find a small image-classification dataset on HF
   and show me its schema`. Expected: `ml_hf_dataset_search` + `ml_hf_dataset_info`
   calls; no errors; persona-shaped response.
2. **Permission gate test** — `/ml-intern run a CPU job that trains MNIST for 1
   epoch`. Expected: Claude Code permission prompt before `ml_hf_jobs_run`
   executes.
3. **Division-of-labor test** — `/ml-intern read pyproject.toml and summarize
   dependencies`. Expected: native Read tool, not a ported file-read tool.
4. **MCP parity test** — confirm `hf-mcp-server` tools appear alongside
   `ml_intern_tools` in the available-tool list.

## Open risks

- **Tool import coupling** — `mcp/ml_intern_tools.py` imports from the
  ml-intern Python package, so the plugin requires ml-intern's runtime deps
  installed (via `uv sync` in the ml-intern repo, or pip-installable wheel).
  Mitigation: ship the plugin as a subdirectory inside the ml-intern repo, or
  publish ml-intern's tool subset as its own PyPI package.
- **Tool signatures** — ml-intern tools return Python objects; MCP wants JSON.
  Adapter layer must serialize consistently. Mitigation: use existing
  `ToolSpec` metadata from `agent/core/tools.py` as the source of truth for
  schemas.
- **HF MCP auth** — `huggingface.co/mcp?login` uses OAuth; first-run UX in CC
  may differ from ml-intern's. Acceptable for v1.
