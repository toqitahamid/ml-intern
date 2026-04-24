# ml-intern Claude Code Plugin — Design

**Date:** 2026-04-24
**Status:** Proposed

## Goal

A self-contained Claude Code plugin that, once installed, turns Claude Code
into an autonomous ML-research agent matching the behavior of the standalone
`ml-intern` agent. The user installs one plugin and does not configure or
install anything else.

## Non-goals

- Reimplementing Claude Code infrastructure: agent loop, context compaction,
  doom-loop detector, LLM routing, subagent dispatch.
- Delegating to any other plugin (including the official `huggingface-skills`).
  The plugin is fully self-contained.
- Supporting remote sandboxes or Hugging Face Jobs. The target user runs ML
  compute on their own HPC (SLURM/PBS) via Bash/SSH.

## Approach

Port ml-intern's HF and GitHub tools verbatim into a Claude Code plugin as an
in-process stdio MCP server. Ship `system_prompt_v3.yaml` verbatim as the
plugin's skill, with only the minimal edits needed for Claude Code and the
HPC workflow. Drop infrastructure that Claude Code already provides (loop,
subagent dispatch, file ops, task tracking).

Why this over delegating to `huggingface-skills`: tool names in the system
prompt match the ported tools 1:1, so no translation layer is needed; the
plugin has no external plugin dependency; the behavior is a faithful mirror
of ml-intern rather than a lookalike built from a similar-but-different tool
surface.

## Plugin layout

```
ml-intern-plugin/
├── .claude-plugin/
│   └── plugin.json
├── skills/
│   └── ml-intern/
│       └── SKILL.md              # system_prompt_v3 verbatim with minimal edits
├── commands/
│   └── ml-intern.md              # /ml-intern force-activates the skill
├── mcp/
│   ├── ml_intern_tools.py        # MCP server entrypoint
│   └── tools/                    # vendored copies of ml-intern tool modules
│       ├── docs_tools.py
│       ├── dataset_tools.py
│       ├── hf_repo_files_tool.py
│       ├── hf_repo_git_tool.py
│       ├── papers_tool.py
│       ├── private_hf_repo_tools.py
│       ├── github_find_examples.py
│       ├── github_list_repos.py
│       ├── github_read_file.py
│       └── utilities.py
├── settings.json
└── .mcp.json                     # declares ml_intern_tools + hf-mcp-server
```

## Components

### 1. MCP server — `mcp/ml_intern_tools.py` + `mcp/tools/*`

A stdio MCP server that imports the vendored tool modules and exposes each
public tool function as an MCP tool with the same name as in ml-intern
(`explore_hf_docs`, `fetch_hf_docs`, `find_hf_api`, `hf_inspect_dataset`,
`hf_repo_files`, `hub_repo_details`, `hf_papers`, private repo tools,
`github_find_examples`, `github_list_repos`, `github_read_file`).

Tool names match the system prompt exactly, so no translation between prompt
and runtime is needed.

**Vendoring strategy:** the tool modules are copied into `mcp/tools/` inside
the plugin — no runtime import from the ml-intern package. Imports inside
those files are rewritten from `agent.tools.utilities` to `mcp.tools.utilities`
and from `agent.core.*` to plugin-local equivalents (or inlined). This isolates
the plugin from ml-intern's package layout changes.

**Environment required:** `HF_TOKEN`, `GITHUB_TOKEN`. Loaded from the user's
shell or a `.env` in the plugin directory.

### 2. Skill — `skills/ml-intern/SKILL.md`

Contains `agent/prompts/system_prompt_v3.yaml` verbatim with these minimal
edits only:

| Original line | Edit |
| --- | --- |
| `{{ num_tools }}` in opening line | Replaced with the actual tool count of the ported set |
| `plan_tool` reference | Replaced with `TaskCreate` (one-line mention) |
| `hf_jobs` section | Replaced with HPC section: "submit via your HPC scheduler (SLURM/PBS) with Bash; export `HF_TOKEN` in the submit script" |
| `sandbox_create` section | Dropped (user runs on HPC, not a remote sandbox) |
| "HF_TOKEN auto in job secrets" line | Rewritten for HPC export pattern |
| `research({...})` JSON call pattern | Rewritten to call Claude Code's `Agent` tool with `subagent_type="general-purpose"` and the same task/context structure |

All other content — research-first workflow, hallucinated-imports warnings,
dataset-format-by-method table, hardware sizing table, OOM recovery rules,
pre-flight checklist, autonomous-mode loop, communication style, tool usage
rules — is preserved word-for-word.

The skill description triggers on: "train a model", "fine-tune", "HF dataset",
"Hugging Face", "find a paper on", "ML research", "inference", "SFT", "DPO",
"GRPO", "LoRA".

### 3. Slash command — `commands/ml-intern.md`

`/ml-intern <prompt>` force-activates the skill for the current turn. Handles
cases where the skill description would not auto-fire (ambiguous or
cross-domain prompts).

### 4. MCP config — `.mcp.json`

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

The `hf-mcp-server` entry is copied from ml-intern's
`configs/main_agent_config.json`. First-run OAuth handles HF MCP auth.

### 5. Plugin manifest — `.claude-plugin/plugin.json`

Standard manifest: name, version, description, author. No plugin dependencies.

### 6. Permission config — `settings.json`

None of the ported tools are destructive (sandbox/jobs/upload were dropped).
File retained as a minimal `{"permissions": {}}` for future additions.

## Data flow

1. User submits an ML task in Claude Code.
2. Skill auto-activates (description match) or user runs `/ml-intern`.
3. Claude Code executes the ported ml-intern workflow:
   - **Research** → `Agent(subagent_type="general-purpose", ...)` subagent call
     using `hf_papers`, `explore_hf_docs`, `fetch_hf_docs`, `github_find_examples`,
     `github_read_file`, `hf_inspect_dataset`.
   - **Data validation** → `hf_inspect_dataset`, `hub_repo_details`.
   - **Implementation** → native Read/Edit/Write to produce training scripts.
   - **Launch** → Bash to submit to the user's HPC scheduler (SLURM/PBS).
   - **Progress tracking** → TaskCreate.
4. Results stream into the conversation; Claude Code's loop continues per the
   autonomous-mode rules preserved in the skill.

## What is explicitly not ported

| Dropped | Reason |
| --- | --- |
| `agent/core/agent_loop.py`, `context_manager/`, doom-loop detector | Claude Code provides these |
| `local_tools.py`, `edit_utils.py` | Superseded by Read/Edit/Write |
| `plan_tool.py` | Superseded by TaskCreate |
| `research_tool.py` | Superseded by Claude Code's `Agent` dispatch |
| `jobs_tool.py`, `sandbox_tool.py`, `sandbox_client.py` | User runs ML compute on HPC |
| litellm, `claude_code_backend.py`, `backend/` FastAPI | Out of scope for a Claude Code plugin |

## Testing

1. **Smoke test** — `/ml-intern find a small image-classification dataset on HF
   and show me its schema`. Expected: `hf_inspect_dataset` called; schema
   returned.
2. **Research workflow test** — `/ml-intern plan a SFT recipe for small LLMs on
   a reasoning benchmark`. Expected: `Agent` subagent call; `hf_papers` with
   citation_graph + read_paper; pre-flight checklist emitted.
3. **GitHub tool test** — `/ml-intern find TRL SFTTrainer examples on GitHub
   that use gradient_checkpointing`. Expected: `github_find_examples` call with
   correctly-shaped query.
4. **HPC path test** — `/ml-intern write a SLURM submit script for a 7B SFT
   run`. Expected: no references to `hf_jobs` or sandbox; Bash writes an
   `sbatch` script.
5. **Auto-trigger test** — plain "train a 7B Llama with DPO on UltraFeedback"
   (no slash command) activates the skill.
6. **HF MCP test** — confirm HF MCP server tools appear and work after
   first-run OAuth.
7. **Tool-name parity test** — verify every tool name referenced in the skill
   body matches an MCP tool exposed by `ml_intern_tools` or `hf-mcp-server`.

## Open risks

- **Vendored import rewrites** — `papers_tool.py`, `docs_tools.py`, etc. may
  import from `agent.core.llm_params`, `agent.core.session`, or other internal
  modules. Those imports must be audited and either stubbed, inlined, or
  removed during the port. Mitigation: implementation plan includes an
  explicit "dependency audit" step per tool module before copying.
- **HF MCP OAuth first-run** — new users hit an OAuth redirect the first time
  they use an HF MCP tool. Acceptable for v1.
- **Skill auto-trigger precision** — too broad triggers annoy non-ML work; too
  narrow misses real ML tasks. Trigger list above is a starting point,
  refinable after dogfooding.
- **Python version and dependencies** — the vendored tools will need
  `huggingface_hub`, `requests`, and potentially others. The plugin's MCP
  server runs in a Python environment; the plugin manifest or a bundled
  `requirements.txt` must make this explicit.
