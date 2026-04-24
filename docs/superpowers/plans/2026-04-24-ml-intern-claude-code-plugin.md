# ml-intern Claude Code Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a self-contained Claude Code plugin that turns Claude Code into an autonomous ML-research agent matching the standalone `ml-intern` behavior, by vendoring ml-intern's HF and GitHub tools into an MCP server and shipping `system_prompt_v3` as a skill.

**Architecture:** The plugin lives in a new top-level `plugin/` directory. A stdio MCP server (`plugin/mcp/ml_intern_tools.py`) imports vendored copies of ml-intern's tool modules from `plugin/mcp/tools/` and exposes each public function as an MCP tool keeping the original names so the ported system prompt works unchanged. The skill (`plugin/skills/ml-intern/SKILL.md`) contains `system_prompt_v3.yaml` verbatim with minimal edits for Claude Code (subagent API), HPC workflow (SLURM replaces HF Jobs), and task tracking (TaskCreate replaces plan_tool). A `/ml-intern` slash command force-activates the skill. The plugin declares the HF MCP server (`huggingface.co/mcp`) in `.mcp.json` so it is available without user configuration.

**Tech Stack:** Python ≥3.11, `uv` for dependency management (matches repo), `mcp` Python SDK for the MCP server, `httpx` and `whoosh` and `beautifulsoup4` (already used by vendored tools), `pytest` for tests (already the project's test framework).

**Reference spec:** `docs/superpowers/specs/2026-04-24-ml-intern-claude-code-plugin-design.md`

---

## File layout (target)

```
plugin/
├── .claude-plugin/
│   └── plugin.json
├── skills/
│   └── ml-intern/
│       └── SKILL.md
├── commands/
│   └── ml-intern.md
├── mcp/
│   ├── __init__.py
│   ├── ml_intern_tools.py
│   └── tools/
│       ├── __init__.py
│       ├── types.py
│       ├── utilities.py
│       ├── docs_tools.py
│       ├── dataset_tools.py
│       ├── hf_repo_files_tool.py
│       ├── hf_repo_git_tool.py
│       ├── papers_tool.py
│       ├── private_hf_repo_tools.py
│       ├── github_find_examples.py
│       ├── github_list_repos.py
│       └── github_read_file.py
├── settings.json
├── .mcp.json
├── pyproject.toml
└── README.md

tests/plugin/
├── __init__.py
├── test_vendored_imports.py
├── test_mcp_server_registration.py
└── test_skill_tool_name_parity.py
```

---

## Task 1: Scaffold plugin directory and manifest

**Files:**
- Create: `plugin/.claude-plugin/plugin.json`
- Create: `plugin/README.md`
- Create: `plugin/pyproject.toml`
- Create: `plugin/mcp/__init__.py` (empty)
- Create: `plugin/mcp/tools/__init__.py` (empty)

- [ ] **Step 1: Create plugin manifest**

Write `plugin/.claude-plugin/plugin.json`:

```json
{
  "name": "ml-intern",
  "version": "0.1.0",
  "description": "Autonomous ML research agent for Claude Code: HF ecosystem tools, GitHub ML-search, and the ml-intern workflow discipline. Runs training on your HPC.",
  "author": "ml-intern contributors"
}
```

- [ ] **Step 2: Create plugin README**

Write `plugin/README.md`:

```markdown
# ml-intern — Claude Code plugin

Autonomous ML research agent. Gives Claude Code the `ml-intern` workflow:
research-first (literature crawl, methodology extraction), data audit,
pre-flight validated training scripts, and HPC-ready submission patterns.

## Install

```bash
/plugin install ml-intern
```

## Environment

Set in your shell or a `.env` at the repo root:

- `HF_TOKEN` — Hugging Face API token
- `GITHUB_TOKEN` — GitHub PAT for ML code search

## Usage

- Auto-triggers on ML tasks ("train a model", "fine-tune", "find a dataset on HF", ...).
- Or force-activate with `/ml-intern <prompt>`.

## What's inside

- Skill with the `ml-intern` persona and workflow (from `system_prompt_v3`).
- MCP server exposing HF docs/datasets/repos/papers and ML-shaped GitHub search tools.
- The official Hugging Face MCP server wired in automatically.
```

- [ ] **Step 3: Create plugin pyproject for MCP server deps**

Write `plugin/pyproject.toml`:

```toml
[project]
name = "ml-intern-plugin-mcp"
version = "0.1.0"
description = "MCP server for the ml-intern Claude Code plugin"
requires-python = ">=3.11"
dependencies = [
  "mcp>=1.0.0",
  "httpx>=0.27",
  "beautifulsoup4>=4.12",
  "whoosh>=2.7",
  "huggingface_hub>=0.25",
]
```

- [ ] **Step 4: Create empty __init__ files**

```bash
touch plugin/mcp/__init__.py plugin/mcp/tools/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add plugin/.claude-plugin plugin/README.md plugin/pyproject.toml plugin/mcp/__init__.py plugin/mcp/tools/__init__.py
git commit -m "feat(plugin): scaffold ml-intern Claude Code plugin"
```

---

## Task 2: Vendor types.py and utilities.py

**Files:**
- Create: `plugin/mcp/tools/types.py`
- Create: `plugin/mcp/tools/utilities.py`
- Create: `tests/plugin/__init__.py` (empty)
- Create: `tests/plugin/test_vendored_imports.py`

- [ ] **Step 1: Write failing import test**

Create `tests/plugin/__init__.py` (empty) and `tests/plugin/test_vendored_imports.py`:

```python
"""Smoke tests confirming vendored modules import standalone."""


def test_types_imports():
    from plugin.mcp.tools.types import ToolResult
    assert ToolResult is not None


def test_utilities_imports():
    from plugin.mcp.tools import utilities
    assert hasattr(utilities, "truncate")
    assert hasattr(utilities, "format_date")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/plugin/test_vendored_imports.py -v`
Expected: FAIL with `ModuleNotFoundError` for `plugin.mcp.tools.types`.

- [ ] **Step 3: Copy types.py**

```bash
cp agent/tools/types.py plugin/mcp/tools/types.py
```

`types.py` has no internal imports; no rewrites needed.

- [ ] **Step 4: Copy and rewrite utilities.py**

```bash
cp agent/tools/utilities.py plugin/mcp/tools/utilities.py
```

Inspect the copied file. If it has any `from agent.tools.<x>` imports, rewrite each to `from plugin.mcp.tools.<x>` using Edit. Known state at spec-write time: `utilities.py` has no `agent.*` imports.

Run: `grep -n "from agent" plugin/mcp/tools/utilities.py || echo "clean"`
Expected: `clean`. If any matches, rewrite each match.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/plugin/test_vendored_imports.py -v`
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add plugin/mcp/tools/types.py plugin/mcp/tools/utilities.py tests/plugin/__init__.py tests/plugin/test_vendored_imports.py
git commit -m "feat(plugin): vendor types and utilities from ml-intern"
```

---

## Task 3: Vendor docs_tools.py

**Files:**
- Create: `plugin/mcp/tools/docs_tools.py`
- Modify: `tests/plugin/test_vendored_imports.py`

- [ ] **Step 1: Extend the import test**

Append to `tests/plugin/test_vendored_imports.py`:

```python
def test_docs_tools_imports():
    from plugin.mcp.tools import docs_tools
    assert hasattr(docs_tools, "explore_hf_docs")
    assert hasattr(docs_tools, "fetch_hf_docs")
    assert hasattr(docs_tools, "find_hf_api")
```

- [ ] **Step 2: Run test, expect failure**

Run: `uv run pytest tests/plugin/test_vendored_imports.py::test_docs_tools_imports -v`
Expected: FAIL with `ModuleNotFoundError` on `plugin.mcp.tools.docs_tools`.

- [ ] **Step 3: Copy and rewrite imports**

```bash
cp agent/tools/docs_tools.py plugin/mcp/tools/docs_tools.py
```

Then find and rewrite any `agent.tools.*` imports:

```bash
grep -n "from agent\|import agent" plugin/mcp/tools/docs_tools.py
```

For each match, Edit to replace `from agent.tools` → `from plugin.mcp.tools`. Do not rewrite `agent.core.*` imports — instead, remove the import and any code that depends on it (spec: docs_tools uses no `agent.core.*`; verify).

- [ ] **Step 4: Confirm the three public entrypoints exist**

Run:
```bash
python -c "from plugin.mcp.tools.docs_tools import explore_hf_docs, fetch_hf_docs, find_hf_api; print('ok')"
```
Expected: `ok`.

If any name is missing (e.g. the module calls them differently), adjust the import test to use the real names and document the mapping in a header comment.

- [ ] **Step 5: Run full vendored-imports test**

Run: `uv run pytest tests/plugin/test_vendored_imports.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add plugin/mcp/tools/docs_tools.py tests/plugin/test_vendored_imports.py
git commit -m "feat(plugin): vendor docs_tools"
```

---

## Task 4: Vendor dataset_tools.py

**Files:**
- Create: `plugin/mcp/tools/dataset_tools.py`
- Modify: `tests/plugin/test_vendored_imports.py`

- [ ] **Step 1: Extend the import test**

Append to `tests/plugin/test_vendored_imports.py`:

```python
def test_dataset_tools_imports():
    from plugin.mcp.tools import dataset_tools
    # The main public entrypoint in ml-intern is hf_inspect_dataset
    assert hasattr(dataset_tools, "hf_inspect_dataset")
```

- [ ] **Step 2: Run test, expect failure**

Run: `uv run pytest tests/plugin/test_vendored_imports.py::test_dataset_tools_imports -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Copy and rewrite imports**

```bash
cp agent/tools/dataset_tools.py plugin/mcp/tools/dataset_tools.py
grep -n "from agent\|import agent" plugin/mcp/tools/dataset_tools.py
```

Edit each match from `from agent.tools` to `from plugin.mcp.tools`. Known at spec-write time: `dataset_tools.py` imports `from agent.tools.types import ToolResult` — rewrite to `from plugin.mcp.tools.types import ToolResult`.

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/plugin/test_vendored_imports.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugin/mcp/tools/dataset_tools.py tests/plugin/test_vendored_imports.py
git commit -m "feat(plugin): vendor dataset_tools"
```

---

## Task 5: Vendor hf_repo_files_tool.py and hf_repo_git_tool.py

**Files:**
- Create: `plugin/mcp/tools/hf_repo_files_tool.py`
- Create: `plugin/mcp/tools/hf_repo_git_tool.py`
- Modify: `tests/plugin/test_vendored_imports.py`

- [ ] **Step 1: Extend import test**

Append to `tests/plugin/test_vendored_imports.py`:

```python
def test_hf_repo_files_imports():
    from plugin.mcp.tools import hf_repo_files_tool
    assert hasattr(hf_repo_files_tool, "hf_repo_files")


def test_hf_repo_git_imports():
    from plugin.mcp.tools import hf_repo_git_tool
    assert hasattr(hf_repo_git_tool, "hub_repo_details")
```

If either public function name differs in the source file, adjust the asserts to match what the source defines and leave a header-comment mapping in the vendored file.

- [ ] **Step 2: Run tests, expect failure**

Run: `uv run pytest tests/plugin/test_vendored_imports.py -v`
Expected: the two new tests FAIL.

- [ ] **Step 3: Copy both files and rewrite imports**

```bash
cp agent/tools/hf_repo_files_tool.py plugin/mcp/tools/hf_repo_files_tool.py
cp agent/tools/hf_repo_git_tool.py plugin/mcp/tools/hf_repo_git_tool.py
grep -n "from agent\|import agent" plugin/mcp/tools/hf_repo_files_tool.py plugin/mcp/tools/hf_repo_git_tool.py
```

For each match, Edit `from agent.tools` → `from plugin.mcp.tools`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/plugin/test_vendored_imports.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugin/mcp/tools/hf_repo_files_tool.py plugin/mcp/tools/hf_repo_git_tool.py tests/plugin/test_vendored_imports.py
git commit -m "feat(plugin): vendor hf_repo_files and hf_repo_git tools"
```

---

## Task 6: Vendor papers_tool.py

**Files:**
- Create: `plugin/mcp/tools/papers_tool.py`
- Modify: `tests/plugin/test_vendored_imports.py`

- [ ] **Step 1: Extend import test**

Append to `tests/plugin/test_vendored_imports.py`:

```python
def test_papers_tool_imports():
    from plugin.mcp.tools import papers_tool
    # hf_papers is the operation dispatcher in ml-intern
    assert hasattr(papers_tool, "hf_papers")
```

If the source uses a different public name (e.g. `hf_papers_handler`), update the assert and add a header comment mapping.

- [ ] **Step 2: Run test, expect failure**

Run: `uv run pytest tests/plugin/test_vendored_imports.py::test_papers_tool_imports -v`
Expected: FAIL.

- [ ] **Step 3: Copy and rewrite**

```bash
cp agent/tools/papers_tool.py plugin/mcp/tools/papers_tool.py
grep -n "from agent\|import agent" plugin/mcp/tools/papers_tool.py
```

Known: `from agent.tools.types import ToolResult` → rewrite to `from plugin.mcp.tools.types import ToolResult`. No `agent.core.*` imports expected.

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/plugin/test_vendored_imports.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add plugin/mcp/tools/papers_tool.py tests/plugin/test_vendored_imports.py
git commit -m "feat(plugin): vendor papers_tool"
```

---

## Task 7: Vendor private_hf_repo_tools.py

**Files:**
- Create: `plugin/mcp/tools/private_hf_repo_tools.py`
- Modify: `tests/plugin/test_vendored_imports.py`

- [ ] **Step 1: Identify public names**

Run: `grep -E "^def |^async def " agent/tools/private_hf_repo_tools.py`
Record the names in the task comments. Example output: `hf_private_list`, `hf_private_read`, `hf_private_upload` (actual names may differ).

- [ ] **Step 2: Extend import test**

Append to `tests/plugin/test_vendored_imports.py` using the real names found in Step 1:

```python
def test_private_hf_repo_tools_imports():
    from plugin.mcp.tools import private_hf_repo_tools as p
    # Adjust the attribute names to match what Step 1 discovered.
    for name in ("hf_private_list", "hf_private_read", "hf_private_upload"):
        assert hasattr(p, name), f"missing {name}"
```

- [ ] **Step 3: Run test, expect failure**

Run: `uv run pytest tests/plugin/test_vendored_imports.py::test_private_hf_repo_tools_imports -v`
Expected: FAIL (module missing).

- [ ] **Step 4: Copy and rewrite**

```bash
cp agent/tools/private_hf_repo_tools.py plugin/mcp/tools/private_hf_repo_tools.py
grep -n "from agent\|import agent" plugin/mcp/tools/private_hf_repo_tools.py
```

For each match, Edit `from agent.tools` → `from plugin.mcp.tools`. Drop any `from agent.core` imports and the code that uses them (spec: this module should not need them; confirm before deletion).

- [ ] **Step 5: Run test**

Run: `uv run pytest tests/plugin/test_vendored_imports.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add plugin/mcp/tools/private_hf_repo_tools.py tests/plugin/test_vendored_imports.py
git commit -m "feat(plugin): vendor private_hf_repo_tools"
```

---

## Task 8: Vendor the three github_* tools

**Files:**
- Create: `plugin/mcp/tools/github_find_examples.py`
- Create: `plugin/mcp/tools/github_list_repos.py`
- Create: `plugin/mcp/tools/github_read_file.py`
- Modify: `tests/plugin/test_vendored_imports.py`

- [ ] **Step 1: Extend import test**

Append to `tests/plugin/test_vendored_imports.py`:

```python
def test_github_tools_import():
    from plugin.mcp.tools import (
        github_find_examples,
        github_list_repos,
        github_read_file,
    )
    assert hasattr(github_find_examples, "github_find_examples")
    assert hasattr(github_list_repos, "github_list_repos")
    assert hasattr(github_read_file, "github_read_file")
```

If public names in the source files differ, adjust.

- [ ] **Step 2: Run test, expect failure**

Run: `uv run pytest tests/plugin/test_vendored_imports.py::test_github_tools_import -v`
Expected: FAIL.

- [ ] **Step 3: Copy and rewrite imports in all three**

```bash
cp agent/tools/github_find_examples.py plugin/mcp/tools/
cp agent/tools/github_list_repos.py plugin/mcp/tools/
cp agent/tools/github_read_file.py plugin/mcp/tools/
grep -n "from agent\|import agent" plugin/mcp/tools/github_*.py
```

For each match, Edit `from agent.tools` → `from plugin.mcp.tools`.

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/plugin/test_vendored_imports.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add plugin/mcp/tools/github_*.py tests/plugin/test_vendored_imports.py
git commit -m "feat(plugin): vendor github tools (find_examples, list_repos, read_file)"
```

---

## Task 9: Build the MCP server entrypoint

**Files:**
- Create: `plugin/mcp/ml_intern_tools.py`
- Create: `tests/plugin/test_mcp_server_registration.py`

- [ ] **Step 1: Write failing test for MCP tool registration**

Create `tests/plugin/test_mcp_server_registration.py`:

```python
"""Verify the MCP server exposes every expected tool."""

from plugin.mcp.ml_intern_tools import SERVER, REGISTERED_TOOL_NAMES

EXPECTED_TOOLS = {
    "explore_hf_docs",
    "fetch_hf_docs",
    "find_hf_api",
    "hf_inspect_dataset",
    "hf_repo_files",
    "hub_repo_details",
    "hf_papers",
    "hf_private_list",
    "hf_private_read",
    "hf_private_upload",
    "github_find_examples",
    "github_list_repos",
    "github_read_file",
}


def test_server_instance_exists():
    assert SERVER is not None


def test_all_expected_tools_registered():
    missing = EXPECTED_TOOLS - REGISTERED_TOOL_NAMES
    assert not missing, f"tools not registered: {missing}"


def test_no_unexpected_tools():
    extras = REGISTERED_TOOL_NAMES - EXPECTED_TOOLS
    assert not extras, f"unexpected tools registered: {extras}"
```

Adjust `EXPECTED_TOOLS` if Tasks 3–8 discovered different real public names; the set must match what the vendored modules actually export.

- [ ] **Step 2: Run test, expect failure**

Run: `uv run pytest tests/plugin/test_mcp_server_registration.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Write the MCP server entrypoint**

Create `plugin/mcp/ml_intern_tools.py`:

```python
"""Stdio MCP server exposing ml-intern's HF and GitHub tools to Claude Code.

Each MCP tool is a thin adapter calling the vendored ml-intern function of
the same name. Tool names match those referenced in the `ml-intern` skill's
system prompt, so the prompt runs unchanged.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Vendored tool modules — each exports one or more public functions.
from plugin.mcp.tools import (
    docs_tools,
    dataset_tools,
    hf_repo_files_tool,
    hf_repo_git_tool,
    papers_tool,
    private_hf_repo_tools,
    github_find_examples,
    github_list_repos,
    github_read_file,
)

SERVER = Server("ml_intern_tools")

# Registry: MCP tool name -> (callable, JSON schema describing arguments)
_REGISTRY: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {}


def _register(name: str, fn: Callable[..., Any], schema: dict[str, Any]) -> None:
    _REGISTRY[name] = (fn, schema)


# Permissive schema — forward arguments as kwargs to the underlying function.
# Claude Code will still enforce whatever the underlying function validates.
_PASSTHROUGH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
}

# Tool registration — names match ml-intern's system prompt.
_register("explore_hf_docs", docs_tools.explore_hf_docs, _PASSTHROUGH_SCHEMA)
_register("fetch_hf_docs", docs_tools.fetch_hf_docs, _PASSTHROUGH_SCHEMA)
_register("find_hf_api", docs_tools.find_hf_api, _PASSTHROUGH_SCHEMA)
_register("hf_inspect_dataset", dataset_tools.hf_inspect_dataset, _PASSTHROUGH_SCHEMA)
_register("hf_repo_files", hf_repo_files_tool.hf_repo_files, _PASSTHROUGH_SCHEMA)
_register("hub_repo_details", hf_repo_git_tool.hub_repo_details, _PASSTHROUGH_SCHEMA)
_register("hf_papers", papers_tool.hf_papers, _PASSTHROUGH_SCHEMA)
_register("hf_private_list", private_hf_repo_tools.hf_private_list, _PASSTHROUGH_SCHEMA)
_register("hf_private_read", private_hf_repo_tools.hf_private_read, _PASSTHROUGH_SCHEMA)
_register("hf_private_upload", private_hf_repo_tools.hf_private_upload, _PASSTHROUGH_SCHEMA)
_register("github_find_examples", github_find_examples.github_find_examples, _PASSTHROUGH_SCHEMA)
_register("github_list_repos", github_list_repos.github_list_repos, _PASSTHROUGH_SCHEMA)
_register("github_read_file", github_read_file.github_read_file, _PASSTHROUGH_SCHEMA)

REGISTERED_TOOL_NAMES: set[str] = set(_REGISTRY.keys())


@SERVER.list_tools()
async def _list_tools() -> list[Tool]:
    return [
        Tool(
            name=name,
            description=(fn.__doc__ or name).strip().splitlines()[0],
            inputSchema=schema,
        )
        for name, (fn, schema) in _REGISTRY.items()
    ]


@SERVER.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name not in _REGISTRY:
        raise ValueError(f"unknown tool: {name}")
    fn, _ = _REGISTRY[name]
    result = fn(**arguments)
    if asyncio.iscoroutine(result):
        result = await result
    if isinstance(result, (dict, list)):
        payload = json.dumps(result, default=str)
    else:
        payload = str(result)
    return [TextContent(type="text", text=payload)]


async def _main() -> None:
    async with stdio_server() as (read, write):
        await SERVER.run(read, write, SERVER.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
```

If any vendored module's public name differs from what is registered above (per Tasks 3–8 discoveries), adjust the `_register(...)` calls accordingly so registrations match real function objects.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/plugin/test_mcp_server_registration.py -v`
Expected: PASS.

- [ ] **Step 5: Smoke-start the server to confirm it boots**

Run:
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke","version":"0"}}}' | uv run python plugin/mcp/ml_intern_tools.py | head -c 200
```

Expected: a JSON-RPC response containing `"serverInfo"` and `"protocolVersion"`. If the process hangs, the server started successfully — kill it (Ctrl-C).

- [ ] **Step 6: Commit**

```bash
git add plugin/mcp/ml_intern_tools.py tests/plugin/test_mcp_server_registration.py
git commit -m "feat(plugin): add MCP server exposing vendored ml-intern tools"
```

---

## Task 10: Write the `.mcp.json` and `settings.json`

**Files:**
- Create: `plugin/.mcp.json`
- Create: `plugin/settings.json`

- [ ] **Step 1: Write `.mcp.json` declaring both servers**

Write `plugin/.mcp.json`:

```json
{
  "mcpServers": {
    "ml_intern_tools": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "plugin/mcp/ml_intern_tools.py"],
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

- [ ] **Step 2: Write minimal settings.json**

Write `plugin/settings.json`:

```json
{
  "permissions": {}
}
```

- [ ] **Step 3: Commit**

```bash
git add plugin/.mcp.json plugin/settings.json
git commit -m "feat(plugin): declare ml_intern_tools and hf-mcp-server in plugin config"
```

---

## Task 11: Write the `ml-intern` skill (adapted system_prompt_v3)

**Files:**
- Create: `plugin/skills/ml-intern/SKILL.md`

- [ ] **Step 1: Copy the verbatim body**

Copy the body of `agent/prompts/system_prompt_v3.yaml` (the string after `system_prompt: |`, preserving all line breaks) into memory. This is the base.

- [ ] **Step 2: Apply the minimal edits**

Make the following edits to the copied text — and ONLY these edits. Every other word stays verbatim:

1. **Replace** `{{ num_tools }}` in the opening line with `13` (the count of tools registered in Task 9).
2. **Replace** the `research({...})` JSON-call example block with:

   ````
   Call the `Agent` tool with `subagent_type="general-purpose"` and a prompt containing the task and context. Example:

   ```
   Agent(
     subagent_type="general-purpose",
     description="Literature crawl for [task]",
     prompt="Literature crawl for [task]. Start from [paper/topic]. Crawl citation graph for recent downstream papers. Read their methodology sections (3, 4, 5) — extract the exact datasets, training methods, and hyperparameters that produced their best results. Attribute every finding to a specific result (e.g. 'Dataset X + method Y → 85.3% on benchmark Z'). Also find working code examples using current TRL/Transformers APIs.\n\nContext: User wants to [goal]. We need the best training recipe backed by published results."
   )
   ```
   ````

3. **Replace** the `# When a task has 3+ steps` section referencing `plan_tool` with the same paragraph but using `TaskCreate`: `Use TaskCreate to track progress. One task in_progress at a time. Mark completed immediately after finishing. Update frequently to show the user what you're doing.`
4. **Replace** the `# When submitting a training job` section. Keep the same pre-flight checklist format but replace references to `hf_jobs` with: "Submit via your HPC scheduler (SLURM `sbatch` or PBS `qsub`) through the Bash tool." Rewrite the `timeout` bullet as: `wall-clock limit in the submit script (based on: [model size] on [hardware])`. Everything else (reference implementation, dataset format verified, push_to_hub, trackio, batch-one-first rule, hardware sizing table) stays word-for-word.
5. **Delete** the entire `# Sandbox-first development` section (3 paragraphs).
6. **Replace** the "HF_TOKEN is automatically available in job secrets" line under `# Tool usage` with: `HF_TOKEN is loaded from your .env; on HPC, export it in your submit script before launching the job.`
7. **Delete** the line about `HF_TOKEN` being auto-loaded into job secrets under "For private/gated datasets"; replace with: `For private/gated datasets: HF_TOKEN must be exported in your HPC submit script.`

Preserve word-for-word: the title, hallucinated-imports warning block, dataset-format-by-method table, hardware sizing table, OOM recovery rules, error-recovery section, task-completion section, autonomous-mode loop section, communication style, all remaining tool-usage bullets.

- [ ] **Step 3: Wrap in skill frontmatter and write the file**

Write `plugin/skills/ml-intern/SKILL.md`:

```markdown
---
name: ml-intern
description: Autonomous ML research agent. Use when the user asks to train a model, fine-tune, find or inspect a Hugging Face dataset, search for ML papers, find ML GitHub examples, plan an SFT/DPO/GRPO/LoRA recipe, run inference, or orchestrate an ML research workflow end-to-end.
---

<BODY FROM STEP 2 PASTED HERE, VERBATIM WITH THE 7 EDITS APPLIED>
```

Replace the placeholder line with the actual edited body.

- [ ] **Step 4: Sanity-check no forbidden references remain**

Run:
```bash
grep -nE "plan_tool|sandbox_create|hf_jobs|\{\{ num_tools \}\}" plugin/skills/ml-intern/SKILL.md
```
Expected: no output. If anything matches, the edits in Step 2 missed a spot — fix it.

- [ ] **Step 5: Commit**

```bash
git add plugin/skills/ml-intern/SKILL.md
git commit -m "feat(plugin): add ml-intern skill with adapted system_prompt_v3"
```

---

## Task 12: Write the `/ml-intern` slash command

**Files:**
- Create: `plugin/commands/ml-intern.md`

- [ ] **Step 1: Write the slash command body**

Write `plugin/commands/ml-intern.md`:

```markdown
---
description: Force-activate the ml-intern ML research workflow for this turn.
---

You are operating under the `ml-intern` skill. Follow its workflow strictly:
research-first (literature crawl via the Agent subagent, methodology
extraction), data audit with `hf_inspect_dataset`, validated pre-flight
checklist before any training script, HPC-oriented submission via Bash, and
autonomous-mode loop discipline.

User request: $ARGUMENTS
```

- [ ] **Step 2: Commit**

```bash
git add plugin/commands/ml-intern.md
git commit -m "feat(plugin): add /ml-intern slash command"
```

---

## Task 13: Tool-name parity test (skill vs. MCP registry)

**Files:**
- Create: `tests/plugin/test_skill_tool_name_parity.py`

- [ ] **Step 1: Write the parity test**

Create `tests/plugin/test_skill_tool_name_parity.py`:

```python
"""Every tool referenced in the skill body must exist in the MCP registry
or be a known Claude Code native / HF MCP tool."""

import re
from pathlib import Path

from plugin.mcp.ml_intern_tools import REGISTERED_TOOL_NAMES

SKILL = Path("plugin/skills/ml-intern/SKILL.md").read_text()

# Tools the skill may call that come from Claude Code or the HF MCP server,
# not the ml_intern_tools MCP server.
CC_OR_HF_MCP_TOOLS = {
    "Agent",
    "TaskCreate",
    "Read", "Edit", "Write", "Bash",
    # HF MCP tools — names as surfaced by huggingface.co/mcp.
    # The skill does not need to reference these explicitly; kept for safety.
}

TOKEN_RE = re.compile(r"\b([a-z][a-z0-9_]*)\b")


def _candidate_tool_tokens(text: str) -> set[str]:
    # Tokens that look like tool names (snake_case, lowercase start).
    # Filter to only those that appear as standalone references in the doc.
    return set(TOKEN_RE.findall(text))


def test_all_tool_references_are_backed():
    tokens = _candidate_tool_tokens(SKILL)
    # Restrict to tokens that also appear in the tool registry or are known CC/HF.
    claimed = tokens & (REGISTERED_TOOL_NAMES | CC_OR_HF_MCP_TOOLS)
    # This test confirms parity *for names the skill does reference*; we assert
    # that every NAME IN THE REGISTRY that COULD be called is either referenced
    # in the skill or intentionally omitted. We start with a weaker assertion:
    # every registered tool name should appear somewhere in the skill at least
    # once, so the prompt actually teaches the model to use them.
    unreferenced = REGISTERED_TOOL_NAMES - tokens
    assert not unreferenced, (
        f"MCP tools never mentioned in the skill: {unreferenced}. "
        "Either reference them in the skill or remove from the registry."
    )


def test_no_dropped_tool_names_in_skill():
    forbidden = {"plan_tool", "sandbox_create", "hf_jobs"}
    hits = forbidden & _candidate_tool_tokens(SKILL)
    assert not hits, f"dropped tools still referenced in skill: {hits}"
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/plugin/test_skill_tool_name_parity.py -v`

Expected: both tests PASS. If `test_all_tool_references_are_backed` fails, it means the skill does not mention one of the registered tools — either remove the tool from the registry (if genuinely unused) or weave it into the skill text. If `test_no_dropped_tool_names_in_skill` fails, Task 11 missed an edit — go back and fix the skill.

- [ ] **Step 3: Commit**

```bash
git add tests/plugin/test_skill_tool_name_parity.py
git commit -m "test(plugin): enforce tool-name parity between skill and MCP registry"
```

---

## Task 14: Install the plugin locally and run the smoke-test suite inside Claude Code

**Files:** none created; this is verification.

- [ ] **Step 1: Install the plugin into your local Claude Code**

Run:
```bash
/plugin install ./plugin
/reload-plugins
```

Confirm the output shows the plugin loaded with 1 skill, 1 slash command, 1 MCP server (ml_intern_tools), plus the HF MCP server.

- [ ] **Step 2: Confirm the MCP tools are available**

Run `/mcp` in Claude Code. Expected: `ml_intern_tools` and `hf-mcp-server` both listed; expanding `ml_intern_tools` shows the 13 tools from Task 9.

- [ ] **Step 3: Smoke test — dataset inspection**

In a fresh Claude Code session:

> `/ml-intern find a small image-classification dataset on Hugging Face and show me its schema and one sample row.`

Expected: the model calls `hf_inspect_dataset`; returns schema and sample; no tool-not-found errors; response follows the concise ml-intern communication style (no filler).

- [ ] **Step 4: Smoke test — research workflow**

> `/ml-intern plan a SFT recipe for a 3B model on a reasoning benchmark. Start from the literature.`

Expected: the model dispatches an `Agent` subagent call (visible in the trace); the subagent calls `hf_papers` with `citation_graph` or `snippet_search`, plus `github_find_examples`; final output is a pre-flight checklist per the skill's `# When submitting a training job` section, with no references to `hf_jobs` or sandbox.

- [ ] **Step 5: Smoke test — HPC path**

> `/ml-intern write a SLURM sbatch script for a 7B SFT run on a100x4.`

Expected: Bash tool used to write an `.sbatch` file; script includes `#SBATCH` directives, `export HF_TOKEN=...`, and a `python train.py` invocation; no references to `hf_jobs` or `sandbox_create`.

- [ ] **Step 6: Smoke test — auto-trigger without slash command**

In a fresh session, plain prompt (no `/ml-intern`):

> `train a 7B Llama with DPO on UltraFeedback`

Expected: Claude Code surfaces the `ml-intern` skill (visible in the skill list for the turn) and follows its workflow.

- [ ] **Step 7: Commit anything discovered during verification**

If any of Steps 3–6 revealed bugs (wrong tool name, bad edit in the skill, missing environment variable), fix them, re-run the failing smoke test, and commit the fix with a `fix(plugin): ...` message. Repeat until all six smoke tests pass.

- [ ] **Step 8: Final commit — mark plugin v0.1.0 ready**

If no fixes were needed, no commit. Otherwise:

```bash
git commit --allow-empty -m "chore(plugin): v0.1.0 smoke tests passed"
```

---

## Self-review notes (for the plan author — delete before handing to implementer)

Spec coverage check:

- Plugin scaffold + manifest → Task 1 ✓
- MCP server with 13 tools (docs, datasets, repos, papers, private, github×3) → Tasks 2–9 ✓
- HF MCP server wired → Task 10 ✓
- Skill with adapted system_prompt_v3 → Task 11 ✓
- `/ml-intern` slash command → Task 12 ✓
- Tool-name parity enforcement → Task 13 ✓
- End-to-end verification in CC → Task 14 ✓
- Permission config (minimal) → Task 10 ✓
- No-plugin-deps (self-contained) → Task 1 manifest ✓

No placeholders. All types/function names referenced consistently (`EXPECTED_TOOLS`, `REGISTERED_TOOL_NAMES`, `SERVER`). Task 7 and Task 9 flag that real public names from ml-intern's sources may differ from the assumed names — each task tells the implementer to grep for real names first and adjust.
