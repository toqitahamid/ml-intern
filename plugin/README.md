# ml-intern — Claude Code plugin

Autonomous ML research agent. Gives Claude Code the `ml-intern` workflow:
research-first (literature crawl, methodology extraction), data audit,
pre-flight validated training scripts, and HPC-ready submission patterns.

## Install

Pick one of the three options below depending on how you got the plugin.

### Option A — Local install (you have the ml-intern repo cloned)

From inside a Claude Code session:

```
/plugin install /absolute/path/to/ml-intern/plugin
/reload-plugins
```

Replace `/absolute/path/to/ml-intern/plugin` with the real path on your
machine — for example `/Users/you/code/ml-intern/plugin`. Claude Code reads
the plugin directly from disk; edits to the plugin take effect after
`/reload-plugins`.

### Option B — GitHub install (you or a teammate published the repo)

```
/plugin install github:toqitahamid/ml-intern --path plugin
/reload-plugins
```

Claude Code clones the repo into its plugin cache and points at the
`plugin/` subdirectory. Use this if you want your team to pull updates with
a plain `/plugin update ml-intern`.

### Option C — Marketplace install (publicly discoverable)

Not yet published. To publish later, either (1) submit to an existing Claude
Code marketplace or (2) host a marketplace manifest yourself and add it with
`/plugin marketplace add <url>`. Once listed, users just run:

```
/plugin install ml-intern
/reload-plugins
```

---

After installation, confirm:

```
/plugin
```

…should list `ml-intern` with 1 skill, 1 slash command, and 2 MCP servers
(`ml_intern_tools` + `hf-mcp-server`). Then:

```
/mcp
```

…should show all 10 tools under `ml_intern_tools`.

## Environment setup

Two tokens are required for the plugin's tools to work without rate-limiting
or auth errors.

### 1. Get the tokens

- **`HF_TOKEN`** — create at https://huggingface.co/settings/tokens
  - Type: **Fine-grained** (recommended) or **Read** token
  - Scopes: `Read access to contents of all public gated repos` is enough
    for most workflows; add `Write` only if you intend to push models/datasets
- **`GITHUB_TOKEN`** — create at https://github.com/settings/tokens
  - Type: **Classic PAT** or **Fine-grained**
  - Scopes: only `public_repo` is needed for ML code search

### 2. Set them in your shell profile

Add to `~/.zshrc` (macOS default) or `~/.bashrc`:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
```

Then reload your shell:

```bash
source ~/.zshrc   # or: source ~/.bashrc
```

Verify:

```bash
echo $HF_TOKEN | head -c 5   # should print "hf_xx"
echo $GITHUB_TOKEN | head -c 5
```

Restart Claude Code after setting the tokens — it reads the env at launch.

### Alternative: `.env` file

If you prefer per-repo scope, put the tokens in `.env` at the ml-intern
repo root:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

The repo already ignores `.env` in `.gitignore`, so you won't commit them
by accident. Do **not** check these files into any git repo.

### What happens without them

| Missing | Effect |
| --- | --- |
| `HF_TOKEN` | Public HF data works but is rate-limited; private/gated datasets and models fail with 401 |
| `GITHUB_TOKEN` | GitHub search capped at 10 requests/min, unauthenticated — two literature-search turns will rate-limit you |
| Both | The HF MCP server's first call triggers an OAuth dialog in Claude Code, which handles HF auth interactively; GitHub tools still fail |

## Usage

- Auto-triggers on ML tasks: "train a model", "fine-tune", "find a dataset on
  HF", "plan a DPO recipe", etc.
- Or force-activate with `/ml-intern <prompt>` for ambiguous prompts.

## What's inside

- Skill with the `ml-intern` persona and workflow (from `system_prompt_v3`).
- MCP server exposing 10 HF and GitHub tools: `explore_hf_docs`, `fetch_hf_docs`,
  `find_hf_api`, `hf_inspect_dataset`, `hf_repo_files`, `hf_repo_git`,
  `hf_papers`, `github_find_examples`, `github_list_repos`, `github_read_file`.
- The official Hugging Face MCP server (`huggingface.co/mcp`) wired in
  automatically.
