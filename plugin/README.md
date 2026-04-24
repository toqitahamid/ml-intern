# ml-intern — Claude Code plugin

Autonomous ML research agent. Gives Claude Code the `ml-intern` workflow:
research-first (literature crawl, methodology extraction), data audit,
pre-flight validated training scripts, and HPC-ready submission patterns.

## Install

```
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
