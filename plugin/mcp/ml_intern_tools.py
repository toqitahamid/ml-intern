"""Stdio MCP server exposing ml-intern's HF and GitHub tools to Claude Code.

Each MCP tool is a thin adapter calling the vendored ml-intern handler of
the same MCP name. Tool names match those referenced in the `ml-intern`
skill's system prompt, so the prompt runs unchanged.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from plugin.mcp.tools import (
    dataset_tools,
    docs_tools,
    github_find_examples,
    github_list_repos,
    github_read_file,
    hf_repo_files_tool,
    hf_repo_git_tool,
    papers_tool,
)

SERVER = Server("ml_intern_tools")

_Handler = Callable[[dict[str, Any]], Awaitable[tuple[str, bool]]]


def _spec(name: str, description: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": parameters
        or {"type": "object", "properties": {}, "additionalProperties": True},
    }


# Registry entries: MCP tool name -> (spec dict, async handler).
# Specs are sourced from the vendored TOOL_SPEC constants where available;
# find_hf_api has a runtime-generated spec in ml-intern, so we supply a
# permissive static fallback here (the handler still validates args).
_FIND_HF_API_STATIC_SPEC = _spec(
    "find_hf_api",
    "Find HuggingFace Hub REST API endpoints. Keyword search across endpoint "
    "summaries and descriptions; filter by category tag. Returns curl examples.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "tag": {"type": "string"},
        },
        "required": [],
    },
)

_REGISTRY: dict[str, tuple[dict[str, Any], _Handler]] = {
    "explore_hf_docs": (docs_tools.EXPLORE_HF_DOCS_TOOL_SPEC, docs_tools.explore_hf_docs_handler),
    "fetch_hf_docs": (docs_tools.HF_DOCS_FETCH_TOOL_SPEC, docs_tools.hf_docs_fetch_handler),
    "find_hf_api": (_FIND_HF_API_STATIC_SPEC, docs_tools.search_openapi_handler),
    "hf_inspect_dataset": (dataset_tools.HF_INSPECT_DATASET_TOOL_SPEC, dataset_tools.hf_inspect_dataset_handler),
    "hf_repo_files": (hf_repo_files_tool.HF_REPO_FILES_TOOL_SPEC, hf_repo_files_tool.hf_repo_files_handler),
    "hf_repo_git": (hf_repo_git_tool.HF_REPO_GIT_TOOL_SPEC, hf_repo_git_tool.hf_repo_git_handler),
    "hf_papers": (papers_tool.HF_PAPERS_TOOL_SPEC, papers_tool.hf_papers_handler),
    "github_find_examples": (
        github_find_examples.GITHUB_FIND_EXAMPLES_TOOL_SPEC,
        github_find_examples.github_find_examples_handler,
    ),
    "github_list_repos": (
        github_list_repos.GITHUB_LIST_REPOS_TOOL_SPEC,
        github_list_repos.github_list_repos_handler,
    ),
    "github_read_file": (
        github_read_file.GITHUB_READ_FILE_TOOL_SPEC,
        github_read_file.github_read_file_handler,
    ),
}

REGISTERED_TOOL_NAMES: set[str] = set(_REGISTRY.keys())


@SERVER.list_tools()
async def _list_tools() -> list[Tool]:
    tools: list[Tool] = []
    for name, (spec, _handler) in _REGISTRY.items():
        tools.append(
            Tool(
                name=name,
                description=str(spec.get("description", name)),
                inputSchema=spec.get("parameters")
                or {"type": "object", "properties": {}, "additionalProperties": True},
            )
        )
    return tools


@SERVER.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name not in _REGISTRY:
        raise ValueError(f"unknown tool: {name}")
    _spec_dict, handler = _REGISTRY[name]
    result = handler(arguments)
    if asyncio.iscoroutine(result):
        result_text, _is_error = await result
    else:
        result_text, _is_error = result
    if isinstance(result_text, (dict, list)):
        payload = json.dumps(result_text, default=str)
    else:
        payload = str(result_text)
    return [TextContent(type="text", text=payload)]


async def _main() -> None:
    async with stdio_server() as (read, write):
        await SERVER.run(read, write, SERVER.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
