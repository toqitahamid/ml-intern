"""Verify the MCP server exposes every expected tool."""

from plugin.mcp.ml_intern_tools import REGISTERED_TOOL_NAMES, SERVER

EXPECTED_TOOLS = {
    "explore_hf_docs",
    "fetch_hf_docs",
    "find_hf_api",
    "hf_inspect_dataset",
    "hf_repo_files",
    "hf_repo_git",
    "hf_papers",
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
