"""Smoke tests confirming every vendored module imports standalone."""


def test_types_imports():
    from plugin.mcp.tools.types import ToolResult
    assert ToolResult is not None


def test_utilities_imports():
    from plugin.mcp.tools import utilities
    assert hasattr(utilities, "truncate")
    assert hasattr(utilities, "format_date")


def test_docs_tools_imports():
    from plugin.mcp.tools import docs_tools
    assert hasattr(docs_tools, "explore_hf_docs_handler")
    assert hasattr(docs_tools, "hf_docs_fetch_handler")
    assert hasattr(docs_tools, "search_openapi_handler")
    assert hasattr(docs_tools, "EXPLORE_HF_DOCS_TOOL_SPEC")
    assert hasattr(docs_tools, "HF_DOCS_FETCH_TOOL_SPEC")


def test_dataset_tools_imports():
    from plugin.mcp.tools import dataset_tools
    assert hasattr(dataset_tools, "hf_inspect_dataset_handler")
    assert hasattr(dataset_tools, "HF_INSPECT_DATASET_TOOL_SPEC")


def test_hf_repo_files_imports():
    from plugin.mcp.tools import hf_repo_files_tool
    assert hasattr(hf_repo_files_tool, "hf_repo_files_handler")
    assert hasattr(hf_repo_files_tool, "HF_REPO_FILES_TOOL_SPEC")


def test_hf_repo_git_imports():
    from plugin.mcp.tools import hf_repo_git_tool
    assert hasattr(hf_repo_git_tool, "hf_repo_git_handler")
    assert hasattr(hf_repo_git_tool, "HF_REPO_GIT_TOOL_SPEC")


def test_papers_tool_imports():
    from plugin.mcp.tools import papers_tool
    assert hasattr(papers_tool, "hf_papers_handler")
    assert hasattr(papers_tool, "HF_PAPERS_TOOL_SPEC")


def test_github_tools_import():
    from plugin.mcp.tools import (
        github_find_examples,
        github_list_repos,
        github_read_file,
    )
    assert hasattr(github_find_examples, "github_find_examples_handler")
    assert hasattr(github_list_repos, "github_list_repos_handler")
    assert hasattr(github_read_file, "github_read_file_handler")
