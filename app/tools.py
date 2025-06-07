import os
import subprocess
from langchain.agents import Tool


def run_shell(command: str) -> str:
    """Execute a shell command and return its output."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr


def read_file(path: str) -> str:
    """Return the contents of a file."""
    with open(path, "r") as f:
        return f.read()


def write_file(path: str, content: str) -> str:
    """Write content to a file and return a confirmation message."""
    with open(path, "w") as f:
        f.write(content)
    return "written"


def list_directory(path: str = ".") -> str:
    """List files in a directory."""
    return "\n".join(os.listdir(path))


def web_search(query: str) -> str:
    """Placeholder web search."""
    return f"[search results for '{query}']"


def get_tools() -> list[Tool]:
    """Return Tool objects for the agent."""
    return [
        Tool.from_function(run_shell, name="shell", description="Run a shell command"),
        Tool.from_function(read_file, name="read_file", description="Read a file"),
        Tool.from_function(write_file, name="write_file", description="Write to a file"),
        Tool.from_function(list_directory, name="list_directory", description="List directory contents"),
        Tool.from_function(web_search, name="web_search", description="Search the web"),
    ]
