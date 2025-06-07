import os
import json
from typing import Dict, Callable, Any, List

from loguru import logger


def read_file(path: str) -> str:
    """Return the contents of a text file."""
    with open(path, "r") as f:
        return f.read()


def write_file(path: str, content: str) -> str:
    """Write content to a text file and return a confirmation message."""
    with open(path, "w") as f:
        f.write(content)
    return "written"


def list_directory(path: str) -> List[str]:
    """Return a list of entries in a directory."""
    return os.listdir(path)


def web_search(query: str) -> str:
    """Placeholder web search implementation."""
    logger.warning("web_search called but no network access available")
    return f"Search results for '{query}' are unavailable"


TOOL_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "web_search": web_search,
}


def execute_tool_call(tool_call: dict) -> dict:
    name = tool_call.get("function", {}).get("name")
    args_str = tool_call.get("function", {}).get("arguments", "{}")
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        args = {}
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        result = f"Unknown tool: {name}"
    else:
        try:
            result = func(**args)
        except Exception as e:
            result = f"Error running {name}: {e}"
    return {
        "tool_call_id": tool_call.get("id"),
        "role": "tool",
        "name": name,
        "content": result,
    }
