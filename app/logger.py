"""Logging configuration and utilities."""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

from loguru import logger
from langchain.callbacks.base import BaseCallbackHandler


# Constants
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_DIR = "logs"
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level:<8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def configure_logger() -> None:
    """Configure loguru with environment-based settings."""
    # Get configuration
    level = os.getenv("LOGURU_LEVEL", DEFAULT_LOG_LEVEL).upper()
    log_file = os.getenv("LOG_FILE")

    # Reset and configure stderr output
    logger.remove()
    logger.add(sys.stderr, level=level, format=LOG_FORMAT)

    # Add file output if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path,
            level=level,
            format=LOG_FORMAT,
            rotation="1 MB",
            retention="10 days",
        )


def log_conversation(
    endpoint: str, request: Dict[str, Any], response: Dict[str, Any]
) -> None:
    """Log API conversations to JSONL files.

    Args:
        endpoint: API endpoint path
        request: Request data
        response: Response data
    """
    # Setup log directory and file
    log_dir = Path(os.getenv("LOG_DIR", DEFAULT_LOG_DIR))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Extract endpoint name
    endpoint_name = endpoint.strip("/").split("/")[-1] or "root"
    log_file = log_dir / f"{endpoint_name}.ndjson"

    # Write conversation
    try:
        with log_file.open("a") as f:
            json.dump({"type": "request", "data": request}, f)
            f.write("\n")
            json.dump({"type": "response", "data": response}, f)
            f.write("\n")
    except Exception as e:
        logger.error(f"Failed to log conversation: {e}")


class LoggingCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for logging LLM interactions."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Log LLM prompts."""
        for i, prompt in enumerate(prompts):
            logger.info(f"[LLM PROMPT {i+1}/{len(prompts)}] >>>\n{prompt}")

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Log LLM responses."""
        try:
            # Extract text from generations
            for i, generation in enumerate(response.generations):
                if generation:
                    text = generation[0].text if generation else "No response"
                    logger.info(f"[LLM RESPONSE {i+1}] <<<\n{text}")
        except Exception as e:
            logger.error(f"Failed to log LLM response: {e}")

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Log LLM errors."""
        logger.error(f"[LLM ERROR] {error}")

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """Log tool invocations."""
        tool_name = serialized.get("name", "Unknown")
        logger.info(f"[TOOL START] {tool_name} with input: {input_str}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Log tool outputs."""
        logger.info(f"[TOOL OUTPUT] {output}")

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Log tool errors."""
        logger.error(f"[TOOL ERROR] {error}")
