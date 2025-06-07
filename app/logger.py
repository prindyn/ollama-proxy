"""Loguru configuration module."""

import os
import sys
from loguru import logger
from pathlib import Path
import json


def configure_logger() -> None:
    """Configure loguru based on environment variables."""
    level = os.getenv("LOGURU_LEVEL", "INFO").upper()
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, format=fmt)

    log_file = os.getenv("LOG_FILE")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level, format=fmt, rotation="1 MB")


def log_conversation(request: dict, response: dict) -> None:
    """Append the request and response as individual JSON lines."""
    log_dir = os.getenv("LOG_DIR", "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / "conversations.log"
    with log_path.open("a") as f:
        json.dump(request, f)
        f.write("\n")
        json.dump(response, f)
        f.write("\n")

