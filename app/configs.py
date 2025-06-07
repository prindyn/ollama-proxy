"""Configuration for LLM Proxy API."""

import os


class Config:
    """Application configuration."""

    API_VERSION = "v1"
    APP_TITLE = "Multi-Provider LLM Proxy"
    APP_DESCRIPTION = "A unified API proxy for multiple LLM providers."
    DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "ollama")
