"""Provider implementations for different LLM services using LangChain."""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

import httpx
from loguru import logger
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Import from your existing modules
from .logger import LoggingCallbackHandler
from .tools import OpenAIToolParser


class BaseProvider(ABC):
    """Abstract base class for LLM providers using LangChain."""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.llm = None
        self.tool_parser = OpenAIToolParser()

    @abstractmethod
    def _create_llm(self):
        """Create the LangChain LLM instance."""
        pass

    def _convert_messages(self, messages: List[Dict[str, str]]):
        """Convert OpenAI-style messages to LangChain format."""
        converted = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                converted.append(SystemMessage(content=content))
            elif role == "user":
                converted.append(HumanMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))

        return converted

    async def chat_completion(
        self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None
    ) -> str:
        """Process chat completion request using LangChain agent."""
        if not self.llm:
            self.llm = self._create_llm()

        # Parse tools if provided
        parsed_tools = self.tool_parser.parse(tools) if tools else []

        # Create agent
        agent = initialize_agent(
            llm=self.llm,
            tools=parsed_tools,
            agent=AgentType.OPENAI_MULTI_FUNCTIONS,
            handle_parsing_errors=True,
            verbose=False,
        )

        # Convert messages to proper format
        # converted_messages = self._convert_messages(messages)

        result = agent.run(messages)
        return result

    @abstractmethod
    async def list_models(self) -> List[Dict[str, str]]:
        """List available models."""
        pass


class OllamaProvider(BaseProvider):
    """Ollama provider implementation using LangChain."""

    def __init__(self, model: str, base_url: Optional[str] = None):
        super().__init__(model)
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )

    def _create_llm(self):
        """Create Ollama LLM instance."""
        return Ollama(
            model=self.model,
            base_url=self.base_url,
            callbacks=[LoggingCallbackHandler()],
        )

    async def list_models(self) -> List[Dict[str, str]]:
        """List available Ollama models."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                return [{"id": m["name"], "object": "model"} for m in models]
            except httpx.HTTPStatusError as e:
                raise Exception(
                    f"Ollama API error: {e.response.status_code} - {e.response.text}"
                )
            except Exception as e:
                raise Exception(f"Failed to fetch Ollama models: {e}")


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation using LangChain."""

    def __init__(
        self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None
    ):
        super().__init__(model)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

    def _create_llm(self):
        """Create OpenAI LLM instance."""
        return ChatOpenAI(
            model=self.model,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
            callbacks=[LoggingCallbackHandler()],
        )

    async def list_models(self) -> List[Dict[str, str]]:
        """List available OpenAI models."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/models", headers=headers)
                response.raise_for_status()
                models_data = response.json()
                # Filter for chat models
                chat_models = [
                    m
                    for m in models_data.get("data", [])
                    if "gpt" in m.get("id", "").lower()
                ]
                return [{"id": m["id"], "object": "model"} for m in chat_models]
            except Exception as e:
                logger.warning(f"Failed to fetch OpenAI models: {e}")
                return []


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek provider implementation (OpenAI-compatible) using LangChain."""

    def __init__(self, model: str, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key is required")
        super().__init__(model, api_key, "https://api.deepseek.com/v1")

    async def list_models(self) -> List[Dict[str, str]]:
        """List available DeepSeek models."""
        try:
            return await super().list_models()
        except Exception as e:
            logger.warning(f"Failed to fetch DeepSeek models: {e}")
            # Return known DeepSeek models as fallback
            return [
                {"id": "deepseek-chat", "object": "model"},
                {"id": "deepseek-coder", "object": "model"},
            ]


# Single source of truth for provider configurations
PROVIDER_REGISTRY = {
    "ollama": {
        "class": OllamaProvider,
        "api_key_env": None,
        "requires_api_key": False,
    },
    "openai": {
        "class": OpenAIProvider,
        "api_key_env": "OPENAI_API_KEY",
        "requires_api_key": True,
    },
    "deepseek": {
        "class": DeepSeekProvider,
        "api_key_env": "DEEPSEEK_API_KEY",
        "requires_api_key": True,
    },
}


# Factory instance
_factory: Optional[Dict[str, str]] = None


def get_provider(provider_type: str, model: str, **kwargs) -> BaseProvider:
    """Create a provider instance."""
    provider_config = PROVIDER_REGISTRY.get(provider_type.lower())
    if not provider_config:
        raise ValueError(f"Unsupported provider: {provider_type}")

    provider_class = provider_config["class"]
    return provider_class(model, **kwargs)


async def get_provider_for_model(
    model: str, default_provider: str = "ollama"
) -> BaseProvider:
    """Get provider instance for a model, with auto-detection."""
    global _factory

    if _factory is None:
        _factory = await _init_factory()

    provider_type = _factory.get(model, default_provider)

    # Check for model prefixes if exact match not found
    if provider_type == default_provider and model not in _factory:
        for mapped_model, mapped_provider in _factory.items():
            if model.startswith(mapped_model):
                provider_type = mapped_provider
                break

    return get_provider(provider_type, model)


async def _init_factory() -> Dict[str, str]:
    """Initialize model-to-provider mapping."""
    model_mapping = {}

    for provider_name, provider_config in PROVIDER_REGISTRY.items():
        # Skip API providers if no key is set
        if not is_provider_available(provider_name):
            continue

        try:
            provider_class = provider_config["class"]
            provider = provider_class("dummy")
            models = await provider.list_models()
            for model in models:
                model_mapping[model["id"]] = provider_name
        except Exception as e:
            logger.debug(f"Failed to list {provider_name} models: {e}")

    return model_mapping


def list_providers() -> List[str]:
    """List available provider types."""
    return list(PROVIDER_REGISTRY.keys())


async def list_all_models() -> List[Dict[str, Any]]:
    """List all available models from all providers."""
    all_models = []

    for provider_name, provider_config in PROVIDER_REGISTRY.items():
        # Skip API providers if no key is set
        if not is_provider_available(provider_name):
            continue

        try:
            provider_class = provider_config["class"]
            provider = provider_class("dummy")
            models = await provider.list_models()
            for model in models:
                model["provider"] = provider_name
                all_models.append(model)
        except Exception as e:
            logger.debug(f"Failed to list {provider_name} models: {e}")

    return all_models


def is_provider_available(provider_name: str) -> bool:
    """Check if a provider is available (has required configuration)."""
    provider_config = PROVIDER_REGISTRY.get(provider_name.lower())
    if not provider_config:
        return False

    if provider_config["requires_api_key"]:
        api_key_env = provider_config["api_key_env"]
        return bool(api_key_env and os.getenv(api_key_env))

    return True


def get_available_providers() -> List[str]:
    """List only available (configured) providers."""
    return [name for name in PROVIDER_REGISTRY.keys() if is_provider_available(name)]
