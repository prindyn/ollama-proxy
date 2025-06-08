"""Provider implementations for different LLM services."""

import os
import json
import uuid
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

import httpx
from loguru import logger


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str, **kwargs):
        self.model = model

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process chat completion request."""
        pass

    @abstractmethod
    async def list_models(self) -> List[Dict[str, str]]:
        """List available models."""
        pass


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""

    def __init__(
        self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None
    ):
        super().__init__(model)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make direct OpenAI API call."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build request payload
        payload = {"model": self.model, "messages": messages, "stream": False}

        # Add optional parameters
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        for param in [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "user",
        ]:
            if param in kwargs and kwargs[param] is not None:
                payload[param] = kwargs[param]

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                error_detail = e.response.text
                logger.error(
                    f"OpenAI API error: {e.response.status_code} - {error_detail}"
                )
                raise Exception(f"OpenAI API error: {error_detail}")
            except Exception as e:
                logger.error(f"Failed to call OpenAI: {e}")
                raise

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
                    if any(
                        prefix in m.get("id", "").lower() for prefix in ["gpt", "o1"]
                    )
                ]
                return [{"id": m["id"], "object": "model"} for m in chat_models]
            except Exception as e:
                logger.warning(f"Failed to fetch OpenAI models: {e}")
                # Return common models as fallback
                return [
                    {"id": "gpt-4", "object": "model"},
                    {"id": "gpt-4-turbo", "object": "model"},
                    {"id": "gpt-3.5-turbo", "object": "model"},
                ]


class OllamaProvider(BaseProvider):
    """Ollama provider implementation."""

    def __init__(self, model: str, base_url: Optional[str] = None):
        super().__init__(model)
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Ollama chat completion with tool call simulation."""

        # If tools are provided, inject instructions into the system message
        if tools:
            tool_instructions = self._create_tool_instructions(tools)
            # Prepend tool instructions to the first system message or add new one
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = (
                    tool_instructions + "\n\n" + messages[0]["content"]
                )
            else:
                messages.insert(0, {"role": "system", "content": tool_instructions})

        # Make Ollama API call
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 1.0),
                "top_p": kwargs.get("top_p", 1.0),
            },
        }

        if "max_tokens" in kwargs:
            payload["options"]["num_predict"] = kwargs["max_tokens"]

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat", json=payload, timeout=60.0
                )
                response.raise_for_status()
                ollama_response = response.json()

                # Convert Ollama response to OpenAI format
                content = ollama_response.get("message", {}).get("content", "")

                # Check if this is a tool call
                tool_calls = self._extract_tool_calls(content) if tools else None

                # Build OpenAI-compatible response
                message = {
                    "role": "assistant",
                    "content": content if not tool_calls else None,
                }

                if tool_calls:
                    message["tool_calls"] = tool_calls

                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
                    "object": "chat.completion",
                    "created": int(datetime.now().timestamp()),
                    "model": self.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if tool_calls else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": ollama_response.get("prompt_eval_count", 0),
                        "completion_tokens": ollama_response.get("eval_count", 0),
                        "total_tokens": ollama_response.get("prompt_eval_count", 0)
                        + ollama_response.get("eval_count", 0),
                    },
                }

            except Exception as e:
                logger.error(f"Ollama API error: {e}")
                raise

    def _create_tool_instructions(self, tools: List[Dict[str, Any]]) -> str:
        """Create instructions for tool usage."""
        tool_list = []
        for tool in tools:
            func = tool.get("function", {})
            tool_list.append(f"- {func.get('name')}: {func.get('description')}")

        return (
            "You have access to the following tools:\n"
            + "\n".join(tool_list)
            + "\n\nTo use a tool, respond ONLY with a JSON object in this exact format:\n"
            + '{"tool_call": {"name": "tool_name", "arguments": {...}}}\n'
            + "Do not include any other text when making a tool call."
        )

    def _extract_tool_calls(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from Ollama response."""
        try:
            # Look for JSON tool call in the content
            if '{"tool_call"' in content:
                start = content.find('{"tool_call"')
                end = content.rfind("}") + 1
                json_str = content[start:end]
                data = json.loads(json_str)

                if "tool_call" in data:
                    tool_info = data["tool_call"]
                    return [
                        {
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": tool_info["name"],
                                "arguments": json.dumps(tool_info["arguments"]),
                            },
                        }
                    ]
        except:
            pass
        return None

    async def list_models(self) -> List[Dict[str, str]]:
        """List available Ollama models."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                return [{"id": m["name"], "object": "model"} for m in models]
            except Exception as e:
                logger.error(f"Failed to fetch Ollama models: {e}")
                raise


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek provider (OpenAI-compatible)."""

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


# Provider registry
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
    """Get provider instance for a model."""
    # Simple provider detection based on model name
    if model.startswith("gpt") or model.startswith("o1"):
        provider_type = "openai"
    elif model.startswith("deepseek"):
        provider_type = "deepseek"
    else:
        provider_type = default_provider

    return get_provider(provider_type, model)


async def list_all_models() -> List[Dict[str, Any]]:
    """List all available models from all providers."""
    all_models = []

    for provider_name, provider_config in PROVIDER_REGISTRY.items():
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
    """Check if a provider is available."""
    provider_config = PROVIDER_REGISTRY.get(provider_name.lower())
    if not provider_config:
        return False

    if provider_config["requires_api_key"]:
        api_key_env = provider_config["api_key_env"]
        return bool(api_key_env and os.getenv(api_key_env))

    return True


def get_available_providers() -> List[str]:
    """List only available providers."""
    return [name for name in PROVIDER_REGISTRY.keys() if is_provider_available(name)]
