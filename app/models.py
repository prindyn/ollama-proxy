"""Pydantic models for request/response validation."""

from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field, validator


class Message(BaseModel):
    """Chat message model."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

    @validator("role")
    def validate_role(cls, v):
        if v not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Invalid role")
        return v


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""

    model: str
    messages: List[Message]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = "auto"
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """Response choice."""

    index: int
    message: Message
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = "default"
