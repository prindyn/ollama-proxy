"""Pydantic models for request/response validation."""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field, validator


class Message(BaseModel):
    """Chat message model."""

    role: str
    content: str

    @validator("role")
    def validate_role(cls, v):
        if v not in {"system", "user", "assistant"}:
            raise ValueError("Invalid role")
        return v


class ChatCompletionRequest(BaseModel):
    """Chat completion request."""

    model: str
    messages: List[Message]
    tools: Optional[List] = []
    tools_choice: Optional[str] = "auto"
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Optional[Dict[str, int]] = None
