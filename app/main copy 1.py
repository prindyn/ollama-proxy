"""Ollama Proxy API - OpenAI compatible interface for Ollama models."""

import os
from datetime import datetime
from typing import Dict, List, Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, AgentType

from .logger import configure_logger, log_conversation, LoggingCallbackHandler
from .tools import OpenAIToolParser

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOG_DIR = os.getenv("LOG_DIR", "logs")

# Setup
configure_logger()
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize app
app = FastAPI(title="Ollama Proxy")
app.state.client = None


@app.on_event("startup")
async def startup():
    """Initialize HTTP client."""
    logger.info(f"Connecting to {OLLAMA_BASE_URL}")
    app.state.client = httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=None)


@app.on_event("shutdown")
async def shutdown():
    """Close HTTP client."""
    if app.state.client:
        await app.state.client.aclose()


@app.get("/")
async def health():
    """Health check."""
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    """List available models."""
    try:
        response = await app.state.client.get("/api/tags")
        response.raise_for_status()

        models = response.json().get("models", [])
        return {
            "object": "list",
            "data": [{"id": m["name"], "object": "model"} for m in models],
        }
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        raise HTTPException(500, str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Process chat completion request."""
    body = await request.json()

    # Extract parameters
    model = body.get("model")
    messages = body.get("messages", [])
    tools = body.get("tools", [])

    # Validate
    if not model:
        raise HTTPException(400, "model is required")
    if not messages:
        raise HTTPException(400, "messages is required")

    try:
        # Initialize LLM and tools
        llm = Ollama(
            model=model, base_url=OLLAMA_BASE_URL, callbacks=[LoggingCallbackHandler()]
        )

        parsed_tools = OpenAIToolParser().parse(tools) if tools else []

        # Create and run agent
        agent = initialize_agent(
            llm=llm,
            tools=parsed_tools,
            agent=AgentType.OPENAI_MULTI_FUNCTIONS,
            handle_parsing_errors=True,
            verbose=False,
        )

        result = agent.run(messages)

        # Build response
        response = {
            "id": f"chatcmpl-{int(datetime.now().timestamp())}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": "stop",
                }
            ],
        }

        # Log and return
        log_conversation(request.url.path, body, response)
        return response

    except Exception as e:
        logger.exception("Chat completion failed")
        error = {"error": str(e)}
        log_conversation(request.url.path, body, error)
        raise HTTPException(500, str(e))
