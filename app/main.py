"""Multi-provider LLM Proxy API."""

from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Request
from loguru import logger

from .configs import Config
from .models import ChatCompletionRequest, ChatCompletionResponse, Message
from .providers import (
    get_provider,
    get_provider_for_model,
    list_all_models,
    get_available_providers,
)
from .logger import configure_logger, log_conversation
from .services import chat_completion_handler


# Configure logging
configure_logger()

# Initialize app
app = FastAPI(
    title=Config.APP_TITLE,
    description=Config.APP_DESCRIPTION,
    version=Config.API_VERSION,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    logger.info(f"{request.method} {request.url.path} from {request.client.host}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


@app.get("/")
async def health() -> Dict[str, Any]:
    """Health check."""
    response = {
        "status": "ok",
        "available_providers": get_available_providers(),
        "default_provider": Config.DEFAULT_PROVIDER,
    }
    logger.info("Health check requested")
    return response


@app.get("/v1/models")
async def list_models(provider: Optional[str] = Query(None)) -> Dict[str, Any]:
    """List available models."""
    try:
        if provider:
            logger.info(f"Listing models for provider: {provider}")
            # List models for specific provider
            provider_instance = get_provider(provider, model="dummy")
            models = await provider_instance.list_models()
            for model in models:
                model["provider"] = provider
            response = {"object": "list", "data": models}
        else:
            logger.info("Listing all available models")
            # List all models
            models = await list_all_models()
            response = {"object": "list", "data": models}

        logger.info(f"Found {len(response['data'])} models")
        return response
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Process chat completion request."""

    request_dict = request.model_dump()
    try:
        # Build response
        response = await chat_completion_handler(request)

        # Log conversation to file
        log_conversation("completions", request_dict, response.model_dump())

        return response
    except Exception as e:
        logger.exception("Chat completion failed")
        # Log failed request
        log_conversation("completions", request_dict, {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
