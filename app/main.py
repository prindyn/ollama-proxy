from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import os
import json
from loguru import logger
from .logger import configure_logger
from .tools import execute_tool_call
from .openai_adapter import format_chat_response, stream_chat_response

configure_logger()

app = FastAPI(title="Ollama Proxy", description="OpenAI compatible API for Ollama models")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
USE_LOCAL_TOOLS = os.getenv("ENABLE_LOCAL_TOOLS", "1") == "1"


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up proxy, connecting to {url}", url=OLLAMA_BASE_URL)
    app.state.client = httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=None)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down proxy")
    await app.state.client.aclose()

@app.get("/")
async def root():
    return {"message": "Ollama proxy running"}

@app.get("/v1/models")
async def list_models():
    """Return available models from Ollama"""
    try:
        logger.debug("Fetching models from Ollama")
        resp = await app.state.client.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()
        logger.debug("Models response: {}", data)
        models = [{"id": m["name"], "object": "model"} for m in data.get("models", [])]
        return {"object": "list", "data": models}
    except Exception as e:
        logger.error("Error fetching models: {}", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy OpenAI chat completions to Ollama"""
    body = await request.json()
    logger.debug("Chat completion request body: {}", body)
    model = body.get("model")
    messages = body.get("messages")
    stream = body.get("stream", False)
    if not model or not messages:
        raise HTTPException(status_code=400, detail="model and messages required")

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    # Add other fields if present
    for key in ["format", "options", "tools", "tool_choice"]:
        if key in body:
            payload[key] = body[key]

    try:
        logger.debug(
            "Forwarding chat completion to Ollama: model=%s stream=%s", model, stream
        )
        resp = await app.state.client.post("/api/chat", json=payload)
        resp.raise_for_status()
        if stream:
            async def iterator():
                async for chunk in stream_chat_response(resp):
                    yield chunk
            return StreamingResponse(iterator(), media_type="text/event-stream")

        data = resp.json()
        logger.debug("Ollama response: {}", data)

        if USE_LOCAL_TOOLS:
            tool_calls = []
            for choice in data.get("choices", []):
                calls = choice.get("message", {}).get("tool_calls")
                if calls:
                    tool_calls.extend(calls)
            if tool_calls:
                tool_messages = [execute_tool_call(tc) for tc in tool_calls]
                model_msgs = [c.get("message") for c in data.get("choices", []) if c.get("message")]
                payload["messages"] = messages + model_msgs + tool_messages
                payload["stream"] = False
                logger.debug("Executing %d tool calls locally", len(tool_calls))
                resp = await app.state.client.post("/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                logger.debug("Ollama response after tools: {}", data)

        formatted = format_chat_response(data)
        logger.debug("Proxy response body: {}", formatted)
        return JSONResponse(formatted)
    except httpx.HTTPError as e:
        logger.error("Error from Ollama: {}", e)
        detail = getattr(e, "response", None)
        if detail is not None:
            try:
                err = detail.json()
            except Exception:
                err = detail.text
            raise HTTPException(status_code=detail.status_code, detail=err)
        raise HTTPException(status_code=500, detail=str(e))
