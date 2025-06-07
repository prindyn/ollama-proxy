from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import os
from loguru import logger
from .logger import configure_logger, log_conversation
from .openai_adapter import format_chat_response, stream_chat_response

configure_logger()

app = FastAPI(title="Ollama Proxy", description="OpenAI compatible API for Ollama models")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")



@app.on_event("startup")
async def startup_event():
    logger.info("Starting up proxy, connecting to {url}", url=OLLAMA_BASE_URL)
    app.state.client = httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=None)
    app.state.responses = {}
    app.state.log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(app.state.log_dir, exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down proxy")
    await app.state.client.aclose()

@app.get("/")
async def root():
    return {"message": "Ollama proxy running"}


@app.get("/v1/responses")
async def list_responses():
    """Return stored responses."""
    return {"object": "list", "data": list(app.state.responses.values())}


@app.get("/v1/responses/{resp_id}")
async def get_response(resp_id: str):
    resp = app.state.responses.get(resp_id)
    if resp is None:
        raise HTTPException(status_code=404, detail="Response not found")
    return JSONResponse(resp)

@app.get("/v1/models")
async def list_models():
    """Return available models from Ollama"""
    try:
        resp = await app.state.client.get("/api/tags")
        resp.raise_for_status()
        data = resp.json()
        models = [{"id": m["name"], "object": "model"} for m in data.get("models", [])]
        return {"object": "list", "data": models}
    except Exception as e:
        logger.error("Error fetching models: {}", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Proxy OpenAI chat completions to Ollama"""
    body = await request.json()
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
        logger.info("Forwarding chat completion to Ollama: model=%s stream=%s", model, stream)
        resp = await app.state.client.post("/api/chat", json=payload)
        resp.raise_for_status()
        if stream:
            def record_response(res):
                app.state.responses[res["id"]] = res
                log_conversation(request.url.path, body, res)

            async def iterator():
                async for chunk in stream_chat_response(resp, on_complete=record_response):
                    yield chunk
            return StreamingResponse(iterator(), media_type="text/event-stream")

        data = resp.json()

        formatted = format_chat_response(data)
        app.state.responses[formatted["id"]] = formatted
        log_conversation(request.url.path, body, formatted)
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
