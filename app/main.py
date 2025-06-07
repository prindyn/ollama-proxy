from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import os
from loguru import logger

app = FastAPI(title="Ollama Proxy", description="OpenAI compatible API for Ollama models")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


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
        logger.debug("Forwarding chat completion to Ollama: model=%s stream=%s", model, stream)
        resp = await app.state.client.post("/api/chat", json=payload)
        resp.raise_for_status()
        if stream:
            async def iterator():
                async for chunk in resp.aiter_text():
                    yield chunk
            return StreamingResponse(iterator(), media_type="text/event-stream")
        return JSONResponse(resp.json())
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
