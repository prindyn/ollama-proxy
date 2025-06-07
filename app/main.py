from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
import os
from datetime import datetime
from loguru import logger
from langchain.llms import Ollama
from langchain.agents import initialize_agent, AgentType
from .logger import configure_logger, log_conversation
from .tools import get_tools

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
    """Generate a chat completion using LangChain tools."""
    body = await request.json()
    model = body.get("model")
    messages = body.get("messages")
    if not model or not messages:
        raise HTTPException(status_code=400, detail="model and messages required")

    user_input = next((m.get("content") for m in reversed(messages) if m.get("role") == "user"), None)
    if not user_input:
        raise HTTPException(status_code=400, detail="No user input found")

    try:
        llm = Ollama(model=model, base_url=OLLAMA_BASE_URL)
        agent = initialize_agent(get_tools(), llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
        answer = agent.invoke({"input": user_input})

        res = {
            "id": f"chatcmpl-{int(datetime.now().timestamp())}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
        }

        app.state.responses[res["id"]] = res
        log_conversation(request.url.path, body, res)
        return JSONResponse(res)
    except httpx.HTTPError as e:
        logger.error("Error from Ollama: {}", e)
        detail = getattr(e, "response", None)
        if detail is not None:
            try:
                err = detail.json()
            except Exception:
                err = {"error": detail.text}
            log_conversation(request.url.path, body, err)
            raise HTTPException(status_code=detail.status_code, detail=err)
        log_conversation(request.url.path, body, {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("Unhandled error: {}", e)
        log_conversation(request.url.path, body, {"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))
