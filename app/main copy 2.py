from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx, os
from datetime import datetime
from loguru import logger
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
from .logger import configure_logger, log_conversation, LoggingCallbackHandler
from .tools import OpenAIToolParser

configure_logger()
logger.add("logs/langchain.log", rotation="1 MB", retention="10 days")

app = FastAPI(
    title="Ollama Proxy", description="OpenAI compatible API for Ollama models"
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOG_DIR = os.getenv("LOG_DIR", "logs")


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up proxy, connecting to {}", OLLAMA_BASE_URL)
    app.state.client = httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=None)
    app.state.responses = {}
    os.makedirs(LOG_DIR, exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down proxy")
    await app.state.client.aclose()


@app.get("/")
async def root():
    return {"message": "Ollama proxy running"}


@app.get("/v1/models")
async def list_models():
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
    body = await request.json()
    model = body.get("model")
    messages = body.get("messages")
    raw_tools = body.get("tools", [])

    if not model or not messages:
        raise HTTPException(status_code=400, detail="model and messages required")

    try:
        # Setup LLM and tools
        llm = ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            callbacks=[LoggingCallbackHandler()],
        )
        tools = OpenAIToolParser().parse(raw_tools)

        # Create agent and invoke
        agent = create_react_agent(model=llm, tools=tools)
        result = await agent.ainvoke({"messages": messages})

        # Extract assistant's last message
        assistant_message = None
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage):
                assistant_message = msg
                break

        if not assistant_message:
            raise ValueError("No assistant response found")

        # Build response
        response = {
            "id": f"chatcmpl-{int(datetime.now().timestamp())}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_message.content,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        log_conversation(request.url.path, body, response)
        return JSONResponse(response)

    except Exception as e:
        logger.exception("Error in chat completion")
        raise HTTPException(status_code=500, detail=str(e))
