from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from datetime import datetime
from loguru import logger
import httpx, os

from langchain.agents import create_react_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from .logger import configure_logger, log_conversation, LoggingCallbackHandler
from .utils import convert_openai_to_langchain
from .tools import OpenAIToolParser

# --- Configuration ---
configure_logger()
logger.add("logs/langchain.log", rotation="1 MB", retention="10 days")

LOG_DIR = os.getenv("LOG_DIR", "logs")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

# --- FastAPI App ---
app = FastAPI(
    title="GPT-4 Tool Proxy",
    description="OpenAI-compatible API using GPT-4 with tools",
)


# --- Startup & Shutdown ---
@app.on_event("startup")
async def startup_event():
    logger.info("Starting GPT-4 tool proxy")
    os.makedirs(LOG_DIR, exist_ok=True)
    app.state.client = httpx.AsyncClient(timeout=None)
    app.state.responses = {}


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down GPT-4 tool proxy")
    await app.state.client.aclose()


# --- Routes ---
@app.get("/")
async def root():
    return {"message": "GPT-4 tool proxy running"}


@app.get("/v1/responses")
async def list_responses():
    return {"object": "list", "data": list(app.state.responses.values())}


@app.get("/v1/responses/{resp_id}")
async def get_response(resp_id: str):
    resp = app.state.responses.get(resp_id)
    if not resp:
        raise HTTPException(status_code=404, detail="Response not found")
    return JSONResponse(resp)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages")
    raw_tools = body.get("tools", [])
    model_name = body.get("model", OPENAI_MODEL)

    if not messages:
        raise HTTPException(status_code=400, detail="`messages` is required")

    try:
        # Parse tools
        tools = OpenAIToolParser().parse(raw_tools)
        tool_names = [tool.name for tool in tools]
        tool_descriptions = "\n".join(
            f"{tool.name}: {tool.description}" for tool in tools
        )

        # Initialize LLM
        llm = ChatOpenAI(
            model=model_name,
            api_key=OPENAI_API_KEY,
            callbacks=[LoggingCallbackHandler()],
        )

        # Prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("messages"),
            ]
        )
        prompt.input_variables = ["messages", "agent_scratchpad", "tools", "tool_names"]

        # Create ReAct Agent and Executor
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

        # FIX: Pass tools to AgentExecutor
        executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True  # <-- This was missing!
        )

        # Invoke agent
        langchain_messages = convert_openai_to_langchain(messages)
        result = await executor.ainvoke(
            {
                "messages": langchain_messages,
                "agent_scratchpad": [],
                "tools": tool_descriptions,
                "tool_names": tool_names,
            }
        )

        # Extract response content
        content = result.content if isinstance(result, AIMessage) else str(result)
        response = {
            "id": f"chatcmpl-{int(datetime.now().timestamp())}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

        app.state.responses[response["id"]] = response
        log_conversation(request.url.path, body, response)
        return JSONResponse(response)

    except httpx.HTTPError as e:
        logger.error("HTTP error from OpenAI: {}", e)
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.exception("Unhandled server error")
        raise HTTPException(status_code=500, detail=str(e))
