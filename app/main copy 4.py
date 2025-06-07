from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from datetime import datetime
from loguru import logger
import httpx, os

from langchain.agents import create_react_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub
from langchain.prompts import load_prompt

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

        # Option 1: Use the default ReAct prompt from LangChain Hub
        try:
            prompt = hub.pull("hwchase17/react-chat")
        except Exception as e:
            logger.warning(f"Could not pull prompt from hub: {e}. Using local default.")
            # Option 2: Fallback to a local default if hub is unavailable
            # Get the default prompt from the create_react_agent function
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", load_prompt("prompts/system_tools.txt")),
                    MessagesPlaceholder(variable_name="chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

        # Create ReAct Agent and Executor
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

        # FIX: Pass tools to AgentExecutor with parsing error handling
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,  # Handle parsing errors gracefully
            max_iterations=3,  # Limit iterations to prevent infinite loops
        )

        # Invoke agent
        langchain_messages = convert_openai_to_langchain(messages)

        # Prepare the input - ReAct expects 'input' and optionally 'chat_history'
        agent_input = {
            "input": langchain_messages[-1].content if langchain_messages else "",
            "chat_history": (
                langchain_messages[:-1] if len(langchain_messages) > 1 else []
            ),
        }

        result = await executor.ainvoke(agent_input)

        # Extract response content - handle different result types
        if isinstance(result, dict) and "output" in result:
            content = result["output"]
        elif isinstance(result, AIMessage):
            content = result.content
        else:
            content = str(result)
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
