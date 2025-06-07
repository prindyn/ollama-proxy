# Ollama Proxy

This project provides a small FastAPI service that exposes an OpenAI-compatible API on top of [Ollama](https://github.com/jmorganca/ollama). It forwards requests to an Ollama server so that clients written for the OpenAI API can interact with local models.

## Features

- `/v1/models` lists available Ollama models.
- `/v1/chat/completions` generates completions using LangChain with an Ollama model.
- Built-in tools (shell, file operations, web search) are available to the agent.
- `/v1/responses` lists previous chat completions.
- Chat requests and responses are saved to NDJSON files under `logs/`. The file
  name matches the last part of the request path, e.g. `completions.ndjson`. Each
  request and response (including errors) is written as a separate line so the
  file can be processed as an NDJSON log.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run an Ollama server locally or set `OLLAMA_BASE_URL` to point to a remote instance.
   Environment variables such as `OLLAMA_BASE_URL` and `LOGURU_LEVEL` can be placed in a `.env` file which is loaded automatically.
3. Start the proxy:
   ```bash
   uvicorn app.main:app --reload
   ```

The service will then be accessible at `http://localhost:8000` and can be used with libraries expecting the OpenAI API. Requests and responses follow the same schema as OpenAI's endpoints. Set `OLLAMA_BASE_URL` to change the upstream Ollama URL. Logging is handled with [Loguru](https://github.com/Delgan/loguru). The logger configuration lives in `app/logger.py` and respects the `LOGURU_LEVEL`, `LOG_FILE`, and `LOG_DIR` environment variables.
The agent uses LangChain to execute simple tools like `shell` or file operations during generation.

