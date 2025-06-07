# Ollama Proxy

This project provides a small FastAPI service that exposes an OpenAI-compatible API on top of [Ollama](https://github.com/jmorganca/ollama). It forwards requests to an Ollama server so that clients written for the OpenAI API can interact with local models.

## Features

- `/v1/models` lists available Ollama models.
- `/v1/chat/completions` proxies chat completion requests using the same payload structure as the OpenAI API.
- Supports tool usage (`tools` and `tool_choice`) when forwarding to Ollama.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run an Ollama server locally or set `OLLAMA_BASE_URL` to point to a remote instance.
3. Start the proxy:
   ```bash
   uvicorn app.main:app --reload
   ```

The service will then be accessible at `http://localhost:8000` and can be used with libraries expecting the OpenAI API. Set `OLLAMA_BASE_URL` to change the upstream Ollama URL.
