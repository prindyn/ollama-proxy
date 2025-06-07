import json
import time
import uuid
from typing import AsyncIterator


def format_chat_response(data: dict) -> dict:
    """Convert Ollama chat response to OpenAI format."""
    res = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model"),
        "choices": [
            {
                "index": 0,
                "message": data.get("message"),
                "finish_reason": data.get("done_reason", "stop"),
            }
        ],
    }
    prompt_tokens = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    if prompt_tokens is not None and completion_tokens is not None:
        res["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    return res


async def stream_chat_response(resp) -> AsyncIterator[str]:
    """Yield OpenAI-style SSE events from an Ollama streaming response."""
    resp_id = f"chatcmpl-{uuid.uuid4().hex}"
    prev = ""
    first = True
    async for line in resp.aiter_lines():
        if not line:
            continue
        data = json.loads(line)
        content = data.get("message", {}).get("content", "")
        diff = content[len(prev):]
        prev = content
        chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": data.get("model"),
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": None,
                }
            ],
        }
        delta = chunk["choices"][0]["delta"]
        if first:
            delta["role"] = "assistant"
            first = False
        if diff:
            delta["content"] = diff
        if data.get("done"):
            chunk["choices"][0]["finish_reason"] = data.get("done_reason", "stop")
            yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
            break
        else:
            yield f"data: {json.dumps(chunk)}\n\n"

