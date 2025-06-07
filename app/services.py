from datetime import datetime
from .providers import get_provider_for_model
from .models import ChatCompletionRequest, ChatCompletionResponse

from loguru import logger


# Example usage in FastAPI endpoint (similar to your example)
async def chat_completion_handler(request: ChatCompletionRequest):
    """Handle chat completion request using appropriate provider."""
    try:
        start_time = datetime.now()

        logger.info(f"Chat completion request for model: {request.model}")
        logger.debug(f"Messages: {len(request.messages)} messages")

        # Get provider for model
        provider = await get_provider_for_model(request.model)
        logger.info(f"Using provider: {provider.__class__.__name__}")

        # Process chat completion
        content = await provider.chat_completion(request.messages, request.tools)

        # Build OpenAI-compatible response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(datetime.now().timestamp())}",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        )

        # Log timing and conversation
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Chat completion finished in {duration:.2f}s")

        return response

    except Exception as e:
        logger.exception("Chat completion failed")
        raise
