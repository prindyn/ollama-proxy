"""Service layer for handling chat completions."""

from datetime import datetime
from loguru import logger

from .providers import get_provider_for_model
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    Usage,
)


async def chat_completion_handler(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Handle chat completion request using appropriate provider."""
    try:
        start_time = datetime.now()

        logger.info(f"Chat completion request for model: {request.model}")
        logger.debug(f"Messages: {len(request.messages)} messages")
        if request.tools:
            logger.debug(f"Tools: {len(request.tools)} tools provided")

        # Get provider for model
        provider = await get_provider_for_model(request.model)
        logger.info(f"Using provider: {provider.__class__.__name__}")

        # Convert messages to dict format
        messages = []
        for msg in request.messages:
            msg_dict = {"role": msg.role}
            if msg.content is not None:
                msg_dict["content"] = msg.content
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            messages.append(msg_dict)

        # Process chat completion
        response_data = await provider.chat_completion(
            messages=messages,
            tools=request.tools,
            tool_choice=request.tool_choice,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            user=request.user,
        )

        # Convert response to our model format
        choices = []
        for choice_data in response_data.get("choices", []):
            msg_data = choice_data.get("message", {})
            message = Message(
                role=msg_data.get("role", "assistant"),
                content=msg_data.get("content"),
                tool_calls=msg_data.get("tool_calls"),
            )

            choice = Choice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason"),
                logprobs=choice_data.get("logprobs"),
            )
            choices.append(choice)

        # Build response
        usage_data = response_data.get("usage", {})
        usage = (
            Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
            if usage_data
            else None
        )

        response = ChatCompletionResponse(
            id=response_data.get("id", f"chatcmpl-{int(datetime.now().timestamp())}"),
            created=response_data.get("created", int(datetime.now().timestamp())),
            model=response_data.get("model", request.model),
            choices=choices,
            usage=usage,
            system_fingerprint=response_data.get("system_fingerprint"),
            service_tier=response_data.get("service_tier", "default"),
        )

        # Log timing
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Chat completion finished in {duration:.2f}s")

        return response

    except Exception as e:
        logger.exception("Chat completion failed")
        raise
