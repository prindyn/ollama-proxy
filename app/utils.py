"""OpenAI to LangChain message converter.

This module provides functionality to convert OpenAI-style messages
to LangChain message objects.
"""

from typing import List, Dict, Any, Optional, Type
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage


class MessageConverter:
    """Converts OpenAI-style messages to LangChain message objects."""

    # Mapping of OpenAI roles to LangChain message classes
    ROLE_TO_MESSAGE_CLASS: Dict[str, Type[BaseMessage]] = {
        "user": HumanMessage,
        "system": SystemMessage,
        "assistant": AIMessage,
    }

    @classmethod
    def convert_message(cls, message: Dict[str, Any]) -> Optional[BaseMessage]:
        """Convert a single OpenAI message to a LangChain message.

        Args:
            message: OpenAI-style message dictionary with 'role' and 'content' keys.

        Returns:
            A LangChain message object, or None if the role is not supported.

        Raises:
            KeyError: If required keys are missing from the message.
            TypeError: If message is not a dictionary.
        """
        if not isinstance(message, dict):
            raise TypeError(f"Expected dict, got {type(message).__name__}")

        role = message.get("role")
        if role not in cls.ROLE_TO_MESSAGE_CLASS:
            return None

        message_class = cls.ROLE_TO_MESSAGE_CLASS[role]
        content = message.get("content", "")

        return message_class(content=content)

    @classmethod
    def convert_messages(cls, messages: List[Dict[str, Any]]) -> List[BaseMessage]:
        """Convert a list of OpenAI messages to LangChain messages.

        Args:
            messages: List of OpenAI-style message dictionaries.

        Returns:
            List of LangChain message objects, excluding unsupported roles.

        Raises:
            TypeError: If messages is not a list.
        """
        if not isinstance(messages, list):
            raise TypeError(f"Expected list, got {type(messages).__name__}")

        converted_messages = []
        for message in messages:
            converted = cls.convert_message(message)
            if converted is not None:
                converted_messages.append(converted)

        return converted_messages


# Convenience function for backward compatibility
def convert_openai_to_langchain(messages: List[Dict[str, Any]]) -> List[BaseMessage]:
    """Convert OpenAI-style messages to LangChain messages.

    This is a convenience function that uses MessageConverter internally.

    Args:
        messages: List of OpenAI-style message dictionaries with 'role' and 'content'.

    Returns:
        List of LangChain message objects.

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": "Hi there!"}
        ... ]
        >>> langchain_messages = convert_openai_to_langchain(messages)
    """
    return MessageConverter.convert_messages(messages)
