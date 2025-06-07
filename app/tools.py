"""OpenAI to LangChain tool converter.

This module provides functionality to convert OpenAI-style tool definitions
into LangChain-compatible StructuredTools.
"""

from typing import List, Dict, Any, Type, Callable, Optional, Set
from pydantic import create_model, Field, BaseModel
from pydantic.config import ConfigDict
from langchain.tools.base import StructuredTool


class OpenAIToolParser:
    """Converts OpenAI-style tool definitions into LangChain-compatible StructuredTools.

    This parser supports function-type tools and maps OpenAI schema definitions
    to Python types for use with Pydantic models.

    Example:
        >>> parser = OpenAIToolParser()
        >>> tools = [{
        ...     "type": "function",
        ...     "function": {
        ...         "name": "get_weather",
        ...         "description": "Get weather data",
        ...         "parameters": {
        ...             "type": "object",
        ...             "properties": {
        ...                 "location": {"type": "string", "description": "City name"}
        ...             },
        ...             "required": ["location"]
        ...         }
        ...     }
        ... }]
        >>> langchain_tools = parser.parse(tools)
    """

    # Type mapping from OpenAI schema types to Python types
    TYPE_MAPPING: Dict[str, Type] = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "object": dict,
        "null": type(None),
    }

    def __init__(self):
        """Initialize the parser with supported tool type handlers."""
        self._tool_parsers: Dict[str, Callable] = {
            "function": self._parse_function_tool,
        }

    def parse(self, tools_json: List[Dict[str, Any]]) -> List[StructuredTool]:
        """Parse a list of OpenAI tool definitions into LangChain StructuredTools.

        Args:
            tools_json: List of OpenAI-style tool definitions.

        Returns:
            List of LangChain StructuredTool instances.

        Raises:
            ValueError: If an unsupported tool type is encountered.
            KeyError: If required fields are missing from tool definitions.
        """
        if not isinstance(tools_json, list):
            raise TypeError(f"Expected list of tools, got {type(tools_json).__name__}")

        parsed_tools = []

        for tool in tools_json:
            tool_type = tool.get("type")

            if tool_type not in self._tool_parsers:
                supported = ", ".join(self._tool_parsers.keys())
                raise ValueError(
                    f"Unsupported tool type: '{tool_type}'. "
                    f"Supported types: {supported}"
                )

            parser = self._tool_parsers[tool_type]
            parsed_tool = parser(tool)
            parsed_tools.append(parsed_tool)

        return parsed_tools

    def _parse_function_tool(self, tool: Dict[str, Any]) -> StructuredTool:
        """Parse a function-type tool definition.

        Args:
            tool: OpenAI function tool definition.

        Returns:
            A StructuredTool instance.

        Raises:
            KeyError: If required fields are missing.
        """
        if "function" not in tool:
            raise KeyError("Function tool must have 'function' field")

        function_def = tool["function"]

        # Extract function metadata
        name = function_def.get("name")
        if not name:
            raise KeyError("Function must have a 'name' field")

        description = function_def.get("description", "")
        parameters = function_def.get("parameters", {})

        # Create Pydantic model from parameters
        input_model = self._create_input_model(name, parameters)

        # Create placeholder function
        def tool_function(**kwargs) -> str:
            """Placeholder function for tool execution."""
            return f"[Tool `{name}` called with arguments: {kwargs}]"

        # Create and return StructuredTool
        return StructuredTool.from_function(
            func=tool_function,
            name=name,
            description=description,
            args_schema=input_model,
            return_direct=True,
        )

    def _create_input_model(
        self, name: str, parameters: Dict[str, Any]
    ) -> Type[BaseModel]:
        """Create a Pydantic model from OpenAI parameter schema.

        Args:
            name: Base name for the model.
            parameters: OpenAI parameters schema.

        Returns:
            A dynamically created Pydantic model class.
        """
        properties = parameters.get("properties", {})
        required_fields: Set[str] = set(parameters.get("required", []))
        allow_extra = parameters.get("additionalProperties", True)

        # Build field definitions
        fields = {}
        for param_name, param_schema in properties.items():
            field_type = self._map_schema_to_type(param_schema)
            field_description = param_schema.get("description", "")

            # Set default based on whether field is required
            default_value = ... if param_name in required_fields else None

            fields[param_name] = (
                field_type,
                Field(default=default_value, description=field_description),
            )

        # Configure model behavior
        model_config = ConfigDict(
            extra="forbid" if not allow_extra else "allow",
            arbitrary_types_allowed=True,
        )

        # Create and return the model
        return create_model(
            f"{name}_Input",
            __base__=BaseModel,
            __config__=model_config,
            **fields,
        )

    def _map_schema_to_type(self, schema: Dict[str, Any]) -> Type:
        """Map OpenAI schema definition to Python type.

        Args:
            schema: OpenAI schema definition for a parameter.

        Returns:
            Corresponding Python type.
        """
        schema_type = schema.get("type", "string")

        # Handle array types
        if schema_type == "array":
            items_schema = schema.get("items", {})
            item_type = self._map_schema_to_type(items_schema)
            return List[item_type]

        # Handle basic types
        return self.TYPE_MAPPING.get(schema_type, str)


# Convenience function for quick parsing
def parse_openai_tools(tools: List[Dict[str, Any]]) -> List[StructuredTool]:
    """Parse OpenAI tool definitions to LangChain StructuredTools.

    This is a convenience function that creates a parser instance internally.

    Args:
        tools: List of OpenAI-style tool definitions.

    Returns:
        List of LangChain StructuredTool instances.
    """
    parser = OpenAIToolParser()
    return parser.parse(tools)
