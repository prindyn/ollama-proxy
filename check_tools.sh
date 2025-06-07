#!/bin/bash

# Model Tool Support Checker for Codex
# Tests if models support function/tool calling and all OpenAI tool types
# Reads from ~/.codex/config.json and ~/.codex.env
# 
# Tests for:
# - Basic function calling support
# - Tool-related fields - tools, tool_choice, etc.
# - Advanced capabilities - multi-tool, streaming
# - OpenAI tool types - web search, file search, code interpreter, etc.

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Default values
PROVIDER=""
MODEL=""
VERBOSE=false
OUTPUT_FILE=""
TEST_ALL=false
CONFIG_FILE="$HOME/.codex/config.json"
ENV_FILE="$HOME/.codex.env"

# Function to print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Function to test OpenAI tool types
test_openai_tools() {
    local model=$1
    local base_request='{
        "model": "'"$model"'",
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 50
    }'
    
    print_color $YELLOW "   Testing various OpenAI tool types:"
    
    # 1. Function calling (already tested above)
    echo "   - Function calling: ✅ (tested above)"
    
    # 2. Web search tool
    local web_search_request='{
        "model": "'"$model"'",
        "messages": [{"role": "user", "content": "Search for recent news"}],
        "tools": [{
            "type": "web_search"
        }],
        "max_tokens": 50
    }'
    
    local response
    response=$(api_call "chat/completions" "$web_search_request" 2>&1)
    if echo "$response" | jq -e '.error' >/dev/null 2>&1; then
        echo "   - Web search: ❌ Not supported"
    else
        echo "   - Web search: ✅ Supported"
    fi
    
    # 3. File search
    local file_search_request='{
        "model": "'"$model"'",
        "messages": [{"role": "user", "content": "Search in files"}],
        "tools": [{
            "type": "file_search"
        }],
        "max_tokens": 50
    }'
    
    response=$(api_call "chat/completions" "$file_search_request" 2>&1)
    if echo "$response" | jq -e '.error' >/dev/null 2>&1; then
        echo "   - File search: ❌ Not supported"
    else
        echo "   - File search: ✅ Supported"
    fi
    
    # 4. Code interpreter
    local code_interpreter_request='{
        "model": "'"$model"'",
        "messages": [{"role": "user", "content": "Run Python code"}],
        "tools": [{
            "type": "code_interpreter"
        }],
        "max_tokens": 50
    }'
    
    response=$(api_call "chat/completions" "$code_interpreter_request" 2>&1)
    if echo "$response" | jq -e '.error' >/dev/null 2>&1; then
        echo "   - Code interpreter: ❌ Not supported"
    else
        echo "   - Code interpreter: ✅ Supported"
    fi
    
    # 5. Image generation (DALL-E)
    print_color $YELLOW "\n   Testing image generation endpoint:"
    local image_gen_request='{
        "model": "dall-e-3",
        "prompt": "A test image",
        "n": 1,
        "size": "1024x1024"
    }'
    
    response=$(api_call "images/generations" "$image_gen_request" 2>&1)
    if echo "$response" | jq -e '.error' >/dev/null 2>&1; then
        local error_msg=$(echo "$response" | jq -r '.error.message // .error' 2>/dev/null)
        if [[ "$error_msg" == *"not found"* ]] || [[ "$error_msg" == *"404"* ]]; then
            echo "   - Image generation: ❌ Endpoint not available"
        else
            echo "   - Image generation: ❌ Not supported (${error_msg:0:50}...)"
        fi
    else
        echo "   - Image generation: ✅ Supported"
    fi
    
    # 6. Computer use (Anthropic-specific)
    local computer_use_request='{
        "model": "'"$model"'",
        "messages": [{"role": "user", "content": "Take a screenshot"}],
        "tools": [{
            "type": "computer_use"
        }],
        "max_tokens": 50
    }'
    
    response=$(api_call "chat/completions" "$computer_use_request" 2>&1)
    if echo "$response" | jq -e '.error' >/dev/null 2>&1; then
        echo "   - Computer use: ❌ Not supported"
    else
        echo "   - Computer use: ✅ Supported"
    fi
    
    # 7. MCP servers
    local mcp_request='{
        "model": "'"$model"'",
        "messages": [{"role": "user", "content": "Use MCP"}],
        "tools": [{
            "type": "mcp_server",
            "server": "test"
        }],
        "max_tokens": 50
    }'
    
    response=$(api_call "chat/completions" "$mcp_request" 2>&1)
    if echo "$response" | jq -e '.error' >/dev/null 2>&1; then
        echo "   - MCP servers: ❌ Not supported"
    else
        echo "   - MCP servers: ✅ Supported"
    fi
    
    # Test assistants API availability
    print_color $YELLOW "\n   Testing Assistants API:"
    response=$(curl -s "$API_BASE/assistants" \
        -H "Authorization: Bearer $API_KEY" \
        -H "OpenAI-Beta: assistants=v2" 2>&1)
    
    if echo "$response" | jq -e '.error' >/dev/null 2>&1; then
        local error_msg=$(echo "$response" | jq -r '.error.message // .error' 2>/dev/null)
        if [[ "$error_msg" == *"not found"* ]] || [[ "$error_msg" == *"404"* ]]; then
            echo "   - Assistants API: ❌ Not available"
        else
            echo "   - Assistants API: ❌ Error (${error_msg:0:50}...)"
        fi
    else
        echo "   - Assistants API: ✅ Available"
        
        # If assistants API is available, test tool availability
        local assistant_request='{
            "model": "'"$model"'",
            "tools": [
                {"type": "code_interpreter"},
                {"type": "file_search"}
            ]
        }'
        
        response=$(curl -s -X POST "$API_BASE/assistants" \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            -H "OpenAI-Beta: assistants=v2" \
            -d "$assistant_request" 2>&1)
            
        if echo "$response" | jq -e '.id' >/dev/null 2>&1; then
            echo "   - Assistant tools: ✅ Code interpreter & File search supported"
            # Clean up - delete the test assistant
            local assistant_id=$(echo "$response" | jq -r '.id')
            curl -s -X DELETE "$API_BASE/assistants/$assistant_id" \
                -H "Authorization: Bearer $API_KEY" \
                -H "OpenAI-Beta: assistants=v2" >/dev/null 2>&1
        fi
    fi
    
    # Test tool_resources parameter (for file uploads with tools)
    local tool_resources_request='{
        "model": "'"$model"'",
        "messages": [{"role": "user", "content": "Analyze this"}],
        "tools": [{"type": "code_interpreter"}],
        "tool_resources": {
            "code_interpreter": {
                "files": []
            }
        },
        "max_tokens": 50
    }'
    
    response=$(api_call "chat/completions" "$tool_resources_request" 2>&1)
    if echo "$response" | jq -e '.error' >/dev/null 2>&1; then
        echo "   - Tool resources: ❌ Not supported"
    else
        echo "   - Tool resources: ✅ Supported"
    fi
}

# Function to print usage
usage() {
    cat << EOF
${BOLD}Model Tool Support Checker for Codex${NC}

${BOLD}USAGE:${NC}
    $0 [OPTIONS] [PROVIDER] [MODEL]

${BOLD}ARGUMENTS:${NC}
    PROVIDER            Provider name - e.g., groq, openai, deepseek
    MODEL               Model name - e.g., llama-3.3-70b-versatile

${BOLD}OPTIONS:${NC}
    -c, --config FILE   Path to config.json (default: ~/.codex/config.json)
    -e, --env FILE      Path to .env file (default: ~/.codex.env)
    -a, --all           Test all models from the provider
    -l, --list          List available providers and exit
    -o, --output FILE   Save results to file (JSON format)
    -v, --verbose       Verbose output (show full API responses)
    -h, --help          Show this help message

${BOLD}EXAMPLES:${NC}
    # Test specific model
    $0 groq llama-3.3-70b-versatile

    # Test default model from config
    $0

    # List available providers
    $0 -l

    # Test all models from a provider
    $0 -a openai

    # Save results with verbose output
    $0 groq llama-3.3-70b-versatile -v -o results.json

EOF
}

# Function to load environment variables from file
load_env_file() {
    if [[ -f "$ENV_FILE" ]]; then
        # Export variables from .env file
        set -a
        source "$ENV_FILE"
        set +a
        print_color $GREEN "Loaded environment from: $ENV_FILE"
    else
        print_color $YELLOW "Warning: Environment file not found: $ENV_FILE"
    fi
}

# Function to get provider config from JSON
get_provider_config() {
    local provider=$1
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_color $RED "Error: Config file not found: $CONFIG_FILE"
        exit 1
    fi
    
    local config
    config=$(jq -r ".providers.\"$provider\"" "$CONFIG_FILE" 2>/dev/null)
    
    if [[ "$config" == "null" ]] || [[ -z "$config" ]]; then
        print_color $RED "Error: Provider '$provider' not found in config"
        exit 1
    fi
    
    echo "$config"
}

# Function to list available providers
list_providers() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_color $RED "Error: Config file not found: $CONFIG_FILE"
        exit 1
    fi
    
    print_color $BLUE "${BOLD}Available Providers:${NC}"
    jq -r '.providers | to_entries[] | "  - \(.key): \(.value.name) (\(.value.baseURL))"' "$CONFIG_FILE"
    
    local default_provider
    default_provider=$(jq -r '.provider // "none"' "$CONFIG_FILE")
    local default_model
    default_model=$(jq -r '.model // "none"' "$CONFIG_FILE")
    
    echo
    print_color $YELLOW "Default provider: $default_provider"
    print_color $YELLOW "Default model: $default_model"
}

# Parse command line arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -e|--env)
            ENV_FILE="$2"
            shift 2
            ;;
        -a|--all)
            TEST_ALL=true
            shift
            ;;
        -l|--list)
            list_providers
            exit 0
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            print_color $RED "Unknown option: $1"
            usage
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Load environment variables
load_env_file

# Process positional arguments
if [[ ${#POSITIONAL_ARGS[@]} -eq 0 ]]; then
    # No arguments - use defaults from config
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_color $RED "Error: No arguments provided and config file not found"
        usage
        exit 1
    fi
    
    PROVIDER=$(jq -r '.provider // empty' "$CONFIG_FILE")
    MODEL=$(jq -r '.model // empty' "$CONFIG_FILE")
    
    if [[ -z "$PROVIDER" ]] || [[ -z "$MODEL" ]]; then
        print_color $RED "Error: No provider/model specified and no defaults in config"
        usage
        exit 1
    fi
    
    print_color $YELLOW "Using defaults from config: provider=$PROVIDER, model=$MODEL"
elif [[ ${#POSITIONAL_ARGS[@]} -eq 1 ]]; then
    if [[ "$TEST_ALL" == true ]]; then
        PROVIDER="${POSITIONAL_ARGS[0]}"
    else
        print_color $RED "Error: Please specify both provider and model, or use -a flag"
        usage
        exit 1
    fi
elif [[ ${#POSITIONAL_ARGS[@]} -eq 2 ]]; then
    PROVIDER="${POSITIONAL_ARGS[0]}"
    MODEL="${POSITIONAL_ARGS[1]}"
else
    print_color $RED "Error: Too many arguments"
    usage
    exit 1
fi

# Get provider configuration
provider_config=$(get_provider_config "$PROVIDER")
API_BASE=$(echo "$provider_config" | jq -r '.baseURL')
ENV_KEY=$(echo "$provider_config" | jq -r '.envKey')
PROVIDER_NAME=$(echo "$provider_config" | jq -r '.name')

# Get API key from environment
API_KEY="${!ENV_KEY:-}"

if [[ -z "$API_KEY" ]]; then
    print_color $RED "Error: API key not found. Expected environment variable: $ENV_KEY"
    print_color $YELLOW "Hint: Add $ENV_KEY to your $ENV_FILE file"
    exit 1
fi

# Remove trailing slash from API_BASE
API_BASE="${API_BASE%/}"

print_color $BLUE "${BOLD}Model Tool Support Checker${NC}"
print_color $BLUE "========================="
print_color $YELLOW "Provider: $PROVIDER_NAME ($PROVIDER)"
print_color $YELLOW "API Base: $API_BASE"
print_color $YELLOW "API Key: ${ENV_KEY}=***${API_KEY: -4}"
echo

# Function to make API call
api_call() {
    local endpoint=$1
    local data=$2
    local method=${3:-POST}
    
    if [[ "$VERBOSE" == true ]]; then
        print_color $YELLOW "Request to $endpoint:"
        echo "$data" | jq . 2>/dev/null || echo "$data"
    fi
    
    local response
    response=$(curl -s -X "$method" \
        "$API_BASE/$endpoint" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$data" 2>&1)
    
    if [[ "$VERBOSE" == true ]]; then
        print_color $YELLOW "Response:"
        echo "$response" | jq . 2>/dev/null || echo "$response"
        echo
    fi
    
    echo "$response"
}

# Function to test basic tool support
test_basic_tool_support() {
    local model=$1
    local test_data='{
        "model": "'"$model"'",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }
            }
        }],
        "tool_choice": "auto",
        "max_tokens": 100
    }'
    
    local response
    response=$(api_call "chat/completions" "$test_data")
    
    # Check if response contains error
    if echo "$response" | jq -e '.error' >/dev/null 2>&1; then
        local error_msg=$(echo "$response" | jq -r '.error.message // .error' 2>/dev/null)
        if [[ "$error_msg" == *"tool"* ]] || [[ "$error_msg" == *"function"* ]]; then
            echo "unsupported"
            return 1
        else
            # Non-tool related error
            echo "error: $error_msg"
            return 1
        fi
    fi
    
    # Check if tool_calls exist in response
    if echo "$response" | jq -e '.choices[0].message.tool_calls' >/dev/null 2>&1; then
        echo "supported"
        return 0
    fi
    
    echo "no_tool_call"
    return 2
}

# Function to test various tool-related fields
test_tool_fields() {
    local model=$1
    local supported_fields=""
    local unsupported_fields=""
    
    # Base request
    local base_request='{
        "model": "'"$model"'",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50
    }'
    
    # Test each field individually (bash 3.2 compatible)
    local field_names=("tools" "tool_choice" "tool_choice_required" "tool_choice_none" "tool_choice_specific" "functions" "function_call" "parallel_tool_calls" "response_format")
    
    # Test tools field
    local test_request=$(echo "$base_request" | jq '. + {"tools": [{"type": "function", "function": {"name": "test", "description": "Test", "parameters": {"type": "object", "properties": {}}}}]}')
    if test_field_support "$test_request" "tools"; then
        supported_fields="$supported_fields tools"
    else
        unsupported_fields="$unsupported_fields tools"
    fi
    
    # Test tool_choice auto
    test_request=$(echo "$base_request" | jq '. + {"tool_choice": "auto"}')
    if test_field_support "$test_request" "tool_choice"; then
        supported_fields="$supported_fields tool_choice"
    else
        unsupported_fields="$unsupported_fields tool_choice"
    fi
    
    # Test tool_choice required
    test_request=$(echo "$base_request" | jq '. + {"tool_choice": "required"}')
    if test_field_support "$test_request" "tool_choice_required"; then
        supported_fields="$supported_fields tool_choice_required"
    else
        unsupported_fields="$unsupported_fields tool_choice_required"
    fi
    
    # Test tool_choice none
    test_request=$(echo "$base_request" | jq '. + {"tool_choice": "none"}')
    if test_field_support "$test_request" "tool_choice_none"; then
        supported_fields="$supported_fields tool_choice_none"
    else
        unsupported_fields="$unsupported_fields tool_choice_none"
    fi
    
    # Test tool_choice specific
    test_request=$(echo "$base_request" | jq '. + {"tool_choice": {"type": "function", "function": {"name": "test"}}}')
    if test_field_support "$test_request" "tool_choice_specific"; then
        supported_fields="$supported_fields tool_choice_specific"
    else
        unsupported_fields="$unsupported_fields tool_choice_specific"
    fi
    
    # Test functions (legacy)
    test_request=$(echo "$base_request" | jq '. + {"functions": [{"name": "test", "description": "Test", "parameters": {"type": "object", "properties": {}}}]}')
    if test_field_support "$test_request" "functions"; then
        supported_fields="$supported_fields functions"
    else
        unsupported_fields="$unsupported_fields functions"
    fi
    
    # Test function_call (legacy)
    test_request=$(echo "$base_request" | jq '. + {"function_call": "auto"}')
    if test_field_support "$test_request" "function_call"; then
        supported_fields="$supported_fields function_call"
    else
        unsupported_fields="$unsupported_fields function_call"
    fi
    
    # Test parallel_tool_calls
    test_request=$(echo "$base_request" | jq '. + {"tools": [{"type": "function", "function": {"name": "test", "description": "Test", "parameters": {"type": "object", "properties": {}}}}], "parallel_tool_calls": true}')
    if test_field_support "$test_request" "parallel_tool_calls"; then
        supported_fields="$supported_fields parallel_tool_calls"
    else
        unsupported_fields="$unsupported_fields parallel_tool_calls"
    fi
    
    # Test response_format
    test_request=$(echo "$base_request" | jq '. + {"response_format": {"type": "json_object"}}')
    if test_field_support "$test_request" "response_format"; then
        supported_fields="$supported_fields response_format"
    else
        unsupported_fields="$unsupported_fields response_format"
    fi
    
    echo "supported:${supported_fields# }"
    echo "unsupported:${unsupported_fields# }"
}

# Helper function to test a single field
test_field_support() {
    local test_request=$1
    local field_name=$2
    
    if [[ "$VERBOSE" == true ]]; then
        print_color $YELLOW "Testing field: $field_name"
    fi
    
    local response
    response=$(api_call "chat/completions" "$test_request" 2>&1)
    
    # Check if field caused an error
    if echo "$response" | jq -e '.error' >/dev/null 2>&1; then
        local error_msg=$(echo "$response" | jq -r '.error.message // .error' 2>/dev/null || echo "Unknown error")
        if [[ "$error_msg" == *"$field_name"* ]] || [[ "$error_msg" == *"not supported"* ]] || [[ "$error_msg" == *"Invalid"* ]]; then
            return 1
        fi
    fi
    return 0
}

# Function to test tool capabilities
test_tool_capabilities() {
    local model=$1
    local capabilities=()
    
    # Test multi-tool support
    local multi_tool_test='{
        "model": "'"$model"'",
        "messages": [{"role": "user", "content": "Get weather and calculate something"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Calculate",
                    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}
                }
            }
        ],
        "max_tokens": 150
    }'
    
    local response
    response=$(api_call "chat/completions" "$multi_tool_test")
    
    if echo "$response" | jq -e '.choices[0].message.tool_calls | length > 1' >/dev/null 2>&1; then
        capabilities+=("multi_tool_calls")
    fi
    
    # Test streaming with tools
    local stream_test=$(echo "$multi_tool_test" | jq '. + {"stream": true}')
    response=$(curl -s -X POST "$API_BASE/chat/completions" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$stream_test" 2>&1 | head -1)
    
    if [[ "$response" == *"data:"* ]]; then
        capabilities+=("streaming_with_tools")
    fi
    
    echo "${capabilities[*]}"
}

# Function to test a single model
test_model() {
    local model=$1
    print_color $GREEN "\nTesting model: ${BOLD}$model${NC}"
    print_color $GREEN "----------------------------------------"
    
    # Test basic tool support
    print_color $BLUE "1. Basic tool support test..."
    local tool_support
    tool_support=$(test_basic_tool_support "$model")
    
    case $tool_support in
        "supported")
            print_color $GREEN "   ✅ Tools/Functions: SUPPORTED"
            ;;
        "unsupported")
            print_color $RED "   ❌ Tools/Functions: NOT SUPPORTED"
            # Skip further tests if basic support is missing
            return
            ;;
        "no_tool_call")
            print_color $YELLOW "   ⚠️  Tools accepted but no tool call made"
            ;;
        error:*)
            print_color $RED "   ❌ Error: ${tool_support#error: }"
            return
            ;;
    esac
    
    # Test supported fields
    print_color $BLUE "\n2. Testing supported fields..."
    local field_results
    field_results=$(test_tool_fields "$model")
    
    local supported_fields=$(echo "$field_results" | grep "^supported:" | cut -d: -f2)
    local unsupported_fields=$(echo "$field_results" | grep "^unsupported:" | cut -d: -f2)
    
    if [[ -n "$supported_fields" ]]; then
        print_color $GREEN "   ✅ Supported fields:"
        for field in $supported_fields; do
            echo "      - $field"
        done
    fi
    
    if [[ -n "$unsupported_fields" ]]; then
        print_color $RED "   ❌ Unsupported fields:"
        for field in $unsupported_fields; do
            echo "      - $field"
        done
    fi
    
    # Test capabilities
    print_color $BLUE "\n3. Testing advanced capabilities..."
    local capabilities
    capabilities=$(test_tool_capabilities "$model")
    
    if [[ -n "$capabilities" ]]; then
        print_color $GREEN "   ✅ Capabilities:"
        for cap in $capabilities; do
            echo "      - $cap"
        done
    else
        print_color $YELLOW "   ⚠️  No advanced capabilities detected"
    fi
    
    # Test OpenAI tool types
    print_color $BLUE "\n4. Testing OpenAI tool types..."
    test_openai_tools "$model"
    
    # Save results if output file specified
    if [[ -n "$OUTPUT_FILE" ]]; then
        # Capture OpenAI tools test results
        local openai_tools_output=$(test_openai_tools "$model" 2>&1 | grep -E "(✅|❌)" | sed 's/^[[:space:]]*//')
        
        local result='{
            "provider": "'"$PROVIDER"'",
            "model": "'"$model"'",
            "tool_support": "'"$tool_support"'",
            "supported_fields": "'"$supported_fields"'",
            "unsupported_fields": "'"$unsupported_fields"'",
            "capabilities": "'"$capabilities"'",
            "openai_tools": "'"$(echo "$openai_tools_output" | tr '\n' ';')"'",
            "tested_at": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"
        }'
        
        if [[ -f "$OUTPUT_FILE" ]]; then
            # Append to existing array
            local existing=$(cat "$OUTPUT_FILE")
            echo "$existing" | jq ". + [$result]" > "$OUTPUT_FILE"
        else
            # Create new array
            echo "[$result]" | jq . > "$OUTPUT_FILE"
        fi
    fi
}

# Main execution
if [[ "$TEST_ALL" == true ]]; then
    print_color $BLUE "Fetching available models from $PROVIDER_NAME..."
    
    # Get list of models
    models_response=$(curl -s "$API_BASE/models" -H "Authorization: Bearer $API_KEY")
    
    if echo "$models_response" | jq -e '.error' >/dev/null 2>&1; then
        print_color $RED "Error fetching models: $(echo "$models_response" | jq -r '.error.message')"
        exit 1
    fi
    
    # Extract model IDs
    models=$(echo "$models_response" | jq -r '.data[].id' 2>/dev/null)
    
    if [[ -z "$models" ]]; then
        print_color $RED "No models found"
        exit 1
    fi
    
    print_color $GREEN "Found $(echo "$models" | wc -l) models"
    
    # Initialize output file with empty array if specified
    if [[ -n "$OUTPUT_FILE" ]]; then
        echo "[]" > "$OUTPUT_FILE"
    fi
    
    # Test each model
    for model in $models; do
        test_model "$model"
    done
    
    if [[ -n "$OUTPUT_FILE" ]]; then
        print_color $GREEN "\nResults saved to: $OUTPUT_FILE"
    fi
else
    # Test single model
    test_model "$MODEL"
fi

print_color $GREEN "\n${BOLD}Testing complete!${NC}"