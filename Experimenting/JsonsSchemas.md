Openai and antheroic fucntion calling framewords look very similar on the surface and botth relies on Json schmea to desciribe different parameters. but the exact fields names are a bit different..


lets look at the examples first 
openai: "https://platform.openai.com/docs/guides/function-calling?utm_source=chatgpt.com&api-mode=responses"


tools = [{
    "type": "function",
    "name": "get_weather",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and country e.g. Bogotá, Colombia"
            }
        },
        "required": [
            "location"
        ],
        "additionalProperties": False
    }
}]


Noe anthropic : " "tools": [
      {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            }
          },
          "required": ["location"]
        }
      }
    ], "

    Differnece : 
 OpenAI: functions
Anthropic: tools

OpenAI: parameters describes the argument schema.

Anthropic: input_schema describes the argument schema.


OpenAI: use function_call="auto" in the request.

Anthropic: use tool_choice="auto" in the request.

OpenAI: model returns a function_call object on the assistant message with name and serialized arguments fields.

Anthropic: model emits one or more "tool_use" blocks in the assistant content array, each with name and input.

