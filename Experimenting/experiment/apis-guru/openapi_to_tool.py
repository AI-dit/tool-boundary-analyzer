import requests
import json
import os

# Sample Swagger v2 URL (from APIs.guru)
swagger_url = "https://petstore.swagger.io/v2/swagger.json"

# File to save the OpenAI-formatted tool calls
output_file = "petstore_openai_tools.json"

def download_swagger(url):
    print("Downloading Swagger JSON...")
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to download Swagger: {response.status_code}")

def convert_to_openai_tools(swagger_data):
    print("Converting to OpenAI tool format...")
    tools = []
    
    for path, methods in list(swagger_data["paths"].items())[:3]:  # Just take first 3 endpoints
        for method, info in methods.items():
            tool = {
                "type": "function",
                "function": {
                    "name": info.get("operationId", f"{method}_{path.strip('/').replace('/', '_')}"),
                    "description": info.get("description", f"{method.upper()} {path}"),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }

            parameters = info.get("parameters", [])
            for param in parameters:
                param_name = param["name"]
                tool["function"]["parameters"]["properties"][param_name] = {
                    "type": param.get("type", "string"),
                    "description": param.get("description", "")
                }
                if param.get("required", False):
                    tool["function"]["parameters"]["required"].append(param_name)
            
            tools.append(tool)
    
    return tools

def save_to_file(tools, filename):
    print(f"Saving to {filename}...")
    with open(filename, "w") as f:
        json.dump(tools, f, indent=2)

def main():
    swagger_data = download_swagger(swagger_url)
    tools = convert_to_openai_tools(swagger_data)
    save_to_file(tools, output_file)
    print("✅ Done! File saved.")

if __name__ == "__main__":
    main()
