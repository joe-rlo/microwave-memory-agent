"""
Microwave Memory Agent - Learning the fundamentals
Lesson 2: Tools

An agent that can DO things, not just talk.

Key concepts:
1. Tools are defined as JSON schemas (name, description, parameters)
2. The model decides when to use a tool (returns tool_call instead of text)
3. We execute the tool and feed results back
4. Loop continues until model responds with text

The model doesn't run the tools — WE do. The model just decides what to call.
"""

import os
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a helpful assistant named Microwave.
You are direct, slightly sarcastic, and efficient.
You have access to tools — use them when they'd help answer the question.
Don't make up information. If you need to know something, use a tool."""

# --- TOOL DEFINITIONS ---
# This is what we tell the model is available
# The model reads these descriptions to decide when to use each tool

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {},  # no parameters needed
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates or overwrites)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and folders in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list (default: current directory)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate (e.g., '2 + 2', 'sqrt(16)', '10 * 5')"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# --- TOOL IMPLEMENTATIONS ---
# This is the actual code that runs when a tool is called
# The model never sees this code — it only sees the schemas above

def get_current_time() -> str:
    """Return current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_file(path: str) -> str:
    """Read a file and return its contents."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

def list_directory(path: str = ".") -> str:
    """List contents of a directory."""
    try:
        items = os.listdir(path)
        return "\n".join(items) if items else "(empty directory)"
    except FileNotFoundError:
        return f"Error: Directory not found: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"

def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    import math
    try:
        # Only allow safe math operations
        allowed_names = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow,
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
            'tan': math.tan, 'log': math.log, 'log10': math.log10,
            'pi': math.pi, 'e': math.e
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# Map tool names to functions
TOOL_FUNCTIONS = {
    "get_current_time": get_current_time,
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "calculator": calculator,
}


def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool and return the result."""
    if tool_name not in TOOL_FUNCTIONS:
        return f"Error: Unknown tool '{tool_name}'"
    
    func = TOOL_FUNCTIONS[tool_name]
    try:
        result = func(**arguments)
        return result
    except Exception as e:
        return f"Error executing {tool_name}: {e}"


# --- THE AGENT LOOP ---

def main():
    """Agent loop with tool support."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    print("Microwave Memory Agent (with tools)")
    print("Tools available: get_current_time, read_file, write_file, list_directory, calculator")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        # Inner loop: keep going until we get a text response (not a tool call)
        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,  # <-- This is new: we pass the tool definitions
                tool_choice="auto",  # Let the model decide when to use tools
            )
            
            assistant_message = response.choices[0].message
            
            # Check if the model wants to call a tool
            if assistant_message.tool_calls:
                # Add the assistant's message (with tool calls) to history
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    print(f"  [Tool: {tool_name}({arguments})]")
                    
                    # Execute the tool
                    result = execute_tool(tool_name, arguments)
                    
                    print(f"  [Result: {result[:100]}{'...' if len(result) > 100 else ''}]")
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                
                # Loop back to get the model's response to the tool results
                continue
            
            else:
                # No tool calls — we have a final text response
                messages.append({"role": "assistant", "content": assistant_message.content})
                print(f"\nMicrowave: {assistant_message.content}\n")
                break  # Exit inner loop, wait for next user input


if __name__ == "__main__":
    main()
