"""
Microwave Memory Agent - Learning the fundamentals
Lesson 4b: Configurable System Prompt

Instead of hardcoding the system prompt, we:
1. Load it from files (SOUL.md, AGENTS.md, etc.)
2. Inject dynamic context at startup
3. Make behavior configurable without changing code

This is how OpenClaw works — the prompt is assembled from workspace files.
"""

import os
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

MEMORY_DIR = "./memory"
CONFIG_DIR = "./config"
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


def load_file_if_exists(path: str) -> str:
    """Load a file's contents, or return empty string if missing."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return ""


def build_system_prompt() -> str:
    """
    Assemble the system prompt from multiple files.
    
    This is the key insight: the prompt is CONFIGURED, not hardcoded.
    You can change agent behavior by editing markdown files.
    """
    
    # Core identity
    soul = load_file_if_exists("./config/SOUL.md")
    
    # Behavioral patterns
    agents = load_file_if_exists("./config/AGENTS.md")
    
    # User-specific context
    user = load_file_if_exists("./config/USER.md")
    
    # Tools documentation
    tools_doc = load_file_if_exists("./config/TOOLS.md")
    
    # Dynamic context (injected at runtime)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Assemble the prompt
    prompt_parts = []
    
    if soul:
        prompt_parts.append(f"# WHO YOU ARE\n{soul}")
    else:
        prompt_parts.append("# WHO YOU ARE\nYou are Microwave, a helpful assistant.")
    
    if agents:
        prompt_parts.append(f"# HOW TO BEHAVE\n{agents}")
    
    if user:
        prompt_parts.append(f"# ABOUT THE USER\n{user}")
    
    if tools_doc:
        prompt_parts.append(f"# TOOLS REFERENCE\n{tools_doc}")
    
    # Always include runtime context
    prompt_parts.append(f"""# RUNTIME CONTEXT
- Current time: {current_time}
- Memory directory: {MEMORY_DIR}/
- Config directory: {CONFIG_DIR}/

## MEMORY SYSTEM
You have access to persistent memory via tools:
- memory_write: Save important information
- memory_search: Search before answering recall questions
- memory_read: Read specific memory files

**Rule: Check memory before guessing about past conversations or user info.**
""")
    
    return "\n\n---\n\n".join(prompt_parts)


# --- TOOLS (same as before) ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Write to long-term memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search memory - USE THIS before answering questions about past conversations or user info",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read a specific memory file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"}
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read any file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write to any file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current date and time",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]


def memory_write(filename: str, content: str) -> str:
    filepath = os.path.join(MEMORY_DIR, f"{filename}.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(filepath, 'a') as f:
        f.write(f"\n## [{timestamp}]\n{content}\n")
    return f"✓ Saved to {filename}.md"

def memory_search(query: str) -> str:
    results = []
    query_lower = query.lower()
    if os.path.exists(MEMORY_DIR):
        for filename in os.listdir(MEMORY_DIR):
            if filename.endswith('.md'):
                with open(os.path.join(MEMORY_DIR, filename), 'r') as f:
                    content = f.read()
                    if query_lower in content.lower():
                        results.append(f"=== {filename} ===\n{content}")
    return "\n\n".join(results) if results else f"No memories matching '{query}'"

def memory_read(filename: str) -> str:
    if not filename.endswith('.md'):
        filename = f"{filename}.md"
    try:
        with open(os.path.join(MEMORY_DIR, filename), 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Not found: {filename}"

def read_file(path: str) -> str:
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

def write_file(path: str, content: str) -> str:
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"✓ Wrote to {path}"
    except Exception as e:
        return f"Error: {e}"

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOL_FUNCTIONS = {
    "memory_write": memory_write,
    "memory_search": memory_search,
    "memory_read": memory_read,
    "read_file": read_file,
    "write_file": write_file,
    "get_current_time": get_current_time,
}

def execute_tool(name: str, args: dict) -> str:
    if name not in TOOL_FUNCTIONS:
        return f"Unknown tool: {name}"
    try:
        return TOOL_FUNCTIONS[name](**args)
    except Exception as e:
        return f"Error: {e}"


def main():
    # Build system prompt from config files
    system_prompt = build_system_prompt()
    
    print("=" * 50)
    print("SYSTEM PROMPT (assembled from config files):")
    print("=" * 50)
    print(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
    print("=" * 50)
    print()
    
    messages = [{"role": "system", "content": system_prompt}]
    
    print("Microwave Agent (configurable)")
    print("Edit files in ./config/ to change behavior!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        if not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        if len(messages) > 21:
            messages = [messages[0]] + messages[-20:]
        
        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            
            msg = response.choices[0].message
            
            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]
                })
                
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    print(f"  [{name}]")
                    result = execute_tool(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })
                continue
            else:
                messages.append({"role": "assistant", "content": msg.content})
                print(f"\nMicrowave: {msg.content}\n")
                break


if __name__ == "__main__":
    main()
