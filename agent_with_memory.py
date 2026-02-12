"""
Microwave Memory Agent - Learning the fundamentals
Lesson 3: Memory

The problem: messages array grows forever, context windows don't.

Solutions we'll explore:
1. Truncation (simple, lossy) 
2. Summarization (compress old context)
3. External memory (write to files, retrieve when needed)
4. The "check notes before guessing" pattern

This lesson focuses on #3 and #4 — the behavioral approach to memory.
"""

import os
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# --- MEMORY DIRECTORY ---
MEMORY_DIR = "./memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

# --- THE KEY INSIGHT ---
# Instead of keeping everything in the messages array,
# we teach the agent to WRITE things down and READ them back.
# Memory becomes a behavior, not just a data structure.

SYSTEM_PROMPT = """You are a helpful assistant named Microwave.
You are direct, slightly sarcastic, and efficient.

## MEMORY INSTRUCTIONS (IMPORTANT)

You have access to a memory system. Use it.

**BEFORE answering questions about past conversations or facts you should remember:**
- Use `memory_search` to check your notes FIRST
- Don't guess or make things up
- If you don't find it, say so

**WHEN you learn something important:**
- Use `memory_write` to save it
- Include: what you learned, context, date
- Be specific enough that future-you can understand it

**WHAT to remember:**
- User preferences and facts about them
- Decisions made and why
- Tasks completed
- Anything the user says "remember this"

Your memory files are in ./memory/ — they persist across sessions.
The messages array does NOT persist. Files do.

Treat memory as a behavior: actively write, actively check."""

# --- TOOLS ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Write something to long-term memory. Use for facts, preferences, decisions, or anything worth remembering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name for the memory file (e.g., 'user_preferences', 'project_notes', 'decisions')"
                    },
                    "content": {
                        "type": "string", 
                        "description": "What to remember. Be specific and include context."
                    },
                    "append": {
                        "type": "boolean",
                        "description": "If true, append to existing file. If false, overwrite. Default: true"
                    }
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search your memory for information. USE THIS BEFORE answering questions about past conversations or things you should remember.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_list",
            "description": "List all memory files available",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Read a specific memory file in full",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the memory file to read"
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# --- TOOL IMPLEMENTATIONS ---

def memory_write(filename: str, content: str, append: bool = True) -> str:
    """Write to a memory file."""
    filepath = os.path.join(MEMORY_DIR, f"{filename}.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    entry = f"\n## [{timestamp}]\n{content}\n"
    
    mode = 'a' if append else 'w'
    try:
        with open(filepath, mode) as f:
            if not append:
                f.write(f"# {filename}\n")
            f.write(entry)
        return f"✓ Saved to memory: {filename}.md"
    except Exception as e:
        return f"Error writing memory: {e}"

def memory_search(query: str) -> str:
    """Search across all memory files for relevant content."""
    results = []
    query_lower = query.lower()
    
    if not os.path.exists(MEMORY_DIR):
        return "No memories found. Memory directory is empty."
    
    for filename in os.listdir(MEMORY_DIR):
        if filename.endswith('.md'):
            filepath = os.path.join(MEMORY_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    if query_lower in content.lower():
                        # Return the whole file if it matches (simple approach)
                        results.append(f"=== {filename} ===\n{content}")
            except Exception as e:
                continue
    
    if results:
        return "\n\n".join(results)
    else:
        return f"No memories found matching '{query}'"

def memory_list() -> str:
    """List all memory files."""
    if not os.path.exists(MEMORY_DIR):
        return "No memory files yet."
    
    files = [f for f in os.listdir(MEMORY_DIR) if f.endswith('.md')]
    if files:
        return "Memory files:\n" + "\n".join(f"- {f}" for f in files)
    else:
        return "No memory files yet."

def memory_read(filename: str) -> str:
    """Read a specific memory file."""
    # Add .md if not present
    if not filename.endswith('.md'):
        filename = f"{filename}.md"
    
    filepath = os.path.join(MEMORY_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Memory file not found: {filename}"
    except Exception as e:
        return f"Error reading memory: {e}"

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

TOOL_FUNCTIONS = {
    "memory_write": memory_write,
    "memory_search": memory_search,
    "memory_list": memory_list,
    "memory_read": memory_read,
    "get_current_time": get_current_time,
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


# --- CONTEXT MANAGEMENT ---
# Keep only the last N messages to avoid context overflow
# Older context should be in memory files, not the messages array

MAX_MESSAGES = 20  # Keep last 20 messages (10 turns)

def trim_messages(messages: list) -> list:
    """Keep system prompt + last N messages."""
    if len(messages) <= MAX_MESSAGES + 1:  # +1 for system prompt
        return messages
    
    # Always keep system prompt (first message)
    system = messages[0]
    recent = messages[-(MAX_MESSAGES):]
    
    return [system] + recent


# --- THE AGENT LOOP ---

def main():
    """Agent loop with memory."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    print("Microwave Memory Agent (with persistent memory)")
    print(f"Memory directory: {MEMORY_DIR}/")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        # Trim messages to avoid context overflow
        messages = trim_messages(messages)
        
        # Inner loop for tool execution
        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            
            assistant_message = response.choices[0].message
            
            if assistant_message.tool_calls:
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
                
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    print(f"  [Memory: {tool_name}({list(arguments.keys())})]")
                    
                    result = execute_tool(tool_name, arguments)
                    
                    # Only show truncated result for reads/searches
                    if tool_name in ['memory_read', 'memory_search']:
                        display = result[:200] + '...' if len(result) > 200 else result
                        print(f"  [Found: {display}]")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                
                continue
            
            else:
                messages.append({"role": "assistant", "content": assistant_message.content})
                print(f"\nMicrowave: {assistant_message.content}\n")
                break


if __name__ == "__main__":
    main()
