"""
Microwave Memory Agent - Learning the fundamentals
Lesson 4: Behavior Patterns

We have tools. We have memory. Now: how do we make the agent behave WELL?

Key patterns:
1. Pre-game routine - structured thinking BEFORE acting
2. Checkpointing - save state so you can recover
3. Reflection - review what happened, learn from it
4. Mode separation - don't mix planning and execution

These are cognitive architecture patterns, not code patterns.
The implementation is mostly in the system prompt.
"""

import os
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

MEMORY_DIR = "./memory"
CHECKPOINT_FILE = "./memory/checkpoint.json"
os.makedirs(MEMORY_DIR, exist_ok=True)


# --- THE BEHAVIORAL SYSTEM PROMPT ---
# This is where the magic happens. Not in code — in instructions.

SYSTEM_PROMPT = """You are a helpful assistant named Microwave.
You are direct, efficient, and thoughtful about HOW you work, not just WHAT you do.

## CORE BEHAVIOR PATTERNS

### 1. PRE-GAME ROUTINE (Before any non-trivial task)

Before executing a task that involves multiple steps or tool use, PAUSE and think:

```
TASK: [restate what you're being asked to do in one sentence]
CONSTRAINTS: [what are the limits or requirements?]
SUCCESS LOOKS LIKE: [how will you know you're done?]
APPROACH: [brief plan - what steps, what tools?]
```

Write this out explicitly. Don't skip it. This prevents drift and wasted effort.

### 2. CHECK BEFORE YOU GUESS

Before answering questions about:
- Past conversations
- User preferences
- Previous decisions
- Anything you "should" remember

ALWAYS use memory_search FIRST. Don't guess. Don't make things up.
If you don't find it, say "I don't have that in my notes."

### 3. CHECKPOINT ON PROGRESS

When you complete a meaningful piece of work:
- Use checkpoint_save to record what was done
- Include: what you did, what's next, any blockers

This lets you (or future-you) pick up where you left off.

### 4. REFLECT ON COMPLETION

After finishing a multi-step task, briefly note:
- What worked well
- What was harder than expected
- What you'd do differently

Save this to memory for future reference.

### 5. DON'T MIX MODES

- PLANNING MODE: Figure out what to do (no tool calls yet)
- EXECUTION MODE: Do the thing (tool calls, actions)
- REVIEW MODE: Evaluate what happened

Don't plan mid-execution. Don't execute mid-planning.
Finish one mode before starting another.

## MEMORY

You have persistent memory in ./memory/. Use it actively:
- memory_write: Save important info
- memory_search: Find relevant info before answering
- memory_read: Get full contents of a memory file

## CHECKPOINTS

You can save and load checkpoints:
- checkpoint_save: Save current task state
- checkpoint_load: See what was in progress

Use these when working on multi-step tasks.

Remember: You're not just an assistant that responds.
You're an agent that WORKS — thoughtfully, reliably, and with awareness of your own process.
"""


# --- TOOLS ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Write to long-term memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Category name"},
                    "content": {"type": "string", "description": "What to save"}
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search memory. USE THIS before answering questions about past conversations or user info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
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
            "name": "checkpoint_save",
            "description": "Save current task state. Use when making progress on multi-step work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What task is in progress"},
                    "status": {"type": "string", "description": "Current status"},
                    "completed": {"type": "array", "items": {"type": "string"}, "description": "Steps completed"},
                    "next_steps": {"type": "array", "items": {"type": "string"}, "description": "What's next"},
                    "blockers": {"type": "string", "description": "Any blockers or issues"}
                },
                "required": ["task", "status"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "checkpoint_load",
            "description": "Load the last checkpoint to see what was in progress",
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
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: current)"}
                },
                "required": []
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


# --- TOOL IMPLEMENTATIONS ---

def memory_write(filename: str, content: str) -> str:
    filepath = os.path.join(MEMORY_DIR, f"{filename}.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n## [{timestamp}]\n{content}\n"
    with open(filepath, 'a') as f:
        f.write(entry)
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
    return "\n\n".join(results) if results else f"No memories found for '{query}'"

def memory_read(filename: str) -> str:
    if not filename.endswith('.md'):
        filename = f"{filename}.md"
    filepath = os.path.join(MEMORY_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {filename}"

def checkpoint_save(task: str, status: str, completed: list = None, next_steps: list = None, blockers: str = None) -> str:
    checkpoint = {
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "status": status,
        "completed": completed or [],
        "next_steps": next_steps or [],
        "blockers": blockers
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    return f"✓ Checkpoint saved: {task} ({status})"

def checkpoint_load() -> str:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            cp = json.load(f)
        return f"""Last checkpoint:
Task: {cp['task']}
Status: {cp['status']}
When: {cp['timestamp']}
Completed: {', '.join(cp['completed']) if cp['completed'] else 'None yet'}
Next steps: {', '.join(cp['next_steps']) if cp['next_steps'] else 'None specified'}
Blockers: {cp['blockers'] or 'None'}"""
    return "No checkpoint found."

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
        return f"✓ Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def list_files(path: str = ".") -> str:
    try:
        items = os.listdir(path)
        return "\n".join(items) if items else "(empty)"
    except Exception as e:
        return f"Error: {e}"

def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOL_FUNCTIONS = {
    "memory_write": memory_write,
    "memory_search": memory_search,
    "memory_read": memory_read,
    "checkpoint_save": checkpoint_save,
    "checkpoint_load": checkpoint_load,
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "get_current_time": get_current_time,
}

def execute_tool(name: str, args: dict) -> str:
    if name not in TOOL_FUNCTIONS:
        return f"Unknown tool: {name}"
    try:
        return TOOL_FUNCTIONS[name](**args)
    except Exception as e:
        return f"Error: {e}"


# --- THE AGENT LOOP ---

def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    print("Microwave Memory Agent (with behavioral patterns)")
    print("This agent thinks before it acts.")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        if not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        # Keep context manageable
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
                    
                    # Show what's happening
                    arg_preview = list(args.values())[0][:40] if args else ""
                    print(f"  [{name}: {arg_preview}...]")
                    
                    result = execute_tool(name, args)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })
                continue
            else:
                # Print the response, preserving the pre-game routine formatting
                messages.append({"role": "assistant", "content": msg.content})
                print(f"\nMicrowave:\n{msg.content}\n")
                break


if __name__ == "__main__":
    main()
