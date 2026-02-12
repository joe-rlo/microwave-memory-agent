"""
Microwave Memory Agent - Learning the fundamentals
Lesson 3b: Semantic Memory with Embeddings

Upgrade from string matching to semantic search.

Key concepts:
1. Embeddings: convert text → vector (list of numbers)
2. Similar meaning = similar vectors
3. Search by comparing vectors (cosine similarity)

This is the foundation of RAG (Retrieval Augmented Generation).
"""

import os
import json
import numpy as np
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"  # Fast, cheap, good enough

MEMORY_DIR = "./memory"
EMBEDDINGS_FILE = "./memory/embeddings.json"
os.makedirs(MEMORY_DIR, exist_ok=True)


# --- EMBEDDINGS EXPLAINED ---
#
# An embedding is a list of numbers (vector) that represents the MEANING of text.
#
# "favorite color" → [0.021, -0.034, 0.089, ...]  (1536 numbers)
# "preferred hue"  → [0.019, -0.031, 0.092, ...]  (similar numbers!)
# "chicken recipe" → [0.445, 0.122, -0.334, ...]  (very different numbers)
#
# Similar meaning = vectors point in similar directions = high cosine similarity


def get_embedding(text: str) -> list[float]:
    """Convert text to a vector using OpenAI's embedding model."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate similarity between two vectors. Returns 0-1 (1 = identical)."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def load_embeddings() -> dict:
    """Load the embeddings index from disk."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'r') as f:
            return json.load(f)
    return {"entries": []}


def save_embeddings(data: dict):
    """Save the embeddings index to disk."""
    with open(EMBEDDINGS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


# --- MEMORY TOOLS ---

def memory_write(filename: str, content: str, append: bool = True) -> str:
    """Write to memory AND create embedding for semantic search."""
    filepath = os.path.join(MEMORY_DIR, f"{filename}.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    entry = f"\n## [{timestamp}]\n{content}\n"
    
    # Write to file
    mode = 'a' if append else 'w'
    try:
        with open(filepath, mode) as f:
            if not append:
                f.write(f"# {filename}\n")
            f.write(entry)
    except Exception as e:
        return f"Error writing memory: {e}"
    
    # Create embedding and add to index
    try:
        embedding = get_embedding(content)
        
        embeddings_data = load_embeddings()
        embeddings_data["entries"].append({
            "id": f"{filename}_{timestamp}",
            "filename": filename,
            "content": content,
            "timestamp": timestamp,
            "embedding": embedding
        })
        save_embeddings(embeddings_data)
        
        return f"✓ Saved to memory: {filename}.md (with embedding)"
    except Exception as e:
        return f"Saved to file but embedding failed: {e}"


def memory_search(query: str, top_k: int = 3) -> str:
    """
    Semantic search: find memories with similar MEANING, not just matching words.
    
    "preferred color" will find "favorite color is blue"
    "what does user like" will find "Joe likes building AI agents"
    """
    embeddings_data = load_embeddings()
    
    if not embeddings_data["entries"]:
        return "No memories stored yet."
    
    # Embed the query
    try:
        query_embedding = get_embedding(query)
    except Exception as e:
        return f"Error creating query embedding: {e}"
    
    # Calculate similarity to each memory
    scored = []
    for entry in embeddings_data["entries"]:
        similarity = cosine_similarity(query_embedding, entry["embedding"])
        scored.append({
            "similarity": similarity,
            "filename": entry["filename"],
            "content": entry["content"],
            "timestamp": entry["timestamp"]
        })
    
    # Sort by similarity (highest first)
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Return top results
    results = []
    for item in scored[:top_k]:
        # Only include if similarity is meaningful (> 0.3 is a reasonable threshold)
        if item["similarity"] > 0.3:
            results.append(
                f"[{item['similarity']:.2f}] ({item['filename']}, {item['timestamp']})\n{item['content']}"
            )
    
    if results:
        return "Found relevant memories:\n\n" + "\n\n---\n\n".join(results)
    else:
        return f"No memories found semantically related to '{query}'"


def memory_list() -> str:
    """List all memory files."""
    files = [f for f in os.listdir(MEMORY_DIR) if f.endswith('.md')]
    if files:
        return "Memory files:\n" + "\n".join(f"- {f}" for f in files)
    return "No memory files yet."


def memory_read(filename: str) -> str:
    """Read a specific memory file."""
    if not filename.endswith('.md'):
        filename = f"{filename}.md"
    filepath = os.path.join(MEMORY_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Memory file not found: {filename}"


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --- TOOL DEFINITIONS ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Write something to long-term memory with semantic indexing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Category for the memory"},
                    "content": {"type": "string", "description": "What to remember"}
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Semantic search - finds memories by MEANING, not just keywords. Use this before answering questions about things you should remember.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for (can be natural language)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_list",
            "description": "List all memory files",
            "parameters": {"type": "object", "properties": {}, "required": []}
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
                    "filename": {"type": "string", "description": "Name of the file"}
                },
                "required": ["filename"]
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

TOOL_FUNCTIONS = {
    "memory_write": memory_write,
    "memory_search": memory_search,
    "memory_list": memory_list,
    "memory_read": memory_read,
    "get_current_time": get_current_time,
}

SYSTEM_PROMPT = """You are a helpful assistant named Microwave.
You have semantic memory - you can find memories by meaning, not just keywords.

BEFORE answering questions about past conversations or facts:
- Use memory_search with a natural language query
- It finds semantically similar content, so "preferred color" finds "favorite color"
- Don't guess - check first

WHEN learning something important:
- Use memory_write to save it
- Be specific with the content"""


def execute_tool(tool_name: str, arguments: dict) -> str:
    if tool_name not in TOOL_FUNCTIONS:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        return TOOL_FUNCTIONS[tool_name](**arguments)
    except Exception as e:
        return f"Error: {e}"


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    print("Microwave Memory Agent (with SEMANTIC memory)")
    print("This version understands meaning, not just keywords.")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        if not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        # Keep only last 20 messages
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
                    print(f"  [🧠 {name}: {args.get('query') or args.get('content', '')[:50]}...]")
                    
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
