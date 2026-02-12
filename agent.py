"""
Microwave Memory Agent - Learning the fundamentals
Lesson 1: The Basic Loop

The simplest possible agent:
1. Get user input
2. Build a prompt (system + user message)
3. Send to LLM
4. Print response
5. Repeat
"""

import os
from openai import OpenAI

# --- Configuration ---
# We'll use OpenAI's API format (works with OpenAI, OpenRouter, local models, etc.)
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    # Uncomment below to use OpenRouter instead:
    # base_url="https://openrouter.ai/api/v1",
    # api_key=os.environ.get("OPENROUTER_API_KEY"),
)

MODEL = "gpt-4o-mini"  # cheap and fast for learning

# --- The System Prompt ---
# This is where behavior shaping happens
SYSTEM_PROMPT = """You are a helpful assistant named Microwave.
You are direct, slightly sarcastic, and efficient.
You don't use filler phrases like "Great question!" or "I'd be happy to help!"
Just help."""

# --- The Core Loop ---
def main():
    """The simplest agent loop possible."""
    
    # This is our "memory" for now - just the conversation history
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    
    print("Microwave Memory Agent")
    print("Type 'quit' to exit\n")
    
    while True:
        # 1. Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # 2. Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # 3. Send to LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
        )
        
        # 4. Extract the response
        assistant_message = response.choices[0].message.content
        
        # 5. Add assistant response to history (this is how it "remembers")
        messages.append({"role": "assistant", "content": assistant_message})
        
        # 6. Print and loop
        print(f"\nMicrowave: {assistant_message}\n")


if __name__ == "__main__":
    main()
