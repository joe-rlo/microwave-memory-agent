# Microwave Memory Agent

Learning the fundamentals of AI agent behavior by building from scratch.

## The Lessons

### Lesson 1: The Basic Loop (`agent.py`)

The simplest possible agent. Just:
1. Get user input
2. Build messages array (system prompt + history)
3. Send to LLM
4. Print response
5. Repeat

**Key insight:** The "memory" is just the `messages` array. Each turn, we append the user message and the assistant response. The LLM sees the full history every time.

```
messages = [
    {"role": "system", "content": "You are..."},      # Always first
    {"role": "user", "content": "Hello"},             # Turn 1
    {"role": "assistant", "content": "Hi there"},     # Turn 1 response
    {"role": "user", "content": "What's 2+2?"},       # Turn 2
    {"role": "assistant", "content": "4"},            # Turn 2 response
    {"role": "user", "content": "Thanks"},            # Turn 3 (current)
    # LLM generates response for turn 3...
]
```

**The problem:** This array grows forever. Eventually you hit the context limit. That's why we need smarter memory strategies (coming in later lessons).

## Setup

```bash
# Install dependencies
pip install openai

# Set your API key
export OPENAI_API_KEY="sk-..."

# Or use OpenRouter (edit agent.py to uncomment the base_url)
export OPENROUTER_API_KEY="sk-or-..."

# Run it
python agent.py
```

## What's Next

- Lesson 2: Adding tools (let the agent do things)
- Lesson 3: Memory strategies (what happens when context fills up?)
- Lesson 4: Behavior patterns (pre-game routines, checkpointing)
- Lesson 5: Evaluation (how do you know if changes help?)
