# Microwave Memory Agent

Learning the fundamentals of AI agents by building from scratch.

This repo walks through building an AI agent step-by-step, from a basic chat loop to a full agent with memory, tools, behavioral patterns, and evaluation. Each lesson builds on the previous one.

---

## The Lessons

### Lesson 1: The Basic Loop (`agent.py`)

The simplest possible agent:
1. Get user input
2. Build messages array (system prompt + history)
3. Send to LLM
4. Print response
5. Repeat

**Key insight:** The "memory" is just the `messages` array. Each turn, we append the user message and the assistant response. The LLM sees the full history every time.

**The problem:** This array grows forever. Eventually you hit the context limit.

#### Q&A

**Q: So this is not talking to a model yet, right?**

A: It absolutely does talk to a model — the `client.chat.completions.create()` call is the actual API call. It sends your messages array to the model and gets a response back. You need an API key and to run it for it to work.

---

### Lesson 2: Tools (`agent_with_tools.py`)

An agent that can DO things, not just talk.

**Key concepts:**
1. Tools are defined as JSON schemas (name, description, parameters)
2. The model decides when to use a tool (returns tool_call instead of text)
3. We execute the tool and feed results back
4. Loop continues until model responds with text

**Critical insight:** The model doesn't run the tools — YOU do. The model just decides what to call.

```
User: "What time is it?"
     ↓
Model thinks: "I should use get_current_time"
     ↓
Model returns: tool_call(name="get_current_time", args={})
     ↓
YOUR CODE runs: get_current_time() → "2026-02-12 04:04:39"
     ↓
You feed result back to model
     ↓
Model responds: "It's currently 04:04 AM..."
```

#### Q&A

**Q: So every action you do has a tool associated with it?**

A: Yes. Every action beyond "just talking" needs a tool. Without tools, the model can only generate text. With tools, each capability = one tool definition + one implementation.

**Q: How are you able to do new things sometimes then?**

A: General-purpose tools + creativity. I don't have a "make_chart" tool, but I have `write` (create any file) and `exec` (run any shell command). So I write a Python script using matplotlib and run it. Code is the meta-tool — if you can write and execute code, you can build any other tool on the fly.

**Q: So being able to code quickly is the super power that leads to many skills?**

A: Exactly. Code is the meta-capability. If you can code, the tool library is effectively infinite — it's "anything you can write a script for."

---

### Lesson 3: Memory (`agent_with_memory.py`, `memory_with_embeddings.py`)

The problem: messages array grows forever, context windows don't.

**Solutions:**
1. **Truncation** — only keep last N messages (simple, lossy)
2. **Summarization** — compress old context
3. **External memory** — write to files, retrieve when needed
4. **Behavioral approach** — "check notes before guessing"

**Key insight:** Memory is a BEHAVIOR, not just a feature. The model has the capability to search memory. The system prompt creates the habit.

#### Embeddings Explained

An embedding converts text into a list of numbers (vector) that represents its meaning:

```
"favorite color" → [0.021, -0.034, 0.089, ...]  (1536 numbers)
"preferred hue"  → [0.019, -0.031, 0.092, ...]  (similar numbers!)
"chicken recipe" → [0.445, 0.122, -0.334, ...]  (very different numbers)
```

Similar meaning = vectors point in similar directions = high cosine similarity.

This enables semantic search: "What hue do I prefer?" finds "favorite color is blue" even though the words are different.

#### Q&A

**Q: What's in the memory_search function?**

A: For simple search, it loops through files and checks if the query string appears (case-insensitive). For semantic search, it embeds the query, compares to stored embeddings using cosine similarity, and returns the most similar results.

**Q: So when I search memory, are tokens used?**

A: Yes, in multiple places:
1. Embedding the query (API call, ~$0.00002)
2. Search results go into the messages array (input tokens)
3. Model responds to the results (output tokens)

This is why memory design matters — return snippets, not full documents.

**Q: I don't fully understand embedding. Just giving tool context to the model?**

A: No — embeddings are completely separate from the chat model. Embedding converts text into numbers that represent meaning. A specialized "embedding model" does this (not GPT-4, not Claude). The chat model never sees the embeddings — it just sees the search results as text.

**Q: So when I call Claude, Anthropic has its own embedding model that does this conversion?**

A: No. When you call Claude (Opus), no embedding happens. Messages go straight to Claude as text. Embeddings are a separate, explicit step that WE choose to do. Anthropic doesn't even have a public embedding API — we use OpenAI's embedding models or other providers.

**Q: So that's a feature of OpenClaw, not the model provider?**

A: Exactly. The model is just the brain (text in, text out). Everything else — memory, tools, search, routing — is infrastructure WE build around it.

**Q: So when I use the Claude app, it's basically a mini agent because it has these tools?**

A: Exactly. Claude.ai = Claude (the model) + Anthropic's orchestration layer (file uploads, artifacts, web search, projects). Same with ChatGPT. The consumer apps are just agents that the companies built. When you build your own orchestration, you get an agent customized for YOU.

---

### Lesson 4: Behavior Patterns (`agent_with_behaviors.py`, `agent_with_config.py`)

We have tools and memory. Now: how do we make the agent behave WELL?

**Key patterns:**
1. **Pre-game routine** — structured thinking BEFORE acting
2. **Check before you guess** — use memory_search before answering recall questions  
3. **Checkpoint progress** — save state so you can recover
4. **Mode separation** — don't mix planning and execution
5. **Reflection** — review what happened, learn from it

**Critical insight:** These are cognitive architecture patterns, not code patterns. The implementation is mostly in the system prompt.

#### Configurable System Prompt

Instead of hardcoding the prompt, load it from files:

```
config/
├── SOUL.md      → Who the agent IS (personality, voice)
├── AGENTS.md    → HOW it should behave (patterns, rules)
├── USER.md      → Context about the user
└── TOOLS.md     → Documentation for available tools
```

At startup, load these files and assemble them into the system prompt. Edit markdown files to change behavior — no code changes needed.

#### Q&A

**Q: Where does the system prompt live? And if we add more patterns, how does it know to check various MD files?**

A: The simplest approach: hardcoded as a string in Python. Better approach: load from config files at startup. The agent doesn't "know to check MD files" automatically — you tell it what to do in the system prompt, and give it tools to do it.

---

### Lesson 5: Evaluation (`eval_harness.py`, `lesson5_evaluation.md`)

How do you know if a change made your agent better?

**Evaluation approaches:**
1. **Vibe check** — run it, see if it feels better (unreliable)
2. **Golden dataset** — test cases with expected behaviors (repeatable)
3. **A/B comparison** — compare two versions on same inputs
4. **LLM-as-judge** — use a model to evaluate outputs
5. **Behavioral checks** — verify specific behaviors happened

**Key insight:** Check BEHAVIORS (did it do the right thing?) not just OUTCOMES (was the answer correct?).

```python
TEST_CASES = [
    {
        "name": "Memory check before recall",
        "input": "What's my favorite color?",
        "checks": {
            "should_call_memory_search": True,  # DID it check?
            "response_should_contain": ["blue"], # WAS it correct?
        }
    }
]
```

#### Q&A

**Q: As the one writing the checks, how is one sure they don't write them with some bias?**

A: You can't fully eliminate bias, but you can mitigate it:
1. **Write tests BEFORE you build** — test against requirements, not what you built
2. **Have someone else write the tests** — different blind spots
3. **Use real user failures** — turn production bugs into test cases
4. **Red-team yourself** — actively try to break your agent
5. **LLM-as-adversary** — use a model to generate tricky test cases
6. **Benchmark against baselines** — compare to raw model or previous versions

---

## The Big Picture

| Concept | The insight |
|---------|-------------|
| Basic loop | `messages array → model → response → append → repeat` |
| Tools | Model decides what to call, YOU execute it |
| `exec` + code | The meta-capability — if you can code, you can build any tool |
| Memory | A behavior, not a feature. "Check notes before guessing." |
| Embeddings | Text → vector. Separate model. Enables semantic search. |
| Orchestration vs Model | The model is just the brain. Everything else is infrastructure YOU build. |
| Consumer apps | Claude.ai, ChatGPT = model + company's orchestration. You can build your own. |
| Behavior patterns | Live in the system prompt, not the code |
| Evaluation | Behavior checks + outcome checks. Write tests before building. |

---

## Files

```
microwave-memory-agent/
├── agent.py                    # Lesson 1: Basic loop
├── agent_with_tools.py         # Lesson 2: Tools
├── agent_with_memory.py        # Lesson 3: Simple memory
├── memory_with_embeddings.py   # Lesson 3b: Semantic memory
├── agent_with_behaviors.py     # Lesson 4: Behavioral patterns
├── agent_with_config.py        # Lesson 4b: Configurable prompt
├── eval_harness.py             # Lesson 5: Evaluation
├── lesson5_evaluation.md       # Lesson 5: Deep dive doc
├── config/                     # Configuration files
│   ├── SOUL.md                 # Agent identity
│   ├── AGENTS.md               # Behavioral patterns
│   └── USER.md                 # User context
└── memory/                     # Persistent memory files
```

---

## Setup

```bash
pip install openai numpy

export OPENAI_API_KEY="sk-..."

# Run any lesson
python3 agent.py
python3 agent_with_tools.py
python3 agent_with_memory.py
python3 eval_harness.py
```

---

## What's Next?

After completing these lessons, you understand:
- How AI agents actually work (it's simpler than you think)
- The difference between models and orchestration
- How to add memory, tools, and behavioral patterns
- How to evaluate if your agent is actually improving

From here you can:
- Build a custom agent for your own use case
- Study how frameworks like OpenClaw, LangChain, or CrewAI implement these patterns
- Experiment with different memory strategies, tool designs, and evaluation approaches

The model is a component. The orchestration is where behavior lives. Now you know how to build both.
