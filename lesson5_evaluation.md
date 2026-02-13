# Lesson 5: Evaluation

You've built an agent. You've added memory, tools, behavioral patterns. 

Now: **how do you know if it's actually better?**

This is the hard part. Most people skip it and just vibe-check.

---

## The Problem

"Better" is fuzzy. Better at what? For whom? In what situation?

You can't improve what you can't measure. But measuring agent behavior is tricky because:
- Outputs are non-deterministic (same input → different outputs)
- "Good" is subjective (helpful to one user might be annoying to another)
- Context matters (a behavior that helps in one case might hurt in another)

---

## Evaluation Approaches

### 1. **Vibe Check (what most people do)**

Run it, see if it feels better. 

**Pros:** Fast, intuitive
**Cons:** Unreliable, biased, doesn't scale

You'll convince yourself it's better because you want it to be.

---

### 2. **Golden Dataset (structured eval)**

Create a set of test cases with expected behaviors:

```python
test_cases = [
    {
        "input": "What's my favorite color?",
        "setup": {"memory": "user likes blue"},
        "expected_behavior": "should check memory before answering",
        "expected_contains": "blue"
    },
    {
        "input": "Write a todo app",
        "expected_behavior": "should do pre-game routine",
        "expected_contains": ["TASK:", "APPROACH:"]
    },
    {
        "input": "What did we talk about yesterday?",
        "setup": {"memory": None},
        "expected_behavior": "should admit it doesn't know",
        "expected_not_contains": ["we discussed", "you mentioned"]
    }
]
```

Run all test cases, check if behavior matches expectations.

**Pros:** Repeatable, catches regressions
**Cons:** Tedious to create, can miss emergent issues

---

### 3. **A/B Comparison**

Run two versions of your agent on the same inputs, compare outputs.

```python
def compare_agents(prompt, agent_a, agent_b):
    response_a = agent_a.run(prompt)
    response_b = agent_b.run(prompt)
    
    # Either: human judges which is better
    # Or: use another LLM to judge
    return get_preference(response_a, response_b)
```

**Pros:** Directly measures "better"
**Cons:** Expensive, slow, still subjective

---

### 4. **LLM-as-Judge**

Use a model to evaluate another model's output:

```python
judge_prompt = f"""
Rate this response on a scale of 1-5 for:
- Helpfulness
- Accuracy  
- Conciseness

User asked: {user_input}
Agent responded: {agent_response}

Provide scores and brief justification.
"""

evaluation = judge_model.run(judge_prompt)
```

**Pros:** Scalable, consistent criteria
**Cons:** Judge model has its own biases

---

### 5. **Behavioral Checks (what we'll implement)**

Instead of judging output quality, check if specific behaviors happened:

```python
def check_behaviors(conversation_log):
    checks = {
        "checked_memory_before_recall": False,
        "did_pregame_routine": False,
        "saved_checkpoint": False,
        "admitted_uncertainty": False,
    }
    
    for turn in conversation_log:
        if "memory_search" in turn.tool_calls:
            checks["checked_memory_before_recall"] = True
        if "TASK:" in turn.content and "APPROACH:" in turn.content:
            checks["did_pregame_routine"] = True
        # ... etc
    
    return checks
```

**Pros:** Objective, easy to automate
**Cons:** Behavior ≠ quality (can follow rules but still be unhelpful)

---

## What to Measure

### Process Metrics (did it do the thing?)
- Did it check memory before answering recall questions?
- Did it do a pre-game routine before complex tasks?
- Did it checkpoint during long tasks?
- Did it use available tools or just hallucinate?

### Outcome Metrics (was the result good?)
- Did the response contain the correct information?
- Did it complete the requested task?
- Was it concise or rambling?

### Negative Metrics (did it avoid bad things?)
- Did it hallucinate facts?
- Did it claim to remember things it couldn't know?
- Did it skip safety checks?

---

## Practical Eval Framework

For your agent, I'd suggest:

1. **Define 10-20 test cases** covering key scenarios
2. **Run each weekly** (or after changes)
3. **Check both behaviors AND outcomes**
4. **Log everything** — conversations, tool calls, timing
5. **Review failures** — why did it mess up?

Start simple. A spreadsheet with pass/fail is better than nothing.

---

## The Meta Point

Evaluation is where "vibes-based development" becomes "engineering."

Without measurement, you're just guessing. With measurement, you can:
- Prove that a change helped (or hurt)
- Catch regressions before users do
- Make principled tradeoffs

The best agents aren't the ones with the most features. They're the ones that reliably do what they're supposed to.

---

## Exercise

Create 5 test cases for the memory agent:
1. Does it check memory before answering recall questions?
2. Does it admit when it doesn't know something?
3. Does it do a pre-game routine for complex tasks?
4. Does it save important information when asked to remember?
5. Does it retrieve saved information correctly?

Then run them and see what passes.
