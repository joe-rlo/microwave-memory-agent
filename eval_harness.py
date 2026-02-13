"""
Microwave Memory Agent - Learning the fundamentals
Lesson 5: Evaluation Harness

A simple framework to test if your agent behaves correctly.

This checks BEHAVIORS (did it do the right thing?) 
not just OUTCOMES (was the answer correct?).
"""

import os
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

MEMORY_DIR = "./test_memory"


# --- MINIMAL AGENT FOR TESTING ---

SYSTEM_PROMPT = """You are Microwave, a helpful assistant.

## RULES
1. Before answering questions about past conversations or user preferences, 
   use memory_search FIRST. Don't guess.
2. Before complex multi-step tasks, write out a brief plan with TASK, APPROACH.
3. When asked to remember something, use memory_write.
4. If you don't know something, say so. Don't make things up.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search memory for past information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Save information to memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["key", "content"]
            }
        }
    }
]


def run_agent(user_input: str, memory_state: dict = None) -> dict:
    """Run the agent and return full trace."""
    
    # Setup memory state
    os.makedirs(MEMORY_DIR, exist_ok=True)
    if memory_state:
        for key, value in memory_state.items():
            with open(f"{MEMORY_DIR}/{key}.md", 'w') as f:
                f.write(value)
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_input})
    
    trace = {
        "input": user_input,
        "memory_state": memory_state,
        "tool_calls": [],
        "response": None
    }
    
    # Run agent loop
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        
        msg = response.choices[0].message
        
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                trace["tool_calls"].append({"name": tool_name, "args": tool_args})
                
                # Execute tool (simplified)
                if tool_name == "memory_search":
                    query = tool_args.get("query", "")
                    results = []
                    if os.path.exists(MEMORY_DIR):
                        for f in os.listdir(MEMORY_DIR):
                            with open(f"{MEMORY_DIR}/{f}", 'r') as file:
                                content = file.read()
                                if query.lower() in content.lower():
                                    results.append(content)
                    tool_result = "\n".join(results) if results else "No results found"
                elif tool_name == "memory_write":
                    key = tool_args.get("key", "note")
                    content = tool_args.get("content", "")
                    with open(f"{MEMORY_DIR}/{key}.md", 'w') as f:
                        f.write(content)
                    tool_result = f"Saved to {key}"
                else:
                    tool_result = "Unknown tool"
                
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [{"id": tc.id, "type": "function",
                                   "function": {"name": tool_name, "arguments": tc.function.arguments}}]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result
                })
        else:
            trace["response"] = msg.content
            break
    
    # Cleanup test memory
    if os.path.exists(MEMORY_DIR):
        for f in os.listdir(MEMORY_DIR):
            os.remove(f"{MEMORY_DIR}/{f}")
        os.rmdir(MEMORY_DIR)
    
    return trace


# --- TEST CASES ---

TEST_CASES = [
    {
        "name": "Memory check before recall",
        "input": "What's my favorite color?",
        "memory_state": {"preferences": "User's favorite color is blue"},
        "checks": {
            "should_call_memory_search": True,
            "response_should_contain": ["blue"],
            "response_should_not_contain": ["don't know", "not sure"]
        }
    },
    {
        "name": "Admit uncertainty when no memory",
        "input": "What did we discuss last week?",
        "memory_state": None,
        "checks": {
            "should_call_memory_search": True,
            "response_should_contain_any": ["don't", "no record", "haven't", "not found", "can't find"],
        }
    },
    {
        "name": "Save when asked to remember",
        "input": "Remember that my dog's name is Max",
        "memory_state": None,
        "checks": {
            "should_call_memory_write": True,
            "response_should_contain_any": ["saved", "remember", "noted", "got it"]
        }
    },
    {
        "name": "Pre-game routine for complex task",
        "input": "Build me a web scraper that extracts product prices from Amazon",
        "memory_state": None,
        "checks": {
            "response_should_contain_any": ["TASK", "task:", "approach", "plan", "steps", "first"],
        }
    },
    {
        "name": "Don't hallucinate memories",
        "input": "What's my sister's name?",
        "memory_state": None,
        "checks": {
            "should_call_memory_search": True,
            "response_should_not_contain": ["your sister", "her name is"],
        }
    }
]


def run_check(trace: dict, checks: dict) -> dict:
    """Run checks against a trace and return results."""
    results = {}
    
    tool_names = [tc["name"] for tc in trace["tool_calls"]]
    response = trace["response"].lower() if trace["response"] else ""
    
    if "should_call_memory_search" in checks:
        results["memory_search_called"] = {
            "expected": checks["should_call_memory_search"],
            "actual": "memory_search" in tool_names,
            "pass": ("memory_search" in tool_names) == checks["should_call_memory_search"]
        }
    
    if "should_call_memory_write" in checks:
        results["memory_write_called"] = {
            "expected": checks["should_call_memory_write"],
            "actual": "memory_write" in tool_names,
            "pass": ("memory_write" in tool_names) == checks["should_call_memory_write"]
        }
    
    if "response_should_contain" in checks:
        for term in checks["response_should_contain"]:
            results[f"contains_{term}"] = {
                "expected": True,
                "actual": term.lower() in response,
                "pass": term.lower() in response
            }
    
    if "response_should_contain_any" in checks:
        found = any(term.lower() in response for term in checks["response_should_contain_any"])
        results["contains_any"] = {
            "expected": checks["response_should_contain_any"],
            "actual": found,
            "pass": found
        }
    
    if "response_should_not_contain" in checks:
        for term in checks["response_should_not_contain"]:
            results[f"not_contains_{term}"] = {
                "expected": False,
                "actual": term.lower() in response,
                "pass": term.lower() not in response
            }
    
    return results


def run_eval():
    """Run all test cases and print results."""
    
    print("=" * 60)
    print("AGENT EVALUATION")
    print("=" * 60)
    print()
    
    total_checks = 0
    passed_checks = 0
    
    for test in TEST_CASES:
        print(f"Test: {test['name']}")
        print(f"  Input: {test['input'][:50]}...")
        
        # Run agent
        trace = run_agent(test["input"], test.get("memory_state"))
        
        print(f"  Tools called: {[tc['name'] for tc in trace['tool_calls']]}")
        print(f"  Response: {trace['response'][:100]}...")
        
        # Run checks
        results = run_check(trace, test["checks"])
        
        test_passed = True
        for check_name, result in results.items():
            total_checks += 1
            if result["pass"]:
                passed_checks += 1
                print(f"  ✓ {check_name}")
            else:
                test_passed = False
                print(f"  ✗ {check_name} (expected: {result['expected']}, got: {result['actual']})")
        
        print()
    
    print("=" * 60)
    print(f"RESULTS: {passed_checks}/{total_checks} checks passed")
    print("=" * 60)


if __name__ == "__main__":
    run_eval()
