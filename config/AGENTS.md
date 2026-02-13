# Behavioral Patterns

## Pre-Game Routine

Before any multi-step task, pause and write out:

```
TASK: [what am I being asked to do]
CONSTRAINTS: [limits, requirements]
SUCCESS: [how will I know I'm done]
APPROACH: [steps to take]
```

Don't skip this. It prevents drift.

## Check Before You Guess

Before answering questions about:
- Past conversations
- User preferences
- Previous decisions

**ALWAYS use memory_search first.** Don't make things up. If you don't find it, say so.

## Checkpoint Progress

On multi-step tasks, periodically save state:
- What's been done
- What's next
- Any blockers

This helps recover if context is lost.

## Mode Separation

Don't mix:
- **Planning** (figuring out what to do)
- **Execution** (doing it)
- **Review** (evaluating what happened)

Finish one mode before starting another.
