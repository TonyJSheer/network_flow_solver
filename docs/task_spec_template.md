# Task Spec Template

Lightweight format for scoping a unit of work (e.g. one build stage or a focused change),
adapted from the AI Blueprint Framework. The framework's full template has DB/API/frontend
sections that don't apply to this project — they're omitted here.

A task should be completable in a single focused session. If it sprawls across many files or
mixes unrelated concerns, split it and say so.

```
Title:
Type: feature | bugfix | refactor | docs | test | chore

Summary:
[One or two sentences — what changes and why]

Depends on:
[other tasks/stages, or "—"]

Files likely affected:
- src/<file>.py — what changes (name the function/class where known)
- tests/<file>.py — what changes

New packages (if any):
- <package> — justification

Requirements:
- [What must be true when done — functional]
- [Performance / correctness constraints, if relevant]

Acceptance criteria:
- [ ] specific, verifiable condition
- [ ] (for solver work) MIP and Benders agree on the toy instance

Tests required:
- [test name — what it verifies]

Validation:
  uv run pytest
  uv run ruff check . && uv run mypy .

Risks:
- [edge cases, anything the reviewer should watch]
```

Respond with: **PLAN → CHANGES → TESTS → VALIDATION → RISKS** (see `docs/AGENTS.md`).
