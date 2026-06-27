# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A clean, runnable reproduction of the **network maintenance scheduling problem**: schedule
arc-maintenance jobs over a discrete time horizon to maximise total source→sink flow. The
headline result to reproduce is that **disaggregated Benders decomposition beats a direct
MIP on larger / wide-window instances**.

Built to serve as a readable interview demo — the formulation will be explained to a
possibly non-specialist panel and whiteboarded from memory, so favour explicit, commented
formulations over clever ones.

## Source material (in repo)

- `claudecode-prompt-maintenance-scheduling.md` — the full task spec / deliverables (read
  this first; it defines the staged build and acceptance criteria).
- `1603.02378v2.pdf` — Pearce & Forbes preprint (the method being reproduced).
- `1703.06581v1.pdf` — companion paper.

## Key docs (read at session start)

- `docs/AGENTS.md` — the operating contract: commands, structure, testing, definition of
  done, output format. **Start here.**
- `docs/ARCHITECTURE.md` — the planned pipeline + both formulations (with diagrams).
- `docs/CODING_STANDARDS.md` — Python conventions, tooling, naming, git (adapted from
  `../repo_template` for this optimisation project).
- `docs/task_spec_template.md` — lightweight format for scoping a stage or change.
- Run Python via **`uv`** only (`uv run python ...`, `uv run pytest`) — never bare `python3`.
- Lint/type/test: `uv run ruff check .`, `uv run mypy .`, `uv run pytest`.

## Build approach (from the task spec)

Staged, commit after each: (1) instance generator → (2) direct MIP baseline (gurobipy
matrix API) → (3) disaggregated Benders (min-cut subproblem, lazy-constraint callback) →
(4) benchmark harness + chart → (5 optional) HiGHS solver-swap. Direct MIP and Benders
**must agree** on the toy instance and on any instance both solve to optimality.

## Tooling notes

- GSD workflow hooks are intentionally disabled for this repo (kept the statusline).
- `superpowers` plugin is enabled; `context7` MCP is configured for library docs
  (gurobipy / networkx / highspy).
