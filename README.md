# network_flow_solver

A clean, runnable reproduction of the **network maintenance scheduling problem**: schedule
arc-maintenance jobs over a discrete time horizon so as to maximise total source→sink flow.
The point of the exercise is to reproduce a known result — **a disaggregated Benders
decomposition beats a direct MIP** on the larger, wide-maintenance-window instances —
following Pearce & Forbes (2019; preprint arXiv:1603.02378).

It's built to be read aloud and whiteboarded, so the formulations are explicit and heavily
commented rather than clever.

> **Status:** the shared **spine** is in place — the instance data model (`src/instance.py`),
> the result record (`src/result.py`), and the backend-selection interface (`src/backends.py`),
> with tooling and a passing test suite. The solver stages (generator, direct MIP, Benders,
> benchmark) are next. The build is staged (see the task spec).

## Quick start

```bash
uv sync                                 # install dependencies
uv run pytest                           # tests (spine modules; 19 passing)
```

> The `run.py` entry point and its flags arrive with the later stages:
>
> ```bash
> uv run python run.py --solver-check     # report which solver backends are available
> uv run python run.py --quick            # run the demo end-to-end → comparison chart
> uv run python run.py --backend cp-sat   # pick a backend: cp-sat | scip | highs | gurobi
> ```

Solving runs on **Google OR-Tools** behind a thin `--backend` interface: **CP-SAT**
(default) plus **SCIP** and **HiGHS** are always available; **Gurobi** is used only if a
license is present and is never a hard dependency. No license or keys are hardcoded.

## What's in this repo

| Path | What it is |
|---|---|
| `claudecode-prompt-maintenance-scheduling.md` | The task spec — staged deliverables and acceptance criteria |
| `1603.02378v2.pdf`, `1703.06581v1.pdf` | The source papers being reproduced |
| `CLAUDE.md` | The operating doc for AI agents — project facts, commands, standards (single source of truth; no separate `AGENTS.md`) |
| `docs/ARCHITECTURE.md` | The planned pipeline and both formulations, with diagrams |
| `docs/CODING_STANDARDS.md` | Python conventions, tooling (uv/ruff/mypy/pytest), naming, git |
| `docs/task_spec_template.md` | Lightweight format for scoping a stage or change |
| `.claude/commands/` | Slash commands: `/test`, `/lint`, `/typecheck`, `/inspect-package` |

`src/` now holds the shared spine — `instance.py` (data model), `result.py` (result record),
`backends.py` (solver-selection interface) — with `tests/` covering them. The solver
formulations and `run.py` arrive with the later stages.

## How this repo is developed

Work is driven by the **[superpowers](https://github.com/obra/superpowers)** Claude Code
plugin, which supplies the *process*: brainstorm → write a plan → test-driven development →
execute → code review → verify. The skills trigger automatically; you don't invoke them by
hand. `CLAUDE.md` supplies the project-specific *facts* those skills consult — it doesn't
re-specify the workflow.

For a human picking this up: read the task spec, then `docs/ARCHITECTURE.md`. To build a
stage, just say what you want (e.g. "let's do Stage 1, the instance generator") and the
plan→TDD→execute loop takes over.

## Method in one paragraph

The schedule is a binary decision (when each maintenance job runs); given a schedule, each
period's maximum flow is an independent **max-flow / min-cut** on the network with
under-maintenance arcs removed. Benders exploits this: a **master** MIP picks the schedule
with a per-period flow estimate `theta_t`; for each candidate schedule we solve the cheap
per-period min-cuts and feed back a **disaggregated optimality cut per period** — injected as
a lazy constraint where the backend supports callbacks, otherwise via an iterative re-solve
loop. Solving the subproblems as min-cuts rather than LPs — and disaggregating the cuts — is
what makes it outrun the monolithic time-indexed MIP on hard instances.
