# CLAUDE.md

The single operating doc for this repo (there is no separate `AGENTS.md`). Read at session
start. This file is the **project reference** — the *what*. The *how* (process) is driven by
the superpowers skills, see below.

## What this is

A readable reproduction of the **network maintenance scheduling problem**: schedule
arc-maintenance jobs over a discrete horizon to maximise total source→sink flow. The
headline result to reproduce is that **disaggregated Benders decomposition beats a direct
MIP** on larger / wide-window instances. It's an interview demo, so formulations must be
explicit and whiteboardable. Full spec: `claudecode-prompt-maintenance-scheduling.md`.

## How work happens here (process)

Process is driven by the **superpowers** plugin skills, not by this file: `brainstorming` →
`writing-plans` → `test-driven-development` → `executing-plans` /
`subagent-driven-development` → `requesting`/`receiving-code-review` →
`verification-before-completion` (plus `systematic-debugging`, `using-git-worktrees`). They
trigger automatically. **When this file and a superpowers skill differ on *process*, the
skill wins** — this file supplies project facts the skills consult, not a competing workflow.

Natural flow per build stage: the spec already exists → `writing-plans` for the stage →
`test-driven-development` (the toy instance is a known-answer test) → execute → verify.

## Source material (in repo)

- `claudecode-prompt-maintenance-scheduling.md` — full task spec / staged deliverables (read first)
- `1603.02378v2.pdf` — Pearce & Forbes preprint (the method reproduced)
- `1703.06581v1.pdf` — companion paper

## Docs

- `docs/ARCHITECTURE.md` — planned pipeline + both formulations + the backend interface, with diagrams
- `docs/CODING_STANDARDS.md` — Python conventions, tooling, naming, git
- `docs/task_spec_template.md` — lightweight format for scoping a stage or change

## Solver stack — Google OR-Tools (read before any solver work)

Model through **OR-Tools** behind one thin **solver-selection interface** (`--backend`);
switching backend must never require touching the formulation. In scope:

- **CP-SAT** (`ortools.sat.python.cp_model`, `CpModel`/`CpSolver`) — the integer constraint
  solver. **Default/primary backend.** Always available.
- **MathOpt** (`ortools.math_opt.python`) — unified interface fronting **SCIP** and **HiGHS**
  (bundled, always available) and **Gurobi** (only if licensed). Prefer MathOpt over the old
  `pywraplp`/`linear_solver` layer.

Backends: **CP-SAT, SCIP, HiGHS** always; **Gurobi optional** — never a hard dependency, used
only if a license is present. Never hardcode any solver license/keys.

**CP-SAT integer-flow caveat:** CP-SAT is integer/Boolean only, so per-period flow vars are
**integer** under CP-SAT. That is exact here (integral arc capacities ⇒ integral max flow) —
note it in a comment. MathOpt MIP backends (SCIP/HiGHS/Gurobi) use **continuous** flow vars.
Keep both cases behind the solver interface so the rest of the code is backend-agnostic.

**Benders cut injection depends on the backend:** lazy constraints during search only on
backends that expose them (Gurobi natively; check whether MathOpt exposes lazy-constraint
callbacks for SCIP). On CP-SAT and any backend without callback support, fall back to an
**iterative cut loop** (re-solve master, add disaggregated cuts as ordinary constraints,
repeat to convergence). Gate on `--backend` and state the assumption in a comment.

## Repository structure (intended)

```
src/
  generator.py     # Stage 1 — random instance generator (--seed), JSON, toy instance
  backends.py      # the thin solver-selection interface (--backend {cp-sat,scip,highs,gurobi})
  direct_mip.py    # Stage 2 — time-indexed formulation, backend-agnostic via backends.py
  benders.py       # Stage 3 — disaggregated Benders: master + per-period min-cut subproblem,
                   #            disaggregated cuts via lazy callback OR iterative loop fallback
  benchmark.py     # Stage 4 — run both methods across sizes/backends, emit CSV + log-y chart
run.py             # entry point: `--quick`, `--solver-check`, `--backend`
tests/             # pytest — toy instance known optimum; methods AND backends MUST agree
results/           # CSV + chart artefacts (gitignored — regenerate via run.py)
```

## Commands

Python 3.12+, `uv`, `ortools` (CP-SAT + MathOpt + bundled SCIP/HiGHS), `networkx`
(max-flow/min-cut), `numpy`, `matplotlib`, `pytest`. Run Python through `uv` — never bare
`python3`.

```bash
uv sync                                      # install dependencies
uv run python run.py --solver-check          # report which backends are available at runtime
uv run python run.py --quick                 # end-to-end demo (~minutes) → comparison chart
uv run python run.py --backend cp-sat        # choose a backend (cp-sat | scip | highs | gurobi)
uv run pytest                                # full test suite
uv run ruff check . && uv run ruff format --check .   # lint + format
uv run mypy .                                          # type check
```

## Standards (key points)

Full details in `docs/CODING_STANDARDS.md`.

- Type annotations on every function; `mypy --strict`; `ruff` clean.
- Formulation files (`direct_mip.py`, `benders.py`) are self-contained and commented **at
  the constraint level**, and stay **backend-agnostic** — backend choice lives in
  `backends.py` only. Module docstring states the math. Readability > cleverness.
- State any formulation ambiguity you resolve as an inline comment with the assumption.
- No bare `except:`; no `# type: ignore` without a justifying comment.
- Don't over-engineer — this is a demo, not a product.

## Testing

- The toy instance has a hand-checkable known optimum; assert it.
- **Direct MIP and Benders must return the same optimum**, and **results must agree across
  backends** (CP-SAT / SCIP / HiGHS), on the toy instance and on any instance solved to
  optimality — the core correctness check.
- Every bugfix gets a regression test.

## Git

- Commit **after each build stage** with a clear `<type>: <message>`. After Stage 4, stop
  and summarise results before attempting Stage 5.
- Branches: `feat/`, `fix/`, `chore/`, `refactor/`, `docs/`, `test/`.

## Reporting (output format)

Structure substantive task responses as **PLAN / CHANGES / TESTS / VALIDATION / RISKS**.
This is a reporting convention — it complements the superpowers process, it doesn't replace
it.

## What NOT to do

- Don't change the agreed formulation or master/subproblem split without flagging it first.
- Don't let backend-specific code leak out of `backends.py` into the formulation files.
- Don't make Gurobi a hard dependency; don't hardcode any solver license or commit
  `.env` / `gurobi.lic`.
- Don't add dependencies beyond the declared set without justification.
- Don't commit generated artefacts (`results/`, large instance JSON, charts).
- Don't push directly to `main` once a remote exists.
