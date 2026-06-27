# AGENTS.md

**The operating contract between this repository and AI coding agents.** Read this at the
start of every session before touching code. Adapted from the AI Blueprint Framework
(`repo_template/`) for this single-package Python optimisation project — the framework's
web/DB/CDK material does not apply here.

---

## Project

**Name**: network_flow_solver

**Description**: A readable reproduction of the **network maintenance scheduling problem** —
schedule arc-maintenance jobs over a discrete horizon to maximise total source→sink flow.
Goal: show that **disaggregated Benders decomposition beats a direct MIP** on larger /
wide-window instances. Serves as an interview demo, so formulations must be explicit and
whiteboardable. Full spec: `claudecode-prompt-maintenance-scheduling.md`.

**Stack**: Python 3.11+, `uv`, `gurobipy` (matrix API), `networkx` (max-flow/min-cut),
`numpy`, `matplotlib`, `pytest`; optional `highspy`.

---

## Repository Structure (intended)

```
src/
  generator.py     # Stage 1 — random instance generator (--seed), JSON serialise, toy instance
  direct_mip.py    # Stage 2 — time-indexed MIP baseline (gurobipy matrix API: addMVar, @)
  benders.py       # Stage 3 — disaggregated Benders: master + per-period min-cut subproblem,
                   #            disaggregated optimality cuts via lazy-constraint callback
  benchmark.py     # Stage 4 — run both methods across sizes, emit CSV + log-y chart
run.py             # entry point: `--quick`, `--solver-check`
tests/             # pytest — toy instance with known optimum; MIP and Benders MUST agree
instances/         # generated instance JSON (kept if small/reference; large ones gitignored)
results/           # CSV + chart artefacts (gitignored — regenerate via run.py)
```

---

## Development Commands

Run all Python through `uv` — never bare `python3`.

```bash
uv sync                              # install dependencies
uv run python run.py --quick         # end-to-end demo (~minutes) → comparison chart
uv run python run.py --solver-check  # verify a Gurobi license is available, fail clearly if not
uv run python -m src.generator --seed 0   # generate instances
uv run python -m src.benchmark            # run the benchmark harness

uv run pytest                        # full test suite
uv run pytest tests/test_toy.py      # a single test file
uv run pytest -k "benders"           # tests matching a pattern

uv run ruff check . && uv run ruff format --check .   # lint + format
uv run mypy .                                          # type check
```

Gurobi license comes from the standard license env — never hardcode keys. If absent,
`--solver-check` must fail with a clear, actionable message.

---

## Coding Standards

Full details in `docs/CODING_STANDARDS.md`. Key points:

- Type annotations on every function (params + return); `mypy --strict`; `ruff` clean.
- Formulation files (`direct_mip.py`, `benders.py`) are self-contained and commented **at
  the constraint level**; the module docstring states the math (sets, vars, objective,
  constraints). Readability > cleverness — a non-specialist panel must follow it.
- State any formulation ambiguity you resolve as an inline comment with the assumption.
- No bare `except:`; no `# type: ignore` without a justifying comment.
- Don't over-engineer — this is a demo, not a product.

---

## Testing Requirements

- The toy instance has a hand-checkable known optimum; assert it.
- **Direct MIP and Benders must return the same optimum** on the toy instance and on any
  instance both solve to optimality — this is the core correctness check.
- Every bugfix gets a test that would have caught it.
- Tests pass before any commit/PR.

---

## Git

- Commit **after each build stage** (see the spec's staged deliverables) with a clear
  `<type>: <message>`. After Stage 4, stop and summarise results before attempting Stage 5.
- Branches: `feat/`, `fix/`, `chore/`, `refactor/`, `docs/`, `test/`. See
  `docs/CODING_STANDARDS.md`.

---

## Definition of Done

- [ ] Acceptance criteria from `claudecode-prompt-maintenance-scheduling.md` met
- [ ] `uv run pytest` passes; toy-instance optimum asserted; MIP and Benders agree
- [ ] `uv run ruff check .` and `uv run mypy .` clean — no new errors
- [ ] `README.md` / `docs/` updated if commands or architecture changed
- [ ] Response uses the Output Format below

---

## Output Format

Structure substantive task responses as:

- **PLAN** — what you intend to do and why, before doing it
- **CHANGES** — files modified and the key change in each
- **TESTS** — tests added or updated
- **VALIDATION** — commands run and their pass/fail result
- **RISKS** — anything incomplete, uncertain, or impactful the reviewer should know

---

## What NOT to Do

- Don't change the agreed formulation or master/subproblem split without flagging it first.
- Don't add dependencies beyond the declared set without justification.
- Don't hardcode solver licenses or commit `.env` / `gurobi.lic`.
- Don't commit generated artefacts (`results/`, large instance JSON, charts).
- Don't push directly to `main` once a remote exists.
