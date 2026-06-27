# Claude Code Prompt — Network Maintenance Scheduling Demo (Pearce-Forbes reproduction)

Paste the block below into Claude Code at the root of a fresh repo.

---

## Objective

Build a clean, runnable reproduction of the **network maintenance scheduling problem** (schedule arc-maintenance jobs over a time horizon to maximise total source->sink flow), to serve as an interview demo. The headline result to reproduce: **disaggregated Benders decomposition beats a direct MIP on larger instances.** This mirrors Pearce & Forbes (2019, *JORS* 70(6):941-953; preprint arXiv:1603.02378), built on Boland, Kalinowski, Waterer & Zheng (*Discrete Applied Mathematics* 163, 2014).

This is a tactical analogue of Aurizon's PACE / maintenance-access-window engine for the Central Queensland Coal Network. Keep the code readable and the formulation explicit — it will be explained to a possibly non-specialist panel and whiteboarded from memory.

## Problem definition (implement exactly this)

- Directed capacitated network `G=(N,A)`, single source `s`, single sink `t`, integer arc capacities.
- Discrete time horizon `T = {1,...,H}`.
- A set of maintenance jobs `J`. Each job `j` belongs to an arc `a(j)`, has processing duration `d_j`, and a window `[r_j, dl_j]` of allowed start times. While job `j` is in progress, capacity of arc `a(j)` is **zero** for those periods (full outage; start simple).
- Each job must be scheduled exactly once, contiguously, within its window.
- Optional side constraint (parameterised, default off): at most `K` jobs in progress per time period (the "bounded jobs per period" variant).
- **Objective:** maximise total flow from `s` to `t` summed over all periods `t in T`.

## Deliverables (staged — commit after each)

**Stage 1 — Instance generator** (`src/generator.py`)
- Generate random instances parameterised to match the Pearce-Forbes / Boland regime: 8 networks of increasing size; 10 random job-lists each; 5-15 jobs per arc; durations 10-30; three start-window regimes (tight `1-10`, medium `1-35`, wide `25-35` start times) — the wide regime is the hard one.
- Reproducible via a `--seed`. Serialise instances to JSON. Include a tiny hand-checkable toy instance for unit tests.

**Stage 2 — Direct MIP baseline** (`src/direct_mip.py`)
- Time-indexed formulation in **gurobipy using the matrix API** (`addMVar`, `@` for linear expressions — not `addVar`/`addConstr` loops).
- Binary `x[j,start]` (or `x[j,t]` "in progress"), continuous flow vars per arc per period, flow conservation, capacity tied to outage state, job-scheduling constraints.
- Returns objective, MIP gap, wall time, node count.

**Stage 3 — Disaggregated Benders** (`src/benders.py`)
- **Master**: binary maintenance schedule + one continuous `theta_t` per period bounding that period's flow.
- **Subproblem per period**: given the schedule, max flow on the residual network = an LP, **but solve it as a max-flow / min-cut** (use `networkx.maximum_flow` or a direct push-relabel) rather than calling an LP solver — this is faster and is a deliberate talking point.
- **Cuts**: derive the **disaggregated** Benders optimality cut per period from the min-cut (one cut per period, not one aggregate cut). Inject as **lazy constraints via a Gurobi callback**.
- Implement the LP-relaxation valid inequalities as an optional warm-start step (flag-gated; fine to stub with a clear TODO if time-boxed).
- Returns same metrics as Stage 2 plus cut count and iteration count.

**Stage 4 — Benchmark harness** (`src/benchmark.py`, `run.py`)
- Run both methods across the generated instance sizes with a per-instance time limit.
- Emit a results table (CSV) and a matplotlib chart: **solve time vs instance size, direct MIP vs Benders**, log-y. This chart is the demo centrepiece.
- Print a one-line summary: the instance size at which Benders overtakes direct MIP.

**Stage 5 (optional, only if Stages 1-4 are solid) — solver-swap**
- Abstract the subproblem/relaxation solver behind a thin interface so the LP layer can run on **HiGHS** (`highspy`) instead of Gurobi. Add a tiny comparison of the evaluation-layer solve on Gurobi vs HiGHS. This evidences the "split-solver: Gurobi for the master MIP, free/GPU LP for the evaluation flood" architecture point.

## Tech constraints
- Python 3.11+. Deps: `gurobipy`, `networkx`, `numpy`, `matplotlib`, `pytest`; optional `highspy`.
- Gurobi via standard license env (do NOT hardcode keys). If no license is found, fail with a clear message and a `--solver-check` helper.
- `pyproject.toml` or `requirements.txt`; `README.md` with exact run commands.
- Type hints, docstrings stating the math, `pytest` covering the toy instance (known optimum) for both methods — they MUST agree on the toy instance.
- Keep each formulation file self-contained and heavily commented at the constraint level; readability > cleverness.

## Acceptance criteria
- [ ] `python run.py --quick` runs end-to-end in a couple of minutes and produces the comparison chart.
- [ ] Direct MIP and Benders return the **same optimum** on every instance both solve to optimality.
- [ ] Benders is demonstrably faster on the larger / wide-window instances.
- [ ] `pytest` passes; toy instance optimum is asserted.
- [ ] `README.md` explains the formulation, the master/subproblem split, and the min-cut-as-subproblem insight in a few sentences I can read aloud.

## Working style
- Commit after each stage with a clear message.
- After Stage 4, stop and summarise results before attempting Stage 5.
- Flag any formulation ambiguity you resolve and state the assumption in a comment.
- Do not over-engineer; this is a demo, not a product.
