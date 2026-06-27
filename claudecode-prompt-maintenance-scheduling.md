# Claude Code Prompt — Network Maintenance Scheduling Demo (Pearce-Forbes reproduction)

Paste the block below into Claude Code at the root of a fresh repo.

---

## Objective

Build a clean, runnable reproduction of the **network maintenance scheduling problem** (schedule arc-maintenance jobs over a time horizon to maximise total source->sink flow), to serve as an interview demo. The headline result to reproduce: **disaggregated Benders decomposition beats a direct MIP on larger instances.** This mirrors Pearce & Forbes (2019, *JORS* 70(6):941-953; preprint arXiv:1603.02378), built on Boland, Kalinowski, Waterer & Zheng (*Discrete Applied Mathematics* 163, 2014).

This is a tactical analogue of Aurizon's PACE / maintenance-access-window engine for the Central Queensland Coal Network. Keep the code readable and the formulation explicit — it will be explained to a possibly non-specialist panel and whiteboarded from memory.

**Solver stack: Google OR-Tools.** Model through OR-Tools and treat the underlying solver as a swappable backend. Two OR-Tools APIs are in scope:
- **CP-SAT** (`ortools.sat.python.cp_model`, `CpModel`/`CpSolver`) — the integer constraint solver. Default/primary backend.
- **MathOpt** (`ortools.math_opt.python`) — the unified interface that fronts **SCIP**, **HiGHS**, and (if licensed) **Gurobi**, plus GLOP for pure LPs. MathOpt is the recommended modern interface (it supersedes the older `linear_solver`/`pywraplp` MPSolver layer); prefer it for the non-CP-SAT backends.

Backends to support/explore: **CP-SAT, SCIP, HiGHS**, with **Gurobi optional** (OR-Tools can route to Gurobi as a MathOpt backend, but it needs a license, so never make it a hard dependency). All backends should be reachable through one thin solver-selection flag — switching backend must not require touching the formulation.

**Modelling caveat to handle explicitly:** CP-SAT is integer/Boolean only, so the per-period flow variables must be **integer** under CP-SAT. That is exact here — arc capacities are integral, so max flow has an integral optimum — but note it in a comment. The MathOpt MIP backends (SCIP/HiGHS/Gurobi) can use **continuous** flow variables. Keep the two cases behind the solver interface so the rest of the code is backend-agnostic.

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
- Time-indexed formulation built through OR-Tools, behind a **thin solver-selection interface** (`--backend {cp-sat,scip,highs,gurobi}`) so the same model runs on any backend.
- Binary `x[j,start]` (or `x[j,t]` "in progress"), flow vars per arc per period, flow conservation, capacity tied to outage state, job-scheduling constraints.
- **CP-SAT path** (`ortools.sat.python.cp_model`): flow vars are integer (exact here — integral capacities). Idiomatic CP-SAT — build with `CpModel`, solve with `CpSolver`; intervals (`new_optional_interval_var` / `new_interval_var`) are a natural fit for the contiguous job windows and `add_no_overlap` / cumulative for the "≤K jobs per period" side constraint, but a plain time-indexed binary formulation is fine too — pick one and comment why.
- **MathOpt path** (`ortools.math_opt.python`): same structure with continuous flow vars; select SCIP / HiGHS / Gurobi via the MathOpt `SolverType`.
- Returns objective, gap, wall time, and node/branch count where the backend exposes it.

**Stage 3 — Disaggregated Benders** (`src/benders.py`)
- **Master**: binary maintenance schedule + one continuous `theta_t` per period bounding that period's flow.
- **Subproblem per period**: given the schedule, max flow on the residual network = an LP, **but solve it as a max-flow / min-cut** (use `networkx.maximum_flow` or a direct push-relabel) rather than calling an LP solver — this is faster and is a deliberate talking point.
- **Cuts**: derive the **disaggregated** Benders optimality cut per period from the min-cut (one cut per period, not one aggregate cut). Inject as **lazy constraints via the OR-Tools callback mechanism** (MathOpt callback registration requesting lazy-constraint events).
- **Backend restriction:** lazy constraints during search are only available on backends that support them — run Benders only on those (Gurobi natively; check whether the OR-Tools/MathOpt interface exposes lazy-constraint callbacks for SCIP). On backends without callback lazy-constraint support (and on CP-SAT), fall back to an **iterative cut loop**: re-solve the master, add the disaggregated cuts as ordinary constraints, repeat to convergence. Pick the lazy path where the interface exposes it and the loop otherwise; gate this on the `--backend` selection and state the assumption in a comment.
- Implement the LP-relaxation valid inequalities as an optional warm-start step (flag-gated; fine to stub with a clear TODO if time-boxed).
- Returns same metrics as Stage 2 plus cut count and iteration count.

**Stage 4 — Benchmark harness** (`src/benchmark.py`, `run.py`)
- Run both methods across the generated instance sizes with a per-instance time limit, parameterised by `--backend` so a single run can sweep the available backends (CP-SAT, SCIP, HiGHS, and Gurobi if licensed).
- Emit a results table (CSV) and a matplotlib chart: **solve time vs instance size, direct MIP vs Benders**, log-y. This chart is the demo centrepiece. Where more than one backend is available, also chart **direct MIP across backends** (CP-SAT vs SCIP vs HiGHS) so the backend trade-off is visible.
- Print a one-line summary: the instance size at which Benders overtakes direct MIP.

**Stage 5 (optional, only if Stages 1-4 are solid) — backend comparison matrix**
- The solver-selection interface from Stages 2-3 already abstracts the backend; here, exercise it as a comparison matrix: tabulate the evaluation/subproblem and master solves across **CP-SAT, SCIP, HiGHS** (and Gurobi if licensed) on a fixed instance set. This evidences the "split-solver: open MIP backend for the master, free/fast LP for the evaluation flood" architecture point — and shows the demo is not tied to any single (licensed) solver.

## Tech constraints
- Python 3.12+. Deps: `ortools` (provides CP-SAT, MathOpt, and the bundled SCIP/HiGHS backends), `networkx`, `numpy`, `matplotlib`, `pytest`. Gurobi is **optional** and only used if a license is present — never a hard dependency.
- `--solver-check` helper that reports which backends are actually available at runtime (CP-SAT and the MathOpt-bundled SCIP/HiGHS should always be; Gurobi only if licensed) and fails clearly if a requested backend is missing. Never hardcode any solver license/keys.
- `pyproject.toml` or `requirements.txt`; `README.md` with exact run commands.
- Type hints, docstrings stating the math, `pytest` covering the toy instance (known optimum) for both methods — they MUST agree on the toy instance, and results MUST agree across backends.
- Keep each formulation file self-contained and heavily commented at the constraint level; readability > cleverness. The backend choice lives behind one thin interface — formulation code stays backend-agnostic.

## Acceptance criteria
- [ ] `python run.py --quick` runs end-to-end in a couple of minutes and produces the comparison chart, using a license-free backend (CP-SAT or MathOpt/SCIP/HiGHS) by default.
- [ ] Direct MIP and Benders return the **same optimum** on every instance both solve to optimality — and the optima agree across backends.
- [ ] Benders is demonstrably faster on the larger / wide-window instances.
- [ ] `pytest` passes; toy instance optimum is asserted.
- [ ] `--solver-check` reports available backends correctly and the demo runs without a Gurobi license.
- [ ] `README.md` explains the formulation, the master/subproblem split, the min-cut-as-subproblem insight, and the backend-abstraction (OR-Tools CP-SAT + MathOpt) in a few sentences I can read aloud.

## Working style
- Commit after each stage with a clear message.
- After Stage 4, stop and summarise results before attempting Stage 5.
- Flag any formulation ambiguity you resolve and state the assumption in a comment.
- Do not over-engineer; this is a demo, not a product.
