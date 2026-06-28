# CPLEX (docplex) direct-MIP and lazy-callback Benders

**Date:** 2026-06-28
**Status:** Approved — ready for planning

## Goal

Add a CPLEX-native reproduction of both solvers, primarily as a **native
lazy-constraint Benders demo** using CPLEX's own `LazyConstraintCallback` (goal B
from brainstorming). OR-Tools only exposes lazy callbacks on SCIP (slow) and
Gurobi (licensed); CPLEX gives a clean, well-documented lazy-callback reference.
Small-instance comparison only; an explicit OR-Tools-vs-CPLEX comparison script
is deferred to a later stage.

## Decisions locked during brainstorming

- **Goal:** mostly (B) native lazy-constraint Benders; (A) small-size comparison
  is a bonus, not the driver.
- **No `backends.py` routing.** CPLEX lives outside the OR-Tools solver-selection
  seam. This is a deliberate, user-approved departure from the CLAUDE.md "model
  through OR-Tools" rule, justified because CPLEX Community Edition is not an
  OR-Tools backend and the goal is to exercise CPLEX's *own* callback machinery.
- **API:** `docplex` (modeling layer), for readability matching the existing
  MathOpt code. Lazy constraints via the documented docplex pattern
  (`ConstraintCallbackMixin` + `cplex.callbacks.LazyConstraintCallback`).
- **Deps:** `docplex` + `cplex` added as **required** deps via `uv add`. pip
  `cplex` is Community Edition (1000 var / 1000 constraint ceiling).
- **Verification:** relaxed for now. No strict OR-Tools agreement test yet.

## Files

**New**
- `src/cplex_mip.py` — direct MIP via docplex.
- `src/cplex_benders.py` — Benders master + native lazy callback.
- `tests/test_cplex.py` — light verification.

**Reused unchanged**
- `src/instance.py` (`Instance`/`Job`/`Arc`), `src/result.py`
  (`Result`/`SolveStatus`), `src/subproblem.py` (`MinCutEvaluator`, `PeriodCut` —
  already backend-agnostic, pure networkx, cut keyed by arc).

**Not touched**
- `src/backends.py`, `run.py`, `src/benchmark.py`.

The small graph helpers (closed-arcs-for-period, jobs-on-arc, full-capacity
max-flow upper bound) are reimplemented locally (~3 lines each) in the CPLEX
modules so each module is self-contained, rather than importing private
`_`-helpers out of `benders.py`.

## `src/cplex_mip.py` — direct MIP

Direct transliteration of the existing MathOpt direct MIP (`_solve_mathopt` in
`src/direct_mip.py`) into docplex, identical formulation so it stays
whiteboard-comparable:

- Vars: `x[j,s]` binary (`model.binary_var`); `f[a,t]` continuous `>= 0` capped at
  `cap_a` (`model.continuous_var`), including the return arc (sink->source) that
  closes the circulation, capacity = total capacity leaving source.
- Constraints:
  - (1) each job scheduled exactly once: `sum_s x[j,s] == 1`.
  - (2) capacity tied to outage: `f[a(j),t] <= cap_{a(j)} * (1 - y[j,t])` where
    `y[j,t]` is the linear in-progress expression (sum of starts in the clamped
    window), big-M free.
  - (3) circulation: flow conservation `inflow == outflow` at every node, every
    period.
  - (4) optional `<=K`: at most K jobs in progress per period.
- Objective: `maximize sum_t f[return arc, t]`.
- Solve: `model.solve()`; map `solve_details` / solution → `Result(method=
  "direct_mip", backend="cplex", ...)` using the same status mapping
  (OPTIMAL / FEASIBLE / INFEASIBLE / UNKNOWN) and gap convention
  (`|primal - dual| / (|primal| + 1e-10)`) as the OR-Tools paths.

## `src/cplex_benders.py` — master + native lazy callback

**Master** mirrors `_build_master_mathopt`:
- `x[j,s]` binary, exactly-once per job.
- `y[a,t]` continuous in `[0,1]` for arcs that carry jobs, with `y <= 1 -
  inprogress_j(t)` for each job j on the arc (matches per-job outage semantics so
  the optimum coincides with the direct MIP).
- `theta[t]` continuous in `[0, ub]`, `ub` = full-capacity max-flow.
- Optional `<=K` lives in the master only.
- Objective: `maximize sum_t theta[t]`.

**Lazy cuts** via the documented docplex pattern:
- A callback class mixing `ConstraintCallbackMixin` + `cplex.callbacks.
  LazyConstraintCallback`, registered with `model.register_callback(...)`.
- At each integer incumbent: read `x` / `theta` values (mixin
  `make_solution_from_vars`), recover the schedule, and for each period run
  `MinCutEvaluator.evaluate(closed_arcs)`. For any period where
  `theta_t > flow_value + eps`, add the disaggregated optimality cut
  `theta_t <= sum_{a in min-cut} cap_a * y[a,t]` (capacity of never-maintained
  arcs folded into the constant), converted via the mixin's `linear_ct_to_cplex`
  and injected with `self.add(...)`.
- This is exactly the cut math the OR-Tools lazy path uses; only the callback
  mechanism differs.
- After solve, recompute the true objective from the final schedule (as the
  OR-Tools lazy path does) and return `Result(method="benders",
  backend="cplex", ...)` with `cut_count` populated.

**Inline caveat:** registering a legacy control callback makes CPLEX disable
dynamic search and run sequentially. Expected; acceptable for a small-instance
mechanism demo.

## Verification (relaxed)

`tests/test_cplex.py` (deps required, so no `importorskip`):
- CPLEX direct-MIP reaches the toy instance's `known_optimum`.
- CPLEX Benders reaches the toy instance's `known_optimum`.
- CPLEX direct-MIP and CPLEX Benders agree with **each other** (objective +
  status) on the toy instance and one small instance. (No OR-Tools cross-check
  yet — deferred to the comparison script.)
- The lazy callback actually fires (`cut_count > 0`) on an instance that needs
  cuts.

Per the project's multi-optimum note, agreement compares objective + status, not
the schedule.

## Risks

- **CE size ceiling (1000 var / 1000 constraint).** The main constraint. Mitigated
  by keeping tests small; catch the CE overflow error and surface a clear message
  rather than a raw CPLEX traceback.
- **Lazy-callback API drift.** The `ConstraintCallbackMixin` pattern is the
  documented one, but verify mechanics on a trivial model first (per the repo's
  "check callbacks cheaply" lesson) before wiring the full subproblem.

## Out of scope

- OR-Tools-vs-CPLEX comparison script / benchmark integration (later stage).
- `run.py` / `benchmark.py` wiring.
- Large-instance runs (blocked by CE ceiling; needs a full CPLEX license).
