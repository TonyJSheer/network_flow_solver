# Direct MIP — time-indexed formulation (Stage 2 model)

**Scope:** the *mathematical model* for the direct MIP baseline (`src/direct_mip.py`).
This is the formulation only. Backend wiring (CP-SAT build via `CpModel`/`CpSolver`
vs. MathOpt build for SCIP/HiGHS/Gurobi) is **Stage 2.1** and is deliberately out of
scope here — the model below is backend-agnostic by construction.

Reproduces the time-indexed maintenance-scheduling MIP of Pearce & Forbes (2019),
built on Boland et al. (2014). It is the baseline the disaggregated Benders
decomposition (Stage 3) must match optimum-for-optimum.

## Design goals

- **Whiteboardable:** one binary family, one flow family, four constraint families.
- **Backend-agnostic:** no big-M, no solver-specific constructs; the only
  backend-visible difference is the domain of the flow vars (see below).
- **Decomposition-ready:** the objective and per-period structure map 1:1 onto the
  Benders master, so the "same optimum" cross-check is structurally obvious.

## Sets & data

- `T = {1, …, H}` — discrete periods (horizon `H`).
- `N`, `A` — nodes and directed arcs; single source `s`, single sink `t_sink`.
  (Generated networks are layered DAGs; the model does not rely on that.)
- `cap_a ∈ ℤ₊` — integer capacity of arc `a`.
- `J` — maintenance jobs. Job `j` has arc `a(j)`, duration `d_j`, and an allowed
  **start** window `[r_j, dl_j]` (`release_j`, `deadline_j` in the data model —
  deadline is the latest *start*, not completion).
- `J(a)` — jobs living on arc `a` (5–15 per arc in the Pearce–Forbes regime; jobs
  may overlap in time).
- `K` — optional cap on jobs in progress per period (`max_jobs_per_period`; default
  off / `None`).

### Return arc (circulation)

Add an artificial **return arc** `r = (t_sink, s)`:

- Uncapacitated in effect — capacity `= Σ_{a out of s} cap_a` (an upper bound on any
  feasible throughput), so it never binds.
- Never subject to maintenance.

It closes the network into a pure **circulation**. This is the linchpin for
disaggregation (see Objective and the cross-formulation note).

## Variables

- **`x[j,s] ∈ {0,1}`** for `s ∈ [r_j, dl_j]` — job `j` *starts* in period `s`.
  The window is the index range; contiguity and duration are implicit.
- **`f[a,t] ≥ 0`** for `a ∈ A ∪ {r}`, `t ∈ T` — flow on arc `a` in period `t`.
  - **Integer** under CP-SAT (CP-SAT is integer/Boolean only). Exact here: integral
    arc capacities ⇒ max flow has an integral optimum, so no relaxation is lost.
  - **Continuous** under the MathOpt MIP backends (SCIP/HiGHS/Gurobi).
  - This domain switch is the *only* backend-visible difference and lives entirely
    behind `backends.py`.

### Derived expression (not a variable)

- **`y[j,t]`** — "job `j` is in progress at period `t`":
  `y[j,t] = Σ_{ s = max(r_j, t−d_j+1) … min(dl_j, t) } x[j,s]`.
  A linear expression over the `x`. Because each job starts exactly once, `y[j,t] ∈
  {0,1}` at any integer-feasible point. This expression is the seam reused verbatim
  as the Benders master's schedule.

## Constraints

1. **Schedule exactly once** — for every job `j`:
   `Σ_{s ∈ [r_j, dl_j]} x[j,s] = 1`.
   Window + contiguity come for free from the index range and the `y` definition.

2. **Capacity tied to outage** — per `(job, period)`, big-M-free; for every job `j`
   and period `t`:
   `f[a(j), t] ≤ cap_{a(j)} · (1 − y[j,t])`.
   Any in-progress job on an arc forces that arc's flow to 0 that period. Arcs with
   **no** job (including the return arc `r`) get the plain bound `f[a,t] ≤ cap_a`.

   *Resolved sub-choice:* one constraint per `(job, period)` rather than an aggregate
   per-arc outage binary. It is tighter (the LP relaxation sees `cap·(1−y)` directly),
   needs no extra OR-linking variables, and stays readable even at 5–15 jobs/arc.
   Multiple overlapping jobs on the same arc each independently zero it — correct, if
   mildly redundant.

3. **Flow conservation (circulation)** — for **every** node `n ∈ N` (including `s` and
   `t_sink`, thanks to the return arc) and every period `t`:
   `Σ_{a into n} f[a,t] − Σ_{a out of n} f[a,t] = 0`.
   Uniform over all nodes: every period is an identical self-contained circulation,
   which is exactly what makes per-period disaggregation clean.

4. **(Optional) ≤ K jobs per period** — only when `K` is set; for every period `t`:
   `Σ_{j ∈ J} y[j,t] ≤ K`.

## Objective

Maximise total throughput, read off the return arc:

`max  Σ_{t ∈ T} f[r, t]`.

In a circulation, `f[r,t]` (flow from sink back to source) equals period `t`'s
source→sink throughput by conservation. Measuring it on the single return arc — not
on the sink-incoming real arcs — is the clean expression.

## Cross-formulation note (why the return arc matters)

`f[r,t]` is exactly the quantity the Benders master's per-period variable `θ_t`
bounds. The disaggregated master is `max Σ_t θ_t`; each disaggregated optimality cut
bounds `θ_t ≤ (period-t min-cut capacity under the candidate schedule)`. So the direct
MIP's objective term and the master's variable share one shape, and "direct MIP and
Benders return the same optimum" is structurally, not coincidentally, true. Each
period is coupled to the others *only* through the shared `x` — the structure Stage 3
exploits.

## Returns (Result record)

`objective` (= `Σ_t f[r,t]`), `status`, `schedule` (job → start period, recovered from
`x`), `wall_time_s`, `gap`, and `node_count` where the backend exposes it. Per the
shared `Result` dataclass.

## Out of scope (later stages)

- CP-SAT vs MathOpt build paths and the integer/continuous flow-var switch — **Stage
  2.1**.
- Intervals / `add_no_overlap` / cumulative as an alternative CP-SAT encoding of the
  schedule and the ≤K constraint — Stage 2.1 may explore, but the plain time-indexed
  binary form above is the agreed model.
- Benders master/subproblem and cut derivation — Stage 3.
