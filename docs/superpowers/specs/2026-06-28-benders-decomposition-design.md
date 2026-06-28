# Disaggregated Benders decomposition (Stage 3)

**Scope:** the design for `src/benders.py` — disaggregated Benders for the maintenance
scheduling problem, reproducing Pearce & Forbes (2019, arXiv:1603.02378), built on
Boland et al. (2014). It must match the direct MIP (`src/direct_mip.py`) optimum-for-optimum
on every instance both solve to optimality, and agree across backends.

Stage 3 is split into **three strategy sub-stages**, each a different way to solve the
per-period subproblem and generate the *same* disaggregated optimality cut:

- **3a — Analytic min-cut (networkx).** Builds all shared infra; the headline talking point.
- **3b — LP-dual (MathOpt).** Solves the subproblem as an LP and reads capacity duals.
- **3c — Pareto (combinatorial two-cut pick).** Selects a non-dominated min-cut.

3b and 3c both depend only on 3a (3c is combinatorial — it does **not** depend on 3b).

## The decomposition

Fix a schedule. The objective `Σ_t flow_t` separates by period: each period is an
independent **s→t max-flow** on the network with arcs-under-maintenance set to capacity 0.
Benders approximates each period's flow with a master variable `θ_t` and tightens it with
optimality cuts derived from the subproblem. Because the subproblem is a max-flow it is
always feasible (flow 0), so there are **optimality cuts only — no feasibility cuts**.

### Master problem

Variables:
- `x[j,s] ∈ {0,1}` — job `j` starts in period `s`, `s ∈ [r_j, dl_j]` (as in the direct MIP).
- `y[a,t] ∈ [0,1]` — **arc-availability proxy**: 1 iff arc `a` is open (no job in progress) at `t`.
- `θ_t ≥ 0` — per-period flow proxy, `θ_t ≤ UB_t`.

Derived (linear in `x`, identical to the direct MIP):
- `inprogress[j,t] = Σ_{s = max(r_j, t−d_j+1)}^{min(dl_j, t)} x[j,s]` — 1 iff job `j` in progress at `t`.

Constraints:
1. `Σ_s x[j,s] = 1` — each job scheduled exactly once.
2. `y[a,t] ≤ 1 − inprogress[j,t]` for **each** job `j ∈ J(a)` — arc open only if no job on it
   is in progress. This matches the direct MIP's per-job capacity semantics under **overlapping
   same-arc jobs** (the generator allows them; a single `1 − Σ inprogress` would go negative and
   produce invalid cuts). `y[a,t]` needs no integrality or lower bound: the maximize objective
   pulls it to exactly `1 − max_j inprogress[j,t]`. Arcs with no jobs have `y[a,t] ≡ 1` (constant).
3. `Σ_{j} inprogress[j,t] ≤ K` for each `t` — optional bounded jobs per period (master-only;
   the subproblem never sees `K`).
4. `θ_t ≤ UB_t` — `UB_t` = full-capacity max-flow (same every period), computed once up front so
   the master is bounded before any cut is added.

Objective: `max Σ_t θ_t`.

The master is the direct MIP **minus** flow vars and conservation, **plus** `y[a,t]` and `θ_t`.
Per CLAUDE.md "formulation files self-contained", the scheduling constraints (1)/(3) and the
`inprogress` expression are **duplicated** from `direct_mip.py` rather than shared; the
cross-method agreement tests guard against drift. Two API paths as in the direct MIP:
`cp_model` for cp-sat, `mathopt` for scip/highs/gurobi.

### Subproblem (per period, given schedule `x̂`)

Build the period-`t` network: each arc `a` with capacity `cap_a` if open at `t`, else 0
(open = no job on `a` in progress at `t`, read from `x̂`). Compute the **s→t max-flow**. No
return arc — that was a circulation device for the direct MIP; here it is plain s–t flow.

### Disaggregated optimality cut

The LP dual of max-flow is min-cut. From a min-cut `C` (a node partition `(S, V\S)`, `s∈S`,
`t∉S`; the cut arcs are those crossing `S→V\S`), the period-`t` flow under **any** schedule is
bounded by the cut's available capacity:

```
θ_t  ≤  Σ_{a ∈ C}  cap_a · y[a,t]
```

linear in the master's `y`. One cut per period = **disaggregated** (summing them would be the
weaker aggregated cut we deliberately avoid). Arcs in `C` with no job contribute the constant
`cap_a`. This is exactly Pearce & Forbes MP1 with the dual `u_at` specialised to the min-cut
0/1 indicator.

## The three strategies (evaluator seam)

All three plug into one seam. An **evaluator** maps a period's outage to a flow value and a cut:

```
evaluate(period_t, outage_vector) -> (flow_value: int, cut_coeffs: dict[arc, capacity])
```

The evaluator is ignorant of master variables — it returns cut **coefficients keyed by arc**;
the master turns them into `θ_t ≤ Σ_a cap_a · y[a,t]`. This is the only seam 3b/3c touch.

- **3a — Analytic min-cut (networkx).** `networkx.maximum_flow` → residual graph → `S` = nodes
  reachable from `s` in the residual → cut arcs = original arcs crossing `(S, V\S)`. 0/1 dual,
  no LP in the loop. For max-flow this *is* the analytic cut (min-cut = closed-form dual),
  exactly as Dijkstra node-potentials are the analytic dual of a shortest path.
- **3b — LP-dual (MathOpt).** Solve the max-flow as an LP (conservation + `f_a ≤ cap_a·open_a`)
  and read the duals of the capacity constraints; the cut arcs are those with non-zero dual.
  Same cut family, proves dual = min-cut, and is the speed comparison. **Solved on a dedicated
  fast LP (GLOP/HiGHS) regardless of the master backend** — this exercises the "open MIP master +
  free fast LP for the evaluation flood" split-solver architecture point.
- **3c — Pareto (combinatorial two-cut pick).** `networkx` yields only the *source-side minimal*
  min-cut; max-flow min-cuts are usually **non-unique**, and the choice changes cut strength.
  Compute both the source-side cut (`S` = reachable from `s` in residual) and the **sink-side**
  cut (`V\S'` where `S'` = nodes that can reach `t` in the residual), then emit whichever has the
  tighter RHS at a **core point** `ẑ`. `ẑ` is maintained as the running average of incumbent
  outage vectors (`ẑ ← (ẑ + outage)/2`, init 0.5), mirroring the `ZCore` Magnanti–Wong pattern.
  Combinatorial, so it depends on 3a only.

## Cut injection (both built in 3a)

| Backend | `supports_lazy` | Injection path |
|---|---|---|
| cp-sat (primary) | False | **iterative loop** (only path available) |
| cp-sat-m | False | iterative loop |
| highs | False | iterative loop |
| scip | True | **lazy callback** |
| gurobi (if licensed) | True | lazy callback |

`Backend.supports_lazy` flips to `True` for scip/gurobi (currently `False` pending this stage).

- **Iterative loop** (required — the only path for the primary cp-sat backend): solve master to
  optimality → for each `t` evaluate `flow_t(x̂)` → add cuts wherever `θ_t > flow_t + ε` → re-solve.
  Stop when no cut is violated. Finite (finitely many distinct min-cuts); at termination
  `Σθ_t = Σflow_t` = the true optimum.
- **Lazy callback** (scip/gurobi via `mathopt.solve(..., callback_reg=CallbackRegistration(
  events={Event.MIP_SOLUTION}, add_lazy_constraints=True), cb=...)`): at each integer incumbent,
  evaluate all periods and `add_lazy_constraint` for each violated `θ_t`.

Both paths share the same evaluator and the same cut-building code; only the injection differs.

## Warm-start / speedups (3a, flag-gated)

- **Bottleneck pre-cuts (§3.1).** The analytic cut applied to the **full-capacity** network up
  front, peeling successive min-cut-sets (raise the cut arcs' caps, re-solve max-flow, repeat
  until the trivial s-out cut binds). Seeds the master with a tight bound. Optional `--pre-cuts`.
- **Config caching.** Cache `(flow_value, cut)` keyed by the per-period outage vector; identical
  outage patterns are recalled, not re-solved. Report **distinct flows solved** vs total.
- **LP-relaxation warm start (LP-R).** Out of scope for the base form; documented TODO.

## Metrics & result

Reuse `Result` (already carries `iteration_count`, `cut_count`). Benders fills:
`objective`, `status`, `wall_time_s`, `gap`, `schedule`, `iteration_count`, `cut_count`.
Add the cache stat (distinct vs total subproblem solves) to the stage log / benchmark, not the
shared `Result` dataclass (keep it a superset, not per-strategy bloat).

## Testing (TDD; the toy instance is a known-answer test)

- **Toy known optimum:** Benders returns the hand-checked toy optimum.
- **Cross-method agreement:** Benders optimum == direct MIP optimum on the toy and on any
  instance both solve to optimality. Compare `(objective, status)`, **not** the schedule
  (multi-optimum: distinct schedules can share the optimum).
- **Cross-strategy agreement:** 3a, 3b, 3c return the same optimum.
- **Cross-backend agreement:** same optimum across cp-sat / scip / highs (loop and lazy paths).
- **Cut validity (regression):** every added cut is a valid upper bound — no cut ever excludes
  the known optimal `Σθ_t`.
- **Convergence:** the iterative loop terminates and the final `Σθ_t` equals `Σ flow_t(x̂)`.

## Resolved assumptions

- **Overlapping same-arc jobs** are allowed (generator regime). Handled by the per-job
  `y[a,t] ≤ 1 − inprogress[j,t]` constraints, matching `direct_mip`'s per-job capacity semantics.
- **`K` is master-only** — it couples the schedule across arcs within a period and never enters
  the max-flow subproblem.
- **Flow integrality:** integral capacities give an integral max-flow; the subproblem value is
  exact regardless of evaluator. cp-sat's integer flow and MathOpt's continuous flow agree.
- **Pareto "stronger":** defined as smaller cut RHS at the running-average core point `ẑ`; both
  extreme cuts are valid, so this only ever picks among valid cuts.

## Risks

- **Lazy callback semantics in MathOpt** (per-solver quirks for SCIP) — verify with a trivial
  model in 3a before wiring the real cut.
- **Master `y[a,t]` count** is `|arcs with jobs| × H`; fine at demo sizes, watch on the widest
  instances.
- **Loop iteration blow-up** on wide-window instances — config caching and pre-cuts mitigate;
  the lazy path avoids re-solving the master from scratch.

## Sub-stage dependency order

```
3a (analytic min-cut + master + loop + lazy + caching)
 ├── 3b (LP-dual evaluator, dedicated fast LP)
 └── 3c (Pareto two-cut pick, combinatorial)
```
