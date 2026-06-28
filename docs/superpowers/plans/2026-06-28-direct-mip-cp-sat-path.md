# Direct MIP — native CP-SAT showcase build (Stage 2.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a native CP-SAT build path for the direct MIP that puts the CP-SAT scheduling engine on display — distinct from, and stronger than, the `cp-sat-m` transliteration — while returning the same optimum as every other backend.

**Architecture:** `solve_direct_mip` already routes by `backend.family`; today the CP-SAT branch raises `NotImplementedError`. We add `_solve_cp_sat`, a sibling of `_solve_mathopt`, that builds the *same problem* through `ortools.sat.python.cp_model` but with CP-native constructs: **presence literals `x[j,s]`** (an exact, reification-free coverage backbone) feeding **optional interval vars**, `add_exactly_one` for "scheduled once", **enforcement-literal outage** (`f==0 .only_enforce_if(x[j,s])` — big-M free, hard propagation) instead of `cap·(1−y)`, and **`add_cumulative`** for the optional `≤K`. Flow vars are **integer** (exact: integral capacities ⇒ integral max flow).

**Tech Stack:** Python 3.12, `ortools` (CP-SAT via `ortools.sat.python.cp_model`), `pytest`. Run everything through `uv`.

## Design rationale (why this model, not the obvious one)

Decided with the user. The jobs in this problem **don't contend** (each independently zeroes its own arc; overlaps allowed) and `≤K` defaults **off**, so CP-SAT's `no_overlap`/`cumulative` propagators have little to bite on by themselves. CP-SAT's real edge here is **enforcement-literal propagation** (hard `f==0`, no LP-relaxation slack). The "obvious" maximal-CP model — one integer `start_j` per job — would force a *two-sided* reification of `active[j,t] ⟺ start_j ≤ t ≤ start_j+d_j−1` (half-reification is unsafe: the flow-maximising objective would otherwise leave arcs up during maintenance), costing ~2 aux booleans + 6 clauses per `(job, period)` and propagating *worse* than the time-indexed form. Keeping the `x[j,s]` presence literals gives an exact coverage channel for free and lets the enforcement literals and `add_cumulative` do the CP-native work cleanly. This also preserves the wide-window binary-explosion scaling, so `cp-sat` can be a showcase line without quietly undermining the "Benders beats the direct MIP" headline (which stays anchored on the MathOpt MIP backends / `cp-sat-m`).

## Global Constraints

- Run Python only through `uv` — never bare `python3`.
- `mypy --strict` clean; `ruff check` and `ruff format --check` clean. No `# type: ignore` without a justifying comment; no bare `except`.
- Type annotations on every function.
- Backend identity stays in `backends.py`; `direct_mip.py` branches only on the existing `backend.family` seam and never names a solver license/product. The native CP build is deliberately a *different formulation* from the MathOpt path (sanctioned by the spec: "intervals are a natural fit… pick one and comment why") — comment why at the top of `_solve_cp_sat`.
- Flow vars are **integer under CP-SAT** — annotate with the exactness justification (integral capacities ⇒ integral max flow).
- Single-threaded for reproducible timing: `solver.parameters.num_workers = NUM_THREADS` (defined in `backends.py`, value 1).
- Direct MIP results MUST agree across backends (cp-sat / cp-sat-m / scip / highs) on the toy instance and any instance solved to optimality. Compare objective + status, not the schedule (multiple optima).
- Period indices are 1-based (`T = {1..H}`); job windows are 1-based, matching the existing model and instance validation (`deadline + duration − 1 ≤ horizon`).
- Reporting convention for the final summary: PLAN / CHANGES / TESTS / VALIDATION / RISKS.

## Confirmed CP-SAT API surface (verified against installed ortools)

- `from ortools.sat.python import cp_model`
- `model = cp_model.CpModel()`
- `model.new_int_var(lb: int, ub: int, name) -> cp_model.IntVar`
- `model.new_bool_var(name) -> cp_model.IntVar`
- `model.new_optional_fixed_size_interval_var(start, size: int, is_present, name) -> cp_model.IntervalVar`
- `model.add_exactly_one(literals)`
- `model.add(bounded_expr)` → returns a constraint with `.only_enforce_if(lit)`
- `model.add_cumulative(intervals, demands, capacity)` — global propagator; with `capacity=0` and a present demand-1 interval it yields INFEASIBLE (verified).
- `model.maximize(linear_expr)`; `cp_model.LinearExpr.sum(iterable) -> LinearExpr` (LinearExpr even for empty).
- `solver = cp_model.CpSolver()`; `solver.parameters.num_workers: int`, `solver.parameters.max_time_in_seconds: float`
- `status = solver.solve(model)` → `cp_model.CpSolverStatus`; compare to `cp_model.OPTIMAL/FEASIBLE/INFEASIBLE/UNKNOWN/MODEL_INVALID`
- Post-solve: `solver.value(var) -> int`, `solver.objective_value: float`, `solver.best_objective_bound: float`, `solver.wall_time: float`, `solver.num_branches: int` (→ `node_count`).
- Types for annotations: `cp_model.IntVar`, `cp_model.IntervalVar`, `cp_model.LinearExprT`, `cp_model.CpSolverStatus`.

## File Structure

- Modify `src/direct_mip.py` — add the `cp_model` import, route the CP-SAT family in `solve_direct_mip`, add `_solve_cp_sat` + `_to_result_cp_sat`, update the module docstring. **`_solve_mathopt` is untouched.**
- Modify `tests/test_direct_mip.py` — add native-CP-SAT optimum + K=0 infeasibility tests, add `cp-sat` to the agreement parametrizations, and replace the "deferred to Stage 2.1" test.

No new files; `backends.py` is unchanged (the `cp-sat` Backend entry with `continuous_flow=False` already exists).

---

### Task 1: Native CP-SAT showcase build

**Files:**
- Modify: `src/direct_mip.py`
- Test: `tests/test_direct_mip.py`

**Interfaces:**
- Consumes: `NUM_THREADS`, `ApiFamily`, `Backend` from `src.backends`; `Instance`; `Result`, `SolveStatus`.
- Produces: `_solve_cp_sat(instance: Instance, backend: Backend, time_limit_s: float | None) -> Result`; `solve_direct_mip(instance, backend, time_limit_s=None)` now returns a real `Result` for `cp-sat` instead of raising.

- [ ] **Step 1: Write the failing tests**

In `tests/test_direct_mip.py`: add the two native-CP-SAT tests below, extend both agreement checks to include `cp-sat`, and **delete** `test_native_cp_sat_is_deferred_to_stage_2_1`.

```python
def test_toy_optimum_on_native_cp_sat() -> None:
    inst = toy_instance()
    res = solve_direct_mip(inst, resolve("cp-sat"))

    assert res.method == "direct_mip"
    assert res.backend == "cp-sat"
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
    assert res.schedule is not None
    assert res.schedule["j0"] in range(1, 6)
    assert res.node_count is not None
    assert res.wall_time_s >= 0.0


def test_native_cp_sat_k_zero_infeasible() -> None:
    # Every job must be scheduled (exactly-one) => in progress somewhere; the
    # cumulative with capacity 0 then has a present demand-1 interval => infeasible.
    inst = replace(toy_instance(), max_jobs_per_period=0)
    res = solve_direct_mip(inst, resolve("cp-sat"))
    assert res.status is SolveStatus.INFEASIBLE
    assert res.objective is None
```

Change the toy-agreement parametrize to include native CP-SAT:

```python
@pytest.mark.parametrize("backend_name", ["highs", "scip", "cp-sat-m", "cp-sat"])
def test_backends_agree_on_toy(backend_name: str) -> None:
    res = solve_direct_mip(toy_instance(), resolve(backend_name))
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
```

Extend the generated-instance agreement loop so all four backends must match:

```python
def test_backends_agree_on_generated_instance() -> None:
    inst = generate_instance(size_idx=1, list_idx=0, regime=Regime.TIGHT, seed=7)
    objectives: list[float] = []
    for name in ("highs", "scip", "cp-sat-m", "cp-sat"):
        res = solve_direct_mip(inst, resolve(name), time_limit_s=60.0)
        assert res.status is SolveStatus.OPTIMAL
        assert res.objective is not None
        objectives.append(res.objective)
    first = objectives[0]
    for obj in objectives[1:]:
        assert obj == pytest.approx(first)
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_direct_mip.py -q`
Expected: FAIL — the native-CP-SAT tests and the `cp-sat` parametrization hit `NotImplementedError` ("CP-SAT integer build is Stage 2.1…").

- [ ] **Step 3: Route the CP-SAT family**

In `src/direct_mip.py`, replace the body of `solve_direct_mip` so it dispatches instead of raising:

```python
def solve_direct_mip(
    instance: Instance,
    backend: Backend,
    time_limit_s: float | None = None,
) -> Result:
    """Solve the time-indexed direct MIP on the selected backend.

    Same problem on every backend. The MathOpt backends build a continuous-flow
    LP/MIP transliteration (_solve_mathopt); the native CP-SAT backend builds a
    CP-idiomatic model with integer flow vars (_solve_cp_sat).
    """
    if backend.family is ApiFamily.CP_SAT:
        return _solve_cp_sat(instance, backend, time_limit_s)
    return _solve_mathopt(instance, backend, time_limit_s)
```

- [ ] **Step 4: Add the `cp_model` import**

Add to the imports block of `src/direct_mip.py` (after the existing `from ortools.math_opt.python import mathopt`):

```python
from ortools.sat.python import cp_model
```

- [ ] **Step 5: Add the native CP-SAT build**

Add `_solve_cp_sat` to `src/direct_mip.py` (sibling of `_solve_mathopt`):

```python
def _solve_cp_sat(instance: Instance, backend: Backend, time_limit_s: float | None) -> Result:
    # Native CP-SAT showcase build — deliberately a different formulation from the
    # MathOpt transliteration (which is also reachable as the cp-sat-m backend).
    # It uses the CP-SAT scheduling engine: presence literals + optional interval
    # vars, add_exactly_one, enforcement-literal outage (big-M free), and
    # add_cumulative for the optional <=K. Flow vars are INTEGER: CP-SAT is
    # integer/Boolean only, which is exact here because integral arc capacities
    # give an integral max flow, so no relaxation is lost vs the continuous build.
    periods = range(1, instance.horizon + 1)
    model = cp_model.CpModel()

    # Return arc (sink -> source) closes the network into a circulation; its
    # capacity bounds any feasible throughput so it never binds. Reserved key.
    return_cap = sum(a.capacity for a in instance.arcs if a.u == instance.source)
    arc_caps: dict[tuple[str, str], int] = {(a.u, a.v): a.capacity for a in instance.arcs}
    if (instance.sink, instance.source) in arc_caps:
        raise ValueError(
            "instance declares a real (sink, source) arc; that key is reserved for the return arc"
        )
    arc_caps[(instance.sink, instance.source)] = return_cap

    # Integer flow vars for every arc (incl. return arc), every period.
    f: dict[tuple[str, str, int], cp_model.IntVar] = {}
    for (u, v), cap in arc_caps.items():
        for t in periods:
            f[(u, v, t)] = model.new_int_var(0, cap, f"f[{u},{v},{t}]")

    # Presence literals x[j,s] over each job's start window. Optional interval
    # vars (one per candidate start) are built only when <=K is active, since the
    # cumulative propagator is their only consumer here.
    x: dict[tuple[str, int], cp_model.IntVar] = {}
    starts: dict[str, range] = {}
    job_intervals: list[cp_model.IntervalVar] = []
    use_cumulative = instance.max_jobs_per_period is not None
    for j in instance.jobs:
        starts[j.id] = range(j.release, j.deadline + 1)
        literals: list[cp_model.IntVar] = []
        for s in starts[j.id]:
            lit = model.new_bool_var(f"x[{j.id},{s}]")
            x[(j.id, s)] = lit
            literals.append(lit)
            if use_cumulative:
                job_intervals.append(
                    model.new_optional_fixed_size_interval_var(
                        start=s, size=j.duration, is_present=lit, name=f"itv[{j.id},{s}]"
                    )
                )
        # (1) scheduled exactly once — a native exactly-one clause.
        model.add_exactly_one(literals)

    # (2) capacity tied to outage via enforcement literals (big-M free): if job j
    # starts at s, its arc is fully down for periods [s, s+d-1]. Exactly one start
    # is chosen per job, so exactly the right periods are zeroed.
    for j in instance.jobs:
        for s in starts[j.id]:
            for t in range(s, s + j.duration):
                model.add(f[(j.arc[0], j.arc[1], t)] == 0).only_enforce_if(x[(j.id, s)])

    # (3) circulation: flow conservation at every node (return arc balances s/t).
    for n in instance.nodes:
        for t in periods:
            inflow = cp_model.LinearExpr.sum([f[(u, v, t)] for (u, v) in arc_caps if v == n])
            outflow = cp_model.LinearExpr.sum([f[(u, v, t)] for (u, v) in arc_caps if u == n])
            model.add(inflow == outflow)

    # (4) optional <=K jobs per period via the cumulative global propagator.
    if use_cumulative:
        cap_k = instance.max_jobs_per_period
        assert cap_k is not None  # narrowed by use_cumulative
        model.add_cumulative(job_intervals, [1] * len(job_intervals), cap_k)

    # Objective: maximise total return-arc flow = total source->sink throughput.
    ret = (instance.sink, instance.source)
    model.maximize(cp_model.LinearExpr.sum([f[(ret[0], ret[1], t)] for t in periods]))

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = NUM_THREADS  # single-threaded for fair timing
    if time_limit_s is not None:
        solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.solve(model)
    return _to_result_cp_sat(backend, solver, status, x, starts)
```

- [ ] **Step 6: Add the CP-SAT result translation**

Add `_to_result_cp_sat` to `src/direct_mip.py`:

```python
def _to_result_cp_sat(
    backend: Backend,
    solver: cp_model.CpSolver,
    status: cp_model.CpSolverStatus,
    x: dict[tuple[str, int], cp_model.IntVar],
    starts: dict[str, range],
) -> Result:
    if status == cp_model.OPTIMAL:
        solve_status = SolveStatus.OPTIMAL
    elif status == cp_model.FEASIBLE:
        solve_status = SolveStatus.FEASIBLE  # incumbent, not proven optimal
    elif status == cp_model.INFEASIBLE:
        solve_status = SolveStatus.INFEASIBLE
    else:
        solve_status = SolveStatus.UNKNOWN  # incl. MODEL_INVALID / no incumbent

    wall = solver.wall_time
    node_count = solver.num_branches

    has_incumbent = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    if not has_incumbent:
        return Result(
            method="direct_mip",
            backend=backend.name,
            status=solve_status,
            objective=None,
            wall_time_s=wall,
            node_count=node_count,
        )

    schedule: dict[str, int] = {}
    for job_id, window in starts.items():
        for s in window:
            if solver.value(x[(job_id, s)]) > 0.5:
                schedule[job_id] = s
                break

    primal = solver.objective_value
    dual = solver.best_objective_bound
    # gap denominates by incumbent |primal| for divide-by-zero safety (matches MathOpt path)
    gap = abs(primal - dual) / (abs(primal) + 1e-10)
    return Result(
        method="direct_mip",
        backend=backend.name,
        status=solve_status,
        objective=primal,
        wall_time_s=wall,
        gap=gap,
        schedule=schedule,
        node_count=node_count,
    )
```

- [ ] **Step 7: Update the module docstring**

In `src/direct_mip.py`, replace the closing "Backend note" paragraph so it reflects the native CP build:

```python
Backend note: the MathOpt backends build the continuous-flow transliteration above
(also reachable as the cp-sat-m backend). The native CP-SAT backend (_solve_cp_sat)
builds a CP-idiomatic model — presence literals + optional intervals, exactly-one,
enforcement-literal outage, and add_cumulative for <=K — with integer flow vars.
Integer is exact here (integral capacities give an integral max flow), so every
backend reaches the same optimum; the cross-backend agreement tests enforce it.
```

- [ ] **Step 8: Run the direct-MIP suite to verify it passes**

Run: `uv run pytest tests/test_direct_mip.py -q`
Expected: PASS — native-CP-SAT optimum (8.0), K=0 INFEASIBLE, and all four backends agreeing on toy + generated instance.

- [ ] **Step 9: Run the full suite + lint + typecheck**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy .`
Expected: all PASS / clean. (If `ruff format --check` flags `direct_mip.py`, run `uv run ruff format src/direct_mip.py` and re-stage.)

- [ ] **Step 10: Commit**

```bash
git add src/direct_mip.py tests/test_direct_mip.py
git commit -m "feat: native CP-SAT showcase build for direct MIP (Stage 2.1)

Presence literals + optional intervals, add_exactly_one, enforcement-literal
outage (big-M free), and add_cumulative for <=K, with integer flow vars
(exact: integral capacities give an integral max flow). Distinct from the
cp-sat-m transliteration; cp-sat now agrees with highs/scip/cp-sat-m on the
toy and generated instances.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage (direct-MIP formulation design spec + CLAUDE.md Stage 2.1 + user direction "CP-SAT at its strongest"):**
- Native CP-SAT build via `cp_model`, integer flow vars, exactness comment — Steps 5, 7. ✓
- Same four constraint families, same return-arc circulation and objective as the agreed model — Step 5. ✓
- CP-native constructs on display: optional intervals, `add_exactly_one`, enforcement-literal outage, `add_cumulative` — Step 5. ✓
- Distinct from the `cp-sat-m` transliteration — rationale + docstring (Step 7). ✓
- Returns objective, gap, wall time, node count (`num_branches`) — Step 6. ✓
- Backend choice stays behind `backends.py`; routing uses the existing `family` seam — Step 3. ✓
- Agreement across cp-sat / cp-sat-m / scip / highs on toy + a solved instance — Step 1 tests. ✓
- Optional `≤K` honored natively via cumulative; K=0 ⇒ INFEASIBLE — Step 5 (4) + Step 1 test (verified against the live API). ✓
- Single-threaded fair timing — `num_workers = NUM_THREADS`. ✓

**Placeholder scan:** No TBD / vague "handle edge cases" — every code step is complete. The one `assert cap_k is not None` is a deliberate mypy narrowing aid with an explanatory comment (not a placeholder). ✓

**Type consistency:** `_solve_cp_sat` / `_to_result_cp_sat` signatures match their call sites; flow vars `cp_model.IntVar`, intervals `cp_model.IntervalVar`, status `cp_model.CpSolverStatus` — all verified to exist. `Result` field names match the dataclass. ✓
