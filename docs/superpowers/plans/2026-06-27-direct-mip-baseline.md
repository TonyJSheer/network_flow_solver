# Direct MIP Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/direct_mip.py` — the time-indexed circulation MIP — building through OR-Tools **MathOpt** (continuous flow) and returning a shared `Result`, validated against the toy instance's known optimum (8) and agreeing across MathOpt backends.

**Architecture:** One public function `solve_direct_mip(instance, backend, time_limit_s)` dispatches on `backend.family`. The MathOpt build path constructs the formulation from `docs/superpowers/specs/2026-06-27-direct-mip-formulation-design.md`: start-indexed binaries `x[j,s]`, continuous flows `f[a,t]` (real arcs + a return arc closing the circulation), derived in-progress expression `y[j,t]`, four constraint families, objective = total return-arc flow. The native CP-SAT integer build is **Stage 2.1** — stubbed here with a clear `NotImplementedError`.

**Tech Stack:** Python 3.12+, `ortools` (`ortools.math_opt.python.mathopt`), `pytest`. Run everything through `uv`.

## Global Constraints

- Python 3.12+; run Python only through `uv` (`uv run …`), never bare `python3`.
- Type annotations on every function; `uv run mypy .` must pass (`--strict` per project config); `uv run ruff check .` and `uv run ruff format --check .` clean.
- No bare `except:`; no `# type: ignore` without a justifying comment.
- Formulation file stays **backend-agnostic** in structure: it branches only on the `Backend` capability flags (`family`, `continuous_flow`, `solver_type`) from `src/backends.py`; no other solver-specific knowledge leaks in.
- Comment at the constraint level; module docstring states the math. Readability > cleverness.
- Reuse the existing `Result` / `SolveStatus` records and the `Backend` resolver as-is; do not modify them.
- Lint policy: `ruff format` + `ruff check --fix`; don't hand-fix formatting.

---

### Task 1: Core MathOpt build + solve (toy optimum)

**Files:**
- Create: `src/direct_mip.py`
- Test: `tests/test_direct_mip.py`

**Interfaces:**
- Consumes: `src.instance.Instance` (fields `name, horizon, source, sink, nodes, arcs, jobs, max_jobs_per_period`; `Arc(u, v, capacity)`, `Job(id, arc, duration, release, deadline)`); `src.backends.Backend` (`name, family, solver_type, continuous_flow`), `src.backends.ApiFamily`, `src.backends.resolve(name)`; `src.result.Result`, `src.result.SolveStatus`; `src.generator.toy_instance()`.
- Produces: `solve_direct_mip(instance: Instance, backend: Backend, time_limit_s: float | None = None) -> Result`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_direct_mip.py`:

```python
"""Direct MIP baseline tests. The toy instance has a hand-checked optimum of 8:
6 periods x bottleneck capacity 2 = 12, minus the unavoidable 2-period outage of
the single job on arc (a, t) (2 periods x 2) = 8.
"""

from __future__ import annotations

import pytest

from src.backends import resolve
from src.direct_mip import solve_direct_mip
from src.generator import toy_instance
from src.result import SolveStatus


def test_toy_optimum_on_highs() -> None:
    inst = toy_instance()
    res = solve_direct_mip(inst, resolve("highs"))

    assert res.method == "direct_mip"
    assert res.backend == "highs"
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
    # schedule recovered; the single job starts within its window [1, 5]
    assert res.schedule is not None
    assert res.schedule["j0"] in range(1, 6)
    assert res.node_count is not None
    assert res.wall_time_s >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_direct_mip.py::test_toy_optimum_on_highs -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.direct_mip'` (or import error).

- [ ] **Step 3: Write minimal implementation**

Create `src/direct_mip.py`:

```python
"""Direct MIP baseline — time-indexed circulation formulation.

Maximise total source->sink flow over a discrete horizon by scheduling each
arc-maintenance job exactly once within its start window; while a job is in
progress its arc's capacity is zero for those periods.

Formulation (docs/superpowers/specs/2026-06-27-direct-mip-formulation-design.md):

  variables
    x[j,s] in {0,1}   job j starts in period s,  s in [release_j, deadline_j]
    f[a,t] >= 0        flow on arc a in period t  (continuous on MathOpt backends)
  derived (linear in x, not a variable)
    y[j,t] = sum_{s = max(release_j, t-d_j+1) .. min(deadline_j, t)} x[j,s]
             (= 1 iff job j is in progress at period t)
  constraints
    (1) sum_s x[j,s] = 1                        each job scheduled exactly once
    (2) f[a(j),t] <= cap_{a(j)} * (1 - y[j,t])  capacity tied to outage
    (3) sum_in f - sum_out f = 0  (every node)  circulation via return arc t->s
    (4) sum_j y[j,t] <= K  (optional)           bounded jobs per period
  objective
    max sum_t f[r,t]   flow on return arc r = (sink, source) = total throughput

Backend note: only continuous-flow MathOpt backends (scip / highs / cp-sat-m /
gurobi) are wired here. The native CP-SAT integer build is Stage 2.1.
"""

from __future__ import annotations

import datetime

from ortools.math_opt.python import mathopt

from src.backends import ApiFamily, Backend
from src.instance import Instance
from src.result import Result, SolveStatus


def solve_direct_mip(
    instance: Instance,
    backend: Backend,
    time_limit_s: float | None = None,
) -> Result:
    """Solve the time-indexed direct MIP on a MathOpt backend."""
    if backend.family is ApiFamily.CP_SAT:
        raise NotImplementedError(
            "CP-SAT integer build is Stage 2.1; use a MathOpt backend "
            "(scip / highs / cp-sat-m / gurobi)."
        )
    return _solve_mathopt(instance, backend, time_limit_s)


def _solve_mathopt(
    instance: Instance, backend: Backend, time_limit_s: float | None
) -> Result:
    assert backend.solver_type is not None  # every MathOpt backend has a SolverType
    periods = range(1, instance.horizon + 1)
    model = mathopt.Model(name=f"direct_mip:{instance.name}")

    # Flow vars for every real arc plus the return arc (sink -> source). The
    # return arc is uncapacitated in effect: its cap is an upper bound on any
    # feasible throughput (total capacity leaving the source). Variable upper
    # bounds enforce the plain capacity limit f[a,t] <= cap_a for all arcs.
    return_cap = sum(a.capacity for a in instance.arcs if a.u == instance.source)
    arc_caps: dict[tuple[str, str], int] = {(a.u, a.v): a.capacity for a in instance.arcs}
    arc_caps[(instance.sink, instance.source)] = return_cap
    f: dict[tuple[str, str, int], mathopt.Variable] = {}
    for (u, v), cap in arc_caps.items():
        for t in periods:
            f[(u, v, t)] = model.add_variable(lb=0.0, ub=float(cap), name=f"f[{u},{v},{t}]")

    # Start-indexed schedule vars over each job's allowed start window.
    x: dict[tuple[str, int], mathopt.Variable] = {}
    starts: dict[str, range] = {}
    for j in instance.jobs:
        starts[j.id] = range(j.release, j.deadline + 1)
        for s in starts[j.id]:
            x[(j.id, s)] = model.add_binary_variable(name=f"x[{j.id},{s}]")

    # (1) each job scheduled exactly once.
    for j in instance.jobs:
        model.add_linear_constraint(sum(x[(j.id, s)] for s in starts[j.id]) == 1)

    def in_progress(job_id: str, duration: int, t: int) -> mathopt.LinearExpression:
        # y[j,t] = sum of starts s with t-d+1 <= s <= t (clamped to the window).
        lo = max(starts[job_id].start, t - duration + 1)
        hi = min(starts[job_id].stop - 1, t)
        return sum((x[(job_id, s)] for s in range(lo, hi + 1)), start=mathopt.LinearExpression())

    # (2) capacity tied to outage, one constraint per (job, period). Big-M free:
    # any in-progress job on an arc forces that arc's flow to 0 that period.
    for j in instance.jobs:
        cap = arc_caps[j.arc]
        for t in periods:
            y = in_progress(j.id, j.duration, t)
            model.add_linear_constraint(f[(j.arc[0], j.arc[1], t)] <= cap * (1 - y))

    # (3) circulation: flow conservation at every node (source/sink included,
    # balanced by the return arc), every period.
    for n in instance.nodes:
        for t in periods:
            inflow = sum(
                (f[(u, v, t)] for (u, v) in arc_caps if v == n),
                start=mathopt.LinearExpression(),
            )
            outflow = sum(
                (f[(u, v, t)] for (u, v) in arc_caps if u == n),
                start=mathopt.LinearExpression(),
            )
            model.add_linear_constraint(inflow - outflow == 0)

    # (4) optional bounded jobs per period.
    if instance.max_jobs_per_period is not None:
        cap_k = instance.max_jobs_per_period
        for t in periods:
            total_in_progress = sum(
                (in_progress(j.id, j.duration, t) for j in instance.jobs),
                start=mathopt.LinearExpression(),
            )
            model.add_linear_constraint(total_in_progress <= cap_k)

    # Objective: maximise total return-arc flow = total source->sink throughput.
    ret = (instance.sink, instance.source)
    model.maximize(sum((f[(ret[0], ret[1], t)] for t in periods), start=mathopt.LinearExpression()))

    params = mathopt.SolveParameters()
    if time_limit_s is not None:
        params = mathopt.SolveParameters(
            time_limit=datetime.timedelta(seconds=time_limit_s)
        )
    result = mathopt.solve(model, backend.solver_type, params=params)
    return _to_result(instance, backend, result, x, starts)


def _to_result(
    instance: Instance,
    backend: Backend,
    result: mathopt.SolveResult,
    x: dict[tuple[str, int], mathopt.Variable],
    starts: dict[str, range],
) -> Result:
    reason = result.termination.reason
    if reason is mathopt.TerminationReason.OPTIMAL:
        status = SolveStatus.OPTIMAL
    elif result.has_primal_feasible_solution():
        status = SolveStatus.FEASIBLE
    elif reason is mathopt.TerminationReason.INFEASIBLE:
        status = SolveStatus.INFEASIBLE
    else:
        status = SolveStatus.UNKNOWN

    wall = result.solve_stats.solve_time.total_seconds()
    node_count = result.solve_stats.node_count

    if not result.has_primal_feasible_solution():
        return Result(
            method="direct_mip",
            backend=backend.name,
            status=status,
            objective=None,
            wall_time_s=wall,
            node_count=node_count,
        )

    schedule: dict[str, int] = {}
    for job_id, window in starts.items():
        for s in window:
            if result.variable_values(x[(job_id, s)]) > 0.5:
                schedule[job_id] = s
                break

    primal = result.objective_value()
    dual = result.best_objective_bound()
    gap = abs(primal - dual) / (abs(primal) + 1e-10)
    return Result(
        method="direct_mip",
        backend=backend.name,
        status=status,
        objective=primal,
        wall_time_s=wall,
        gap=gap,
        schedule=schedule,
        node_count=node_count,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_direct_mip.py::test_toy_optimum_on_highs -v`
Expected: PASS.

- [ ] **Step 5: Lint and type-check**

Run: `uv run ruff check --fix . && uv run ruff format . && uv run mypy .`
Expected: all clean (no errors).

- [ ] **Step 6: Commit**

```bash
git add src/direct_mip.py tests/test_direct_mip.py
git commit -m "feat: direct MIP baseline (MathOpt path), toy optimum

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Bounded-jobs-per-period (K) constraint

**Files:**
- Modify: `tests/test_direct_mip.py` (add two tests)
- (No source change expected — the K constraint is already written in Task 1; these tests prove it works. If a test fails, fix `src/direct_mip.py:` the `max_jobs_per_period` block.)

**Interfaces:**
- Consumes: `solve_direct_mip` from Task 1; `dataclasses.replace` to set `max_jobs_per_period` on the frozen `Instance`.
- Produces: nothing new (validation only).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_direct_mip.py` (top-level imports — add `from dataclasses import replace` at the top with the other imports):

```python
def test_k_zero_makes_instance_infeasible() -> None:
    # Every job must be scheduled (start exactly once) => in progress at some
    # period; K=0 forbids any in-progress period, so the model is infeasible.
    inst = replace(toy_instance(), max_jobs_per_period=0)
    res = solve_direct_mip(inst, resolve("highs"))

    assert res.status is SolveStatus.INFEASIBLE
    assert res.objective is None


def test_k_one_keeps_toy_optimum() -> None:
    # The toy instance has one job, so K=1 cannot bind: optimum stays 8.
    inst = replace(toy_instance(), max_jobs_per_period=1)
    res = solve_direct_mip(inst, resolve("highs"))

    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_direct_mip.py -k "k_zero or k_one" -v`
Expected: both PASS (the K constraint from Task 1 already handles these). If `test_k_zero_makes_instance_infeasible` fails, verify constraint (4) in `src/direct_mip.py` uses `<= cap_k` and that `max_jobs_per_period is not None` gates it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_direct_mip.py
git commit -m "test: bounded jobs-per-period (K) constraint

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Cross-backend agreement + native CP-SAT deferral

**Files:**
- Modify: `tests/test_direct_mip.py` (add three tests)

**Interfaces:**
- Consumes: `solve_direct_mip`, `resolve` (Task 1); `src.generator.generate_instance`, `src.generator.Regime`.
- Produces: nothing new (validation only).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_direct_mip.py` (add `from src.generator import Regime, generate_instance` to the imports):

```python
@pytest.mark.parametrize("backend_name", ["highs", "scip", "cp-sat-m"])
def test_backends_agree_on_toy(backend_name: str) -> None:
    res = solve_direct_mip(toy_instance(), resolve(backend_name))
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)


def test_backends_agree_on_generated_instance() -> None:
    # Smallest size, tight window => fast; all MathOpt backends must agree on the
    # optimum (we compare objective + status, not the schedule: see the
    # multi-optimum cross-check note).
    inst = generate_instance(size_idx=1, list_idx=0, regime=Regime.TIGHT, seed=7)
    objectives: list[float] = []
    for name in ("highs", "scip", "cp-sat-m"):
        res = solve_direct_mip(inst, resolve(name), time_limit_s=60.0)
        assert res.status is SolveStatus.OPTIMAL
        assert res.objective is not None
        objectives.append(res.objective)
    assert objectives[0] == pytest.approx(objectives[1])
    assert objectives[1] == pytest.approx(objectives[2])


def test_native_cp_sat_is_deferred_to_stage_2_1() -> None:
    with pytest.raises(NotImplementedError, match="Stage 2.1"):
        solve_direct_mip(toy_instance(), resolve("cp-sat"))
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_direct_mip.py -k "agree or deferred" -v`
Expected: all PASS. (`generate_instance` uses `size_idx` in 1..8; `size_idx=1` is the smallest.)

- [ ] **Step 3: Run the full suite + lint + types**

Run: `uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy .`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_direct_mip.py
git commit -m "test: direct MIP agrees across MathOpt backends; CP-SAT deferred to 2.1

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **Why MathOpt-only here:** the user scoped the native CP-SAT integer build to Stage 2.1. `cp-sat-m` (MathOpt's `SolverType.CP_SAT`) still works in this stage because it goes through the continuous-flow MathOpt path — so CP-SAT's solver is exercised even before the native build lands.
- **`sum(..., start=mathopt.LinearExpression())`:** seeds the reduction with an empty linear expression so empty ranges (e.g. a period where a job cannot be in progress) yield a valid `0` expression rather than a Python `int`, keeping types uniform for `mypy`.
- **Multi-optimum cross-check:** agreement tests compare `(objective, status)`, never the exact schedule — many schedules can be optimal (any feasible start of the toy job loses the same two periods).
- **`solve_time` source:** `result.solve_stats.solve_time` is a `datetime.timedelta`; `.total_seconds()` gives the float for `wall_time_s`.
```
