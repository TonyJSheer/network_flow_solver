# Benders Stage 3a — Analytic min-cut Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build working disaggregated Benders decomposition for the maintenance-scheduling problem using the **analytic min-cut** subproblem evaluator (networkx), with both cut-injection paths (iterative loop + lazy callback), agreeing optimum-for-optimum with the direct MIP across backends.

**Architecture:** A pluggable **evaluator seam** (`src/subproblem.py`) solves each per-period s→t max-flow with `networkx` and returns the min-cut as disaggregated cut coefficients keyed by arc. `src/benders.py` builds the Benders master (binary schedule + per-period flow proxies `θ_t` + arc-availability proxies `y[a,t]`) on two API paths (`cp_model` for cp-sat, `mathopt` for scip/highs/gurobi), and injects cuts via an iterative re-solve loop (all backends) or a lazy callback (scip/gurobi). This is sub-stage 3a; the LP-dual (3b) and Pareto (3c) evaluators plug into the same seam in later plans.

**Tech Stack:** Python 3.12+, OR-Tools (`ortools.sat.python.cp_model`, `ortools.math_opt.python.mathopt`/`callback`), `networkx` (max-flow/min-cut), `pytest`. Run everything through `uv`.

## Global Constraints

- Run Python only through `uv` — never bare `python3`.
- Type annotations on every function; `uv run mypy .` clean (`--strict`); `uv run ruff check . && uv run ruff format --check .` clean. No bare `except`; no unjustified `# type: ignore`.
- Formulation files stay **backend-agnostic** — backend choice lives only in `src/backends.py`. The master's scheduling constraints are **duplicated** from `direct_mip.py` (CLAUDE.md "formulation files self-contained"); agreement tests guard against drift.
- Benders must return the **same optimum** as `solve_direct_mip` on every instance both solve to optimality, and agree across backends. Agreement tests compare `(objective, status)`, **never** the schedule (multi-optimum).
- CP-SAT flow/availability vars are integer/Boolean (exact here — integral capacities give an integral max-flow); MathOpt uses continuous. Keep both behind the same logic.
- Reuse the existing `Result` dataclass; do not add per-strategy fields to it.
- Commit after each task with `<type>: <message>` and the trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: Subproblem evaluator (analytic min-cut)

**Files:**
- Create: `src/subproblem.py`
- Test: `tests/test_subproblem.py`

**Interfaces:**
- Consumes: `src.instance.Instance` (fields `arcs`, `source`, `sink`, with `Arc(u, v, capacity)`).
- Produces:
  - `PeriodCut` dataclass: `flow_value: int`, `coeffs: dict[tuple[str, str], int]` (arc → **original** capacity, for arcs crossing the min-cut).
  - `class MinCutEvaluator` with `__init__(self, instance: Instance)` and `evaluate(self, closed_arcs: frozenset[tuple[str, str]]) -> PeriodCut`. `closed_arcs` are arcs at capacity 0 this period. Caches by `closed_arcs`; exposes `total_calls: int` and `distinct_solves: int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_subproblem.py
from __future__ import annotations

from src.generator import toy_instance
from src.subproblem import MinCutEvaluator, PeriodCut


def test_open_network_mincut_is_bottleneck_arc() -> None:
    # toy: s->a cap 3, a->t cap 2. Min cut = {(a,t)} value 2.
    ev = MinCutEvaluator(toy_instance())
    cut = ev.evaluate(frozenset())
    assert cut.flow_value == 2
    assert cut.coeffs == {("a", "t"): 2}


def test_closed_bottleneck_gives_zero_flow_but_keeps_original_cap() -> None:
    ev = MinCutEvaluator(toy_instance())
    cut = ev.evaluate(frozenset({("a", "t")}))
    assert cut.flow_value == 0
    # the cut arc still reports its ORIGINAL capacity (master's y handles closure)
    assert cut.coeffs == {("a", "t"): 2}


def test_cache_counts_distinct_configs() -> None:
    ev = MinCutEvaluator(toy_instance())
    ev.evaluate(frozenset())
    ev.evaluate(frozenset())
    ev.evaluate(frozenset({("a", "t")}))
    assert ev.total_calls == 3
    assert ev.distinct_solves == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_subproblem.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.subproblem'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/subproblem.py
"""Per-period subproblem evaluators for Benders decomposition.

Given a fixed schedule, each period is an independent s->t max-flow on the
network with arcs-under-maintenance set to capacity 0. The LP dual of max-flow
is min-cut, so the analytic evaluator solves the flow combinatorially with
networkx and reads the min-cut straight off the result -- no LP in the loop.

The returned cut is the disaggregated Benders optimality cut for that period:

    theta_t <= sum_{a in min-cut} cap_a * y[a,t]

where cap_a is the arc's ORIGINAL capacity and y[a,t] is the master's
arc-availability proxy. Coefficients are keyed by arc; the evaluator never sees
master variables.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from src.instance import Instance


@dataclass(frozen=True)
class PeriodCut:
    """One disaggregated optimality cut. ``coeffs`` maps each min-cut arc to its
    original capacity; ``flow_value`` is the period's max flow under the outage."""

    flow_value: int
    coeffs: dict[tuple[str, str], int]


class MinCutEvaluator:
    """Analytic evaluator: networkx max-flow, read the source-side min-cut.

    Caches results by the frozenset of closed arcs, so identical outage patterns
    across periods/iterations are recalled rather than re-solved.
    """

    def __init__(self, instance: Instance) -> None:
        self._instance = instance
        self._caps: dict[tuple[str, str], int] = {(a.u, a.v): a.capacity for a in instance.arcs}
        self._cache: dict[frozenset[tuple[str, str]], PeriodCut] = {}
        self.total_calls = 0
        self.distinct_solves = 0

    def evaluate(self, closed_arcs: frozenset[tuple[str, str]]) -> PeriodCut:
        self.total_calls += 1
        cached = self._cache.get(closed_arcs)
        if cached is not None:
            return cached
        self.distinct_solves += 1
        cut = self._solve(closed_arcs)
        self._cache[closed_arcs] = cut
        return cut

    def _solve(self, closed_arcs: frozenset[tuple[str, str]]) -> PeriodCut:
        g: nx.DiGraph = nx.DiGraph()
        for (u, v), cap in self._caps.items():
            g.add_edge(u, v, capacity=0 if (u, v) in closed_arcs else cap)
        value, (reachable, _) = nx.minimum_cut(g, self._instance.source, self._instance.sink)
        coeffs = {
            (u, v): cap
            for (u, v), cap in self._caps.items()
            if u in reachable and v not in reachable
        }
        return PeriodCut(flow_value=int(value), coeffs=coeffs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_subproblem.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format src/subproblem.py tests/test_subproblem.py
uv run ruff check --fix src/subproblem.py tests/test_subproblem.py
uv run mypy src/subproblem.py
git add src/subproblem.py tests/test_subproblem.py
git commit -m "feat: analytic min-cut subproblem evaluator (Benders 3a)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: MathOpt Benders master (no cuts → relaxation bound)

**Files:**
- Create: `src/benders.py`
- Test: `tests/test_benders.py`

**Interfaces:**
- Consumes: `src.instance.Instance`, `src.backends.Backend`/`NUM_THREADS`, `src.subproblem.MinCutEvaluator`.
- Produces:
  - `_full_capacity_maxflow(instance: Instance) -> int` — UB for every `θ_t`.
  - `@dataclass _Master`: `model: mathopt.Model`, `x: dict[tuple[str, int], mathopt.Variable]`, `y: dict[tuple[tuple[str, str], int], mathopt.Variable]`, `theta: dict[int, mathopt.Variable]`, `starts: dict[str, range]`.
  - `_jobs_on_arc(instance) -> dict[tuple[str, str], list[Job]]`.
  - `_build_master_mathopt(instance: Instance, ub: int) -> _Master`.
  - `solve_benders(instance, backend, time_limit_s=None, *, pre_cuts=False) -> Result` (stub in this task: build master, solve with no cuts, return the relaxation bound as objective; later tasks add the cut loop/callback).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benders.py
from __future__ import annotations

import pytest

from src.backends import resolve
from src.benders import _full_capacity_maxflow, solve_benders
from src.generator import toy_instance
from src.result import SolveStatus


def test_full_capacity_maxflow_toy() -> None:
    assert _full_capacity_maxflow(toy_instance()) == 2


def test_master_without_cuts_is_relaxation_bound() -> None:
    # No cuts yet: each theta_t hits its UB (=2), summed over horizon 6 => 12.
    # (The true optimum is 8; cuts in later tasks bring it down.)
    res = solve_benders(toy_instance(), resolve("highs"))
    assert res.method == "benders"
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(12.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benders.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.benders'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/benders.py
"""Disaggregated Benders decomposition (Stage 3).

Master: binary schedule x[j,s], arc-availability proxy y[a,t] in [0,1]
(1 iff arc a is open at t), per-period flow proxy theta_t. Objective max sum_t theta_t.
Subproblem: per-period s->t max-flow (src.subproblem), giving disaggregated cuts
    theta_t <= sum_{a in min-cut} cap_a * y[a,t].
This module is backend-agnostic; backend choice lives in src.backends.

Arc availability under OVERLAPPING same-arc jobs: y[a,t] <= 1 - inprogress_j(t) for
EACH job j on arc a (each inprogress_j is 0/1 by exactly-one). This matches the
direct MIP's per-job capacity semantics, so the two formulations share an optimum.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from ortools.math_opt.python import mathopt

from src.backends import Backend
from src.instance import Instance, Job
from src.result import Result, SolveStatus
from src.subproblem import MinCutEvaluator


def _full_capacity_maxflow(instance: Instance) -> int:
    """Upper bound on any period's flow: max-flow with every arc open."""
    g: nx.DiGraph = nx.DiGraph()
    for a in instance.arcs:
        g.add_edge(a.u, a.v, capacity=a.capacity)
    value, _ = nx.minimum_cut(g, instance.source, instance.sink)
    return int(value)


def _jobs_on_arc(instance: Instance) -> dict[tuple[str, str], list[Job]]:
    out: dict[tuple[str, str], list[Job]] = {}
    for j in instance.jobs:
        out.setdefault(j.arc, []).append(j)
    return out


@dataclass
class _Master:
    model: mathopt.Model
    x: dict[tuple[str, int], mathopt.Variable]
    y: dict[tuple[tuple[str, str], int], mathopt.Variable]
    theta: dict[int, mathopt.Variable]
    starts: dict[str, range]


def _build_master_mathopt(instance: Instance, ub: int) -> _Master:
    periods = range(1, instance.horizon + 1)
    model = mathopt.Model(name=f"benders:{instance.name}")

    # Start-indexed schedule vars + exactly-once.
    x: dict[tuple[str, int], mathopt.Variable] = {}
    starts: dict[str, range] = {}
    for j in instance.jobs:
        starts[j.id] = range(j.release, j.deadline + 1)
        for s in starts[j.id]:
            x[(j.id, s)] = model.add_binary_variable(name=f"x[{j.id},{s}]")
        model.add_linear_constraint(sum(x[(j.id, s)] for s in starts[j.id]) == 1)

    def in_progress(job: Job, t: int) -> mathopt.LinearExpression:
        lo = max(starts[job.id].start, t - job.duration + 1)
        hi = min(starts[job.id].stop - 1, t)
        return sum((x[(job.id, s)] for s in range(lo, hi + 1)), start=mathopt.LinearExpression())

    # Arc-availability proxy y[a,t] in [0,1] for arcs that carry jobs; y <= 1 - ip_j
    # per job j on the arc. Maximize pulls y up to exactly "open iff no job in progress".
    y: dict[tuple[tuple[str, str], int], mathopt.Variable] = {}
    for arc, jobs in _jobs_on_arc(instance).items():
        for t in periods:
            yvar = model.add_variable(lb=0.0, ub=1.0, name=f"y[{arc[0]},{arc[1]},{t}]")
            y[(arc, t)] = yvar
            for j in jobs:
                model.add_linear_constraint(yvar <= 1 - in_progress(j, t))

    # Per-period flow proxy, bounded by the full-capacity max-flow.
    theta: dict[int, mathopt.Variable] = {
        t: model.add_variable(lb=0.0, ub=float(ub), name=f"theta[{t}]") for t in periods
    }
    model.maximize(sum((theta[t] for t in periods), start=mathopt.LinearExpression()))
    return _Master(model=model, x=x, y=y, theta=theta, starts=starts)


def solve_benders(
    instance: Instance,
    backend: Backend,
    time_limit_s: float | None = None,
    *,
    pre_cuts: bool = False,
) -> Result:
    """Solve via disaggregated Benders. (This task: master only, no cuts yet.)"""
    ub = _full_capacity_maxflow(instance)
    if backend.solver_type is None:
        raise ValueError(f"backend {backend.name!r} has no MathOpt SolverType")
    master = _build_master_mathopt(instance, ub)
    result = mathopt.solve(master.model, backend.solver_type)
    status = (
        SolveStatus.OPTIMAL
        if result.termination.reason is mathopt.TerminationReason.OPTIMAL
        else SolveStatus.UNKNOWN
    )
    return Result(
        method="benders",
        backend=backend.name,
        status=status,
        objective=result.objective_value() if result.has_primal_feasible_solution() else None,
        wall_time_s=result.solve_stats.solve_time.total_seconds(),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_benders.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format src/benders.py tests/test_benders.py
uv run ruff check --fix src/benders.py tests/test_benders.py
uv run mypy src/benders.py
git add src/benders.py tests/test_benders.py
git commit -m "feat: Benders MathOpt master (relaxation bound, no cuts) (3a)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Iterative cut loop (MathOpt) → toy optimum, agrees with direct MIP

**Files:**
- Modify: `src/benders.py`
- Test: `tests/test_benders.py`

**Interfaces:**
- Consumes: Task 2's `_Master`, `_build_master_mathopt`, `_jobs_on_arc`, `_full_capacity_maxflow`; `MinCutEvaluator`.
- Produces:
  - `_closed_arcs(instance, schedule, t) -> frozenset[tuple[str, str]]`.
  - `_schedule_from_mathopt(result, starts) -> dict[str, int]`.
  - `_cut_rhs_mathopt(master, instance, arcs_with_jobs, t, cut) -> mathopt.LinearExpression` — builds `Σ cap_a·y[a,t]` (constant `cap_a` for cut arcs with no job).
  - `_solve_loop_mathopt(instance, backend, time_limit_s, *, pre_cuts) -> Result` — the iterative re-solve loop; fills `objective`, `status`, `gap`, `schedule`, `iteration_count`, `cut_count`.
  - `solve_benders` dispatches non-lazy MathOpt backends to `_solve_loop_mathopt`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_benders.py
from src.direct_mip import solve_direct_mip


def test_loop_reaches_toy_optimum_highs() -> None:
    res = solve_benders(toy_instance(), resolve("highs"))
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
    assert res.iteration_count is not None and res.iteration_count >= 1
    assert res.cut_count is not None and res.cut_count >= 1
    assert res.schedule is not None and res.schedule["j0"] in range(1, 6)


def test_benders_agrees_with_direct_mip_toy_highs() -> None:
    inst = toy_instance()
    b = solve_benders(inst, resolve("highs"))
    d = solve_direct_mip(inst, resolve("highs"))
    assert b.status is d.status is SolveStatus.OPTIMAL
    assert b.objective == pytest.approx(d.objective)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benders.py::test_loop_reaches_toy_optimum_highs -v`
Expected: FAIL — `objective` is 12.0 (relaxation), not 8.0.

- [ ] **Step 3: Write minimal implementation**

Add to `src/benders.py` (and route `solve_benders` to the loop). Use `time.perf_counter` for wall time:

```python
import time

_EPS = 1e-6


def _closed_arcs(
    instance: Instance, schedule: dict[str, int], t: int
) -> frozenset[tuple[str, str]]:
    closed: set[tuple[str, str]] = set()
    for j in instance.jobs:
        s = schedule[j.id]
        if s <= t <= s + j.duration - 1:
            closed.add(j.arc)
    return frozenset(closed)


def _schedule_from_mathopt(
    result: mathopt.SolveResult, starts: dict[str, range], x: dict[tuple[str, int], mathopt.Variable]
) -> dict[str, int]:
    schedule: dict[str, int] = {}
    for job_id, window in starts.items():
        for s in window:
            if result.variable_values(x[(job_id, s)]) > 0.5:
                schedule[job_id] = s
                break
    return schedule


def _cut_rhs_mathopt(
    master: _Master,
    arcs_with_jobs: frozenset[tuple[str, str]],
    t: int,
    cut: "PeriodCut",
) -> mathopt.LinearExpression:
    rhs = mathopt.LinearExpression()
    for arc, cap in cut.coeffs.items():
        if arc in arcs_with_jobs:
            rhs += cap * master.y[(arc, t)]
        else:
            rhs += cap  # arc never under maintenance => always open
    return rhs


def _solve_loop_mathopt(
    instance: Instance, backend: Backend, time_limit_s: float | None, *, pre_cuts: bool
) -> Result:
    assert backend.solver_type is not None
    start = time.perf_counter()
    ub = _full_capacity_maxflow(instance)
    master = _build_master_mathopt(instance, ub)
    evaluator = MinCutEvaluator(instance)
    arcs_with_jobs = frozenset(_jobs_on_arc(instance))
    periods = range(1, instance.horizon + 1)

    iterations = 0
    cut_count = 0
    schedule: dict[str, int] = {}
    true_objective = 0.0
    while True:
        iterations += 1
        result = mathopt.solve(master.model, backend.solver_type)
        if result.termination.reason is not mathopt.TerminationReason.OPTIMAL:
            return Result(
                method="benders", backend=backend.name, status=SolveStatus.UNKNOWN,
                objective=None, wall_time_s=time.perf_counter() - start,
                iteration_count=iterations, cut_count=cut_count,
            )
        schedule = _schedule_from_mathopt(result, master.starts, master.x)
        added = 0
        true_objective = 0.0
        for t in periods:
            cut = evaluator.evaluate(_closed_arcs(instance, schedule, t))
            true_objective += cut.flow_value
            if result.variable_values(master.theta[t]) > cut.flow_value + _EPS:
                master.model.add_linear_constraint(
                    master.theta[t] <= _cut_rhs_mathopt(master, arcs_with_jobs, t, cut)
                )
                added += 1
                cut_count += 1
        if added == 0:
            break  # no violated cut => master objective is exact

    return Result(
        method="benders", backend=backend.name, status=SolveStatus.OPTIMAL,
        objective=true_objective, wall_time_s=time.perf_counter() - start, gap=0.0,
        schedule=schedule, iteration_count=iterations, cut_count=cut_count,
    )
```

Then replace the body of `solve_benders` so non-lazy MathOpt backends use the loop:

```python
def solve_benders(
    instance: Instance, backend: Backend, time_limit_s: float | None = None, *, pre_cuts: bool = False
) -> Result:
    """Solve via disaggregated Benders (analytic min-cut, iterative loop)."""
    if backend.solver_type is None:
        raise ValueError(f"backend {backend.name!r} has no MathOpt SolverType")
    return _solve_loop_mathopt(instance, backend, time_limit_s, pre_cuts=pre_cuts)
```

Add `from src.subproblem import MinCutEvaluator, PeriodCut` (import `PeriodCut` for the type hint).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benders.py -v`
Expected: PASS (all, including the two new ones). The old `test_master_without_cuts_is_relaxation_bound` now exercises a private helper — update it to call `_build_master_mathopt` + `mathopt.solve` directly instead of `solve_benders`, since `solve_benders` now runs the full loop:

```python
def test_master_without_cuts_is_relaxation_bound() -> None:
    from ortools.math_opt.python import mathopt
    from src.benders import _build_master_mathopt
    master = _build_master_mathopt(toy_instance(), ub=2)
    result = mathopt.solve(master.model, resolve("highs").solver_type)
    assert result.objective_value() == pytest.approx(12.0)
```

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format src/benders.py tests/test_benders.py
uv run ruff check --fix src/benders.py tests/test_benders.py
uv run mypy src/benders.py
git add src/benders.py tests/test_benders.py
git commit -m "feat: Benders iterative cut loop on MathOpt, toy optimum (3a)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: CP-SAT master + loop (primary backend)

**Files:**
- Modify: `src/benders.py`
- Test: `tests/test_benders.py`

**Interfaces:**
- Consumes: `_full_capacity_maxflow`, `_jobs_on_arc`, `_closed_arcs`, `MinCutEvaluator`, `src.backends.NUM_THREADS`, `ortools.sat.python.cp_model`.
- Produces:
  - `@dataclass _MasterCp`: `model: cp_model.CpModel`, `x`, `y`, `theta` (all `cp_model.IntVar`), `starts`.
  - `_build_master_cpsat(instance, ub) -> _MasterCp`.
  - `_solve_loop_cpsat(instance, backend, time_limit_s, *, pre_cuts) -> Result`.
  - `solve_benders` routes `ApiFamily.CP_SAT` backends to `_solve_loop_cpsat`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_benders.py
def test_loop_reaches_toy_optimum_cpsat() -> None:
    res = solve_benders(toy_instance(), resolve("cp-sat"))
    assert res.backend == "cp-sat"
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
    assert res.schedule is not None and res.schedule["j0"] in range(1, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benders.py::test_loop_reaches_toy_optimum_cpsat -v`
Expected: FAIL — `solve_benders` raises `ValueError` (`cp-sat` has no MathOpt SolverType).

- [ ] **Step 3: Write minimal implementation**

Add the CP-SAT path. `y[a,t]` and `theta_t` are integer (exact here); CP-SAT internal LP sees the linear cut. Mirror `_build_master_mathopt` exactly:

```python
from ortools.sat.python import cp_model

from src.backends import NUM_THREADS, ApiFamily


@dataclass
class _MasterCp:
    model: cp_model.CpModel
    x: dict[tuple[str, int], cp_model.IntVar]
    y: dict[tuple[tuple[str, str], int], cp_model.IntVar]
    theta: dict[int, cp_model.IntVar]
    starts: dict[str, range]


def _build_master_cpsat(instance: Instance, ub: int) -> _MasterCp:
    periods = range(1, instance.horizon + 1)
    model = cp_model.CpModel()
    x: dict[tuple[str, int], cp_model.IntVar] = {}
    starts: dict[str, range] = {}
    for j in instance.jobs:
        starts[j.id] = range(j.release, j.deadline + 1)
        lits = [model.new_bool_var(f"x[{j.id},{s}]") for s in starts[j.id]]
        for s, lit in zip(starts[j.id], lits):
            x[(j.id, s)] = lit
        model.add_exactly_one(lits)

    def in_progress(job: Job, t: int) -> cp_model.LinearExpr:
        lo = max(starts[job.id].start, t - job.duration + 1)
        hi = min(starts[job.id].stop - 1, t)
        return cp_model.LinearExpr.sum([x[(job.id, s)] for s in range(lo, hi + 1)])

    y: dict[tuple[tuple[str, str], int], cp_model.IntVar] = {}
    for arc, jobs in _jobs_on_arc(instance).items():
        for t in periods:
            yvar = model.new_int_var(0, 1, f"y[{arc[0]},{arc[1]},{t}]")
            y[(arc, t)] = yvar
            for j in jobs:
                model.add(yvar <= 1 - in_progress(j, t))

    theta = {t: model.new_int_var(0, ub, f"theta[{t}]") for t in periods}
    model.maximize(cp_model.LinearExpr.sum([theta[t] for t in periods]))
    return _MasterCp(model=model, x=x, y=y, theta=theta, starts=starts)


def _solve_loop_cpsat(
    instance: Instance, backend: Backend, time_limit_s: float | None, *, pre_cuts: bool
) -> Result:
    start = time.perf_counter()
    ub = _full_capacity_maxflow(instance)
    master = _build_master_cpsat(instance, ub)
    evaluator = MinCutEvaluator(instance)
    arcs_with_jobs = frozenset(_jobs_on_arc(instance))
    periods = range(1, instance.horizon + 1)
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = NUM_THREADS

    iterations = 0
    cut_count = 0
    schedule: dict[str, int] = {}
    true_objective = 0
    while True:
        iterations += 1
        status = solver.solve(master.model)
        if status != cp_model.OPTIMAL:
            return Result(
                method="benders", backend=backend.name, status=SolveStatus.UNKNOWN,
                objective=None, wall_time_s=time.perf_counter() - start,
                iteration_count=iterations, cut_count=cut_count,
            )
        schedule = {
            job_id: next(s for s in window if solver.value(master.x[(job_id, s)]) > 0.5)
            for job_id, window in master.starts.items()
        }
        added = 0
        true_objective = 0
        for t in periods:
            cut = evaluator.evaluate(_closed_arcs(instance, schedule, t))
            true_objective += cut.flow_value
            if solver.value(master.theta[t]) > cut.flow_value:
                rhs: list[cp_model.LinearExpr] = []
                const = 0
                for arc, cap in cut.coeffs.items():
                    if arc in arcs_with_jobs:
                        rhs.append(cap * master.y[(arc, t)])
                    else:
                        const += cap
                master.model.add(master.theta[t] <= cp_model.LinearExpr.sum(rhs) + const)
                added += 1
                cut_count += 1
        if added == 0:
            break

    return Result(
        method="benders", backend=backend.name, status=SolveStatus.OPTIMAL,
        objective=float(true_objective), wall_time_s=time.perf_counter() - start, gap=0.0,
        schedule=schedule, iteration_count=iterations, cut_count=cut_count,
    )
```

Update `solve_benders` to dispatch:

```python
def solve_benders(
    instance: Instance, backend: Backend, time_limit_s: float | None = None, *, pre_cuts: bool = False
) -> Result:
    if backend.family is ApiFamily.CP_SAT:
        return _solve_loop_cpsat(instance, backend, time_limit_s, pre_cuts=pre_cuts)
    return _solve_loop_mathopt(instance, backend, time_limit_s, pre_cuts=pre_cuts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benders.py -v`
Expected: PASS (all, including cp-sat).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format src/benders.py tests/test_benders.py
uv run ruff check --fix src/benders.py tests/test_benders.py
uv run mypy src/benders.py
git add src/benders.py tests/test_benders.py
git commit -m "feat: Benders CP-SAT master + iterative loop (3a)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Lazy-callback injection (SCIP) + enable `supports_lazy`

**Files:**
- Modify: `src/benders.py`, `src/backends.py`
- Test: `tests/test_benders.py`

**Interfaces:**
- Consumes: `_build_master_mathopt`, `_closed_arcs`, `_cut_rhs_mathopt`, `_full_capacity_maxflow`, `_jobs_on_arc`, `MinCutEvaluator`; `ortools.math_opt.python.callback` (`Event`, `CallbackRegistration`, `CallbackData`, `CallbackResult`).
- Produces:
  - `_solve_lazy_mathopt(instance, backend, time_limit_s, *, pre_cuts) -> Result` — registers a `MIP_SOLUTION` lazy-constraint callback; at each incumbent, evaluates all periods and `add_lazy_constraint` for each violated `θ_t`.
  - `solve_benders` routes MathOpt backends with `backend.supports_lazy` to the lazy path, else the loop.
  - `src/backends.py`: `supports_lazy=True` for `scip` and `gurobi`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_benders.py
def test_lazy_reaches_toy_optimum_scip() -> None:
    res = solve_benders(toy_instance(), resolve("scip"))
    assert res.backend == "scip"
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
    assert res.cut_count is not None and res.cut_count >= 1
```

And in `tests/test_backends.py` assert the capability flags flipped:

```python
# add to tests/test_backends.py
from src.backends import resolve


def test_scip_supports_lazy() -> None:
    assert resolve("scip").supports_lazy is True
    assert resolve("highs").supports_lazy is False
    assert resolve("cp-sat").supports_lazy is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benders.py::test_lazy_reaches_toy_optimum_scip tests/test_backends.py::test_scip_supports_lazy -v`
Expected: FAIL — `scip` currently runs the loop (still passes optimum) but `supports_lazy` is `False`; the backends test fails, and after routing flips, the scip Benders test exercises the new callback path.

- [ ] **Step 3: Write minimal implementation**

In `src/backends.py`, set `supports_lazy=True` for scip and gurobi:

```python
    "scip": Backend("scip", ApiFamily.MATH_OPT, mathopt.SolverType.GSCIP, True, True),
    "gurobi": Backend("gurobi", ApiFamily.MATH_OPT, mathopt.SolverType.GUROBI, True, True),
```

In `src/benders.py`, add the lazy path and route to it:

```python
from ortools.math_opt.python import callback as cb


def _solve_lazy_mathopt(
    instance: Instance, backend: Backend, time_limit_s: float | None, *, pre_cuts: bool
) -> Result:
    assert backend.solver_type is not None
    start = time.perf_counter()
    ub = _full_capacity_maxflow(instance)
    master = _build_master_mathopt(instance, ub)
    evaluator = MinCutEvaluator(instance)
    arcs_with_jobs = frozenset(_jobs_on_arc(instance))
    periods = range(1, instance.horizon + 1)
    cut_count = 0

    def on_incumbent(data: cb.CallbackData) -> cb.CallbackResult:
        nonlocal cut_count
        vals = data.solution
        schedule: dict[str, int] = {}
        for job_id, window in master.starts.items():
            for s in window:
                if vals[master.x[(job_id, s)]] > 0.5:
                    schedule[job_id] = s
                    break
        res = cb.CallbackResult()
        for t in periods:
            cut = evaluator.evaluate(_closed_arcs(instance, schedule, t))
            if vals[master.theta[t]] > cut.flow_value + _EPS:
                res.add_lazy_constraint(
                    master.theta[t] <= _cut_rhs_mathopt(master, arcs_with_jobs, t, cut)
                )
                cut_count += 1
        return res

    reg = cb.CallbackRegistration(
        events={cb.Event.MIP_SOLUTION}, add_lazy_constraints=True
    )
    result = mathopt.solve(master.model, backend.solver_type, callback_reg=reg, cb=on_incumbent)
    if result.termination.reason is not mathopt.TerminationReason.OPTIMAL:
        return Result(
            method="benders", backend=backend.name, status=SolveStatus.UNKNOWN,
            objective=None, wall_time_s=time.perf_counter() - start, cut_count=cut_count,
        )
    schedule = _schedule_from_mathopt(result, master.starts, master.x)
    true_objective = sum(
        evaluator.evaluate(_closed_arcs(instance, schedule, t)).flow_value for t in periods
    )
    return Result(
        method="benders", backend=backend.name, status=SolveStatus.OPTIMAL,
        objective=float(true_objective), wall_time_s=time.perf_counter() - start, gap=0.0,
        schedule=schedule, iteration_count=1, cut_count=cut_count,
    )
```

Update the dispatch in `solve_benders`:

```python
def solve_benders(
    instance: Instance, backend: Backend, time_limit_s: float | None = None, *, pre_cuts: bool = False
) -> Result:
    if backend.family is ApiFamily.CP_SAT:
        return _solve_loop_cpsat(instance, backend, time_limit_s, pre_cuts=pre_cuts)
    if backend.supports_lazy:
        return _solve_lazy_mathopt(instance, backend, time_limit_s, pre_cuts=pre_cuts)
    return _solve_loop_mathopt(instance, backend, time_limit_s, pre_cuts=pre_cuts)
```

NOTE on the callback API: confirm the accessor for variable values on `CallbackData` (`data.solution` as a `Mapping[Variable, float]`). If the installed `ortools` exposes it under a different name, adjust the two `vals[...]` reads only — verify with `uv run python -c "from ortools.math_opt.python import callback; help(callback.CallbackData)"` before implementing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benders.py tests/test_backends.py -v`
Expected: PASS (all).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format src/benders.py tests/test_benders.py tests/test_backends.py
uv run ruff check --fix src/benders.py src/backends.py tests/test_benders.py tests/test_backends.py
uv run mypy src/benders.py src/backends.py
git add src/benders.py src/backends.py tests/test_benders.py tests/test_backends.py
git commit -m "feat: Benders lazy-callback injection on SCIP/Gurobi (3a)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Agreement matrix + K side-constraint coverage

**Files:**
- Test: `tests/test_benders.py`

**Interfaces:**
- Consumes: `solve_benders`, `solve_direct_mip`, `generate_instance`, `Regime`, `replace`.
- Produces: cross-backend, cross-method, and `K`-constraint tests. No source changes — if any fail, the failure is a real bug in earlier tasks (fix there, add a regression test).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_benders.py
from dataclasses import replace

from src.generator import Regime, generate_instance


@pytest.mark.parametrize("backend_name", ["highs", "scip", "cp-sat-m", "cp-sat"])
def test_benders_backends_agree_on_toy(backend_name: str) -> None:
    res = solve_benders(toy_instance(), resolve(backend_name))
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)


def test_benders_matches_direct_mip_on_generated_instance() -> None:
    inst = generate_instance(size_idx=1, list_idx=0, regime=Regime.TIGHT, seed=7)
    b = solve_benders(inst, resolve("cp-sat"), time_limit_s=60.0)
    d = solve_direct_mip(inst, resolve("cp-sat"), time_limit_s=60.0)
    assert b.status is d.status is SolveStatus.OPTIMAL
    assert b.objective == pytest.approx(d.objective)


def test_benders_k_one_keeps_toy_optimum() -> None:
    inst = replace(toy_instance(), max_jobs_per_period=1)
    res = solve_benders(inst, resolve("highs"))
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
```

- [ ] **Step 2: Run tests to verify status**

Run: `uv run pytest tests/test_benders.py -v`
Expected: `test_benders_k_one_keeps_toy_optimum` FAILS — the `≤K` master constraint is not built yet (Tasks 2/4 omit it).

- [ ] **Step 3: Add the optional `≤K` constraint to both masters**

In `_build_master_mathopt`, after the `y`/`theta` setup, before `maximize`:

```python
    if instance.max_jobs_per_period is not None:
        cap_k = instance.max_jobs_per_period
        for t in periods:
            total = sum(
                (in_progress(j, t) for j in instance.jobs), start=mathopt.LinearExpression()
            )
            model.add_linear_constraint(total <= cap_k)
```

In `_build_master_cpsat`, likewise:

```python
    if instance.max_jobs_per_period is not None:
        cap_k = instance.max_jobs_per_period
        for t in periods:
            model.add(
                cp_model.LinearExpr.sum([in_progress(j, t) for j in instance.jobs]) <= cap_k
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benders.py -v`
Expected: PASS (all). Then run the full suite: `uv run pytest -q` — expected all green.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format src/benders.py tests/test_benders.py
uv run ruff check --fix src/benders.py tests/test_benders.py
uv run mypy .
git add src/benders.py tests/test_benders.py
git commit -m "feat: Benders <=K constraint + cross-backend agreement tests (3a)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Bottleneck pre-cuts warm start (flag-gated)

**Files:**
- Modify: `src/benders.py`
- Test: `tests/test_benders.py`

**Interfaces:**
- Consumes: `_jobs_on_arc`, `MinCutEvaluator`; the two master builders.
- Produces:
  - `_bottleneck_precuts(instance) -> list[PeriodCut]` — peel successive min-cut-sets from the full-capacity network: solve max-flow, record the min-cut, raise those arcs' caps beyond the source's total out-capacity, repeat until the source-incident cut binds. Returns the distinct cut-sets (period-independent — they hold for every `t`).
  - Both `_solve_loop_*`/`_solve_lazy_*` honor `pre_cuts`: when set, add `θ_t ≤ Σ cap_a·y[a,t]` for every period and every pre-cut **before** solving.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_benders.py
def test_precuts_give_bottleneck_and_reach_optimum() -> None:
    from src.benders import _bottleneck_precuts
    cuts = _bottleneck_precuts(toy_instance())
    # the single bottleneck cut-set is arc (a,t) cap 2
    assert any(c.coeffs == {("a", "t"): 2} for c in cuts)
    res = solve_benders(toy_instance(), resolve("highs"), pre_cuts=True)
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_benders.py::test_precuts_give_bottleneck_and_reach_optimum -v`
Expected: FAIL with `ImportError: cannot import name '_bottleneck_precuts'`.

- [ ] **Step 3: Write minimal implementation**

```python
def _bottleneck_precuts(instance: Instance) -> list[PeriodCut]:
    """Peel min-cut-sets from the full-capacity network (Pearce & Forbes §3.1)."""
    caps = {(a.u, a.v): a.capacity for a in instance.arcs}
    big = sum(c for (u, v), c in caps.items() if u == instance.source) + 1
    cuts: list[PeriodCut] = []
    seen: set[frozenset[tuple[str, str]]] = set()
    for _ in range(len(caps)):
        g: nx.DiGraph = nx.DiGraph()
        for (u, v), cap in caps.items():
            g.add_edge(u, v, capacity=cap)
        value, (reachable, _) = nx.minimum_cut(g, instance.source, instance.sink)
        crossing = {(u, v) for (u, v) in caps if u in reachable and v not in reachable}
        key = frozenset(crossing)
        if key in seen:
            break
        seen.add(key)
        cuts.append(PeriodCut(flow_value=int(value), coeffs={a: caps[a] for a in crossing}))
        if all(u == instance.source for (u, v) in crossing):
            break  # the source-incident cut now binds; no tighter cut to find
        for a in crossing:
            caps[a] = big  # relax this bottleneck and look for the next
    return cuts


def _precut_constraints_mathopt(master: _Master, arcs_with_jobs: frozenset[tuple[str, str]],
                                periods: range, cuts: list[PeriodCut]) -> None:
    for t in periods:
        for cut in cuts:
            master.model.add_linear_constraint(
                master.theta[t] <= _cut_rhs_mathopt(master, arcs_with_jobs, t, cut)
            )
```

Call `_precut_constraints_mathopt(master, arcs_with_jobs, periods, _bottleneck_precuts(instance))` right after building the master in `_solve_loop_mathopt` and `_solve_lazy_mathopt` when `pre_cuts` is true. Add the analogous inline block in `_solve_loop_cpsat` (build the cut RHS with `cp_model.LinearExpr.sum`, as in Task 4).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_benders.py -v`
Expected: PASS (all). Full suite: `uv run pytest -q`.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format src/benders.py tests/test_benders.py
uv run ruff check --fix src/benders.py tests/test_benders.py
uv run mypy .
git add src/benders.py tests/test_benders.py
git commit -m "feat: Benders bottleneck pre-cuts warm start (flag-gated) (3a)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** master (Tasks 2/4) ✓; per-period max-flow subproblem via networkx (Task 1) ✓; disaggregated min-cut cut (Tasks 1/3) ✓; lazy callback on scip/gurobi + iterative loop on cp-sat/highs with `--backend` gating (Tasks 4/5) ✓; arc-availability proxy for overlapping jobs ✓ (Tasks 2/4); `≤K` master-only ✓ (Task 6); config caching + distinct-solve stats ✓ (Task 1); bottleneck pre-cuts warm start ✓ (Task 7); same-optimum/cross-backend agreement ✓ (Tasks 3/6); `Result` metrics `iteration_count`/`cut_count` ✓. Out of scope for 3a (own plans): LP-dual evaluator (3b), Pareto two-cut pick (3c), LP-relaxation warm start (documented TODO), `run.py`/benchmark wiring (Stage 4).

**Placeholder scan:** none — every step has concrete code/commands. The single explicit verification note (Task 5 callback accessor name) is an instruction to confirm one API name against the installed `ortools`, not a deferred implementation.

**Type consistency:** `PeriodCut(flow_value, coeffs)`, `MinCutEvaluator.evaluate(closed_arcs)`, `_Master`/`_MasterCp` fields, `_closed_arcs`, `_cut_rhs_mathopt`, `solve_benders(..., *, pre_cuts)` are referenced consistently across tasks. `solve_benders` signature matches `solve_direct_mip` plus the `pre_cuts` keyword.
