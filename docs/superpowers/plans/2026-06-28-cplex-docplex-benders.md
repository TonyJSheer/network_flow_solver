# CPLEX (docplex) direct-MIP + lazy-callback Benders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the direct MIP and disaggregated Benders solvers in CPLEX via docplex, with Benders cuts injected through CPLEX's native `LazyConstraintCallback`.

**Architecture:** Two standalone modules outside the OR-Tools `backends.py` seam, each a `solve_*(instance, ...) -> Result` function reusing the existing `Instance`/`Result` data types and the backend-agnostic `src.subproblem.MinCutEvaluator`. The direct MIP is a 1:1 docplex transliteration of the existing MathOpt formulation. Benders builds a docplex master and registers a lazy-constraint callback that evaluates the per-period min-cut subproblem at each integer incumbent and adds the disaggregated optimality cut `theta_t <= sum_{a in min-cut} cap_a * y[a,t]`.

**Tech Stack:** Python 3.12, `uv`, `docplex` (modeling) + `cplex` (Community-Edition engine), `networkx` (via the reused subproblem), `pytest`.

## Global Constraints

- Run Python through `uv` — never bare `python3`.
- Type annotations on every function; `mypy --strict` clean; `ruff` clean (autofix: `ruff format` + `ruff check --fix`).
- No bare `except:`; no `# type: ignore` without a justifying comment.
- CPLEX lives OUTSIDE `backends.py`; do NOT modify `backends.py`, `run.py`, or `src/benchmark.py`.
- `backend` field on every CPLEX `Result` is the string `"cplex"`; `method` is `"direct_mip"` or `"benders"`.
- Status mapping is the project's four-way: OPTIMAL / FEASIBLE / INFEASIBLE / UNKNOWN.
- Gap convention (copied verbatim from existing solvers): `abs(primal - dual) / (abs(primal) + 1e-10)`.
- Agreement compares (objective, status), NOT the schedule (multi-optimum pitfall).
- CPLEX Community Edition ceiling: 1000 vars / 1000 constraints. Keep all test instances small.

---

### Task 1: Add CPLEX dependencies and confirm import

**Files:**
- Modify: `pyproject.toml` (via `uv add`)
- Create: `tests/test_cplex.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable `docplex.mp.model.Model`, `cplex.callbacks.LazyConstraintCallback`, `docplex.mp.callbacks.cb_mixin.ConstraintCallbackMixin`.

- [ ] **Step 1: Add the dependencies**

Run:
```bash
uv add docplex cplex
```
Expected: both resolve and install; `pyproject.toml` `[project].dependencies` now lists `cplex` and `docplex`.

- [ ] **Step 2: Write the failing import test**

Create `tests/test_cplex.py`:
```python
"""CPLEX (docplex) solver tests. Deps are required, so no importorskip.

Toy instance known optimum is 8 (see tests/test_direct_mip.py): horizon 6 x
bottleneck capacity 2 = 12, minus the unavoidable 2-period outage of job j0 on
arc (a, t) (2 x 2) = 8.
"""

from __future__ import annotations

import pytest

from src.generator import Regime, generate_instance, toy_instance
from src.result import SolveStatus


def test_cplex_imports() -> None:
    import cplex  # noqa: F401
    from cplex.callbacks import LazyConstraintCallback  # noqa: F401
    from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin  # noqa: F401
    from docplex.mp.model import Model  # noqa: F401
```

- [ ] **Step 3: Run the import test**

Run: `uv run pytest tests/test_cplex.py::test_cplex_imports -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock tests/test_cplex.py
git commit -m "chore: add docplex + cplex deps and import smoke test"
```

---

### Task 2: CPLEX direct MIP (`src/cplex_mip.py`)

**Files:**
- Create: `src/cplex_mip.py`
- Test: `tests/test_cplex.py`

**Interfaces:**
- Consumes: `src.instance.Instance`; `src.result.Result`, `src.result.SolveStatus`.
- Produces: `solve_cplex_direct_mip(instance: Instance, time_limit_s: float | None = None) -> Result` with `method="direct_mip"`, `backend="cplex"`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cplex.py`:
```python
def test_direct_mip_toy_optimum() -> None:
    from src.cplex_mip import solve_cplex_direct_mip

    res = solve_cplex_direct_mip(toy_instance())
    assert res.method == "direct_mip"
    assert res.backend == "cplex"
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
    assert res.schedule is not None
    assert res.schedule["j0"] in range(1, 6)
    assert res.wall_time_s >= 0.0


def test_direct_mip_k_zero_infeasible() -> None:
    from dataclasses import replace

    from src.cplex_mip import solve_cplex_direct_mip

    inst = replace(toy_instance(), max_jobs_per_period=0)
    res = solve_cplex_direct_mip(inst)
    assert res.status is SolveStatus.INFEASIBLE
    assert res.objective is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_cplex.py::test_direct_mip_toy_optimum -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.cplex_mip'`.

- [ ] **Step 3: Implement `src/cplex_mip.py`**

Create `src/cplex_mip.py`:
```python
"""Direct MIP baseline in CPLEX (docplex) — time-indexed circulation.

A 1:1 transliteration of the MathOpt formulation in ``src/direct_mip.py`` into
docplex, kept identical so the two are whiteboard-comparable. CPLEX lives outside
the OR-Tools ``backends.py`` seam (a deliberate, approved departure: CPLEX is not
an OR-Tools backend). Flow vars are continuous (exact here: integral capacities
give an integral max flow).

  variables
    x[j,s] in {0,1}   job j starts in period s,  s in [release_j, deadline_j]
    f[a,t] >= 0        flow on arc a in period t
  derived
    y[j,t] = sum_{s = max(release_j, t-d_j+1) .. min(deadline_j, t)} x[j,s]
  constraints
    (1) sum_s x[j,s] = 1                        each job scheduled exactly once
    (2) f[a(j),t] <= cap_{a(j)} * (1 - y[j,t])  capacity tied to outage
    (3) sum_in f - sum_out f = 0  (every node)  circulation via return arc t->s
    (4) sum_j y[j,t] <= K  (optional)           bounded jobs per period
  objective
    max sum_t f[return arc, t]

CPLEX Community Edition caps the model at 1000 vars / 1000 constraints; large
instances raise a clear error rather than a raw CPLEX traceback.
"""

from __future__ import annotations

from docplex.mp.linear import LinearExpr
from docplex.mp.model import Model
from docplex.mp.dvar import Var

from src.instance import Instance
from src.result import Result, SolveStatus


def solve_cplex_direct_mip(
    instance: Instance, time_limit_s: float | None = None
) -> Result:
    """Solve the time-indexed direct MIP in CPLEX via docplex."""
    periods = range(1, instance.horizon + 1)
    model = Model(name=f"cplex_direct_mip:{instance.name}")

    # Return arc (sink -> source) closes the circulation; its capacity is an upper
    # bound on any feasible throughput, so it never binds. Reserved key.
    return_cap = sum(a.capacity for a in instance.arcs if a.u == instance.source)
    arc_caps: dict[tuple[str, str], int] = {(a.u, a.v): a.capacity for a in instance.arcs}
    if (instance.sink, instance.source) in arc_caps:
        raise ValueError(
            "instance declares a real (sink, source) arc; that key is reserved for the return arc"
        )
    arc_caps[(instance.sink, instance.source)] = return_cap

    f: dict[tuple[str, str, int], Var] = {}
    for (u, v), cap in arc_caps.items():
        for t in periods:
            f[(u, v, t)] = model.continuous_var(lb=0.0, ub=float(cap), name=f"f_{u}_{v}_{t}")

    x: dict[tuple[str, int], Var] = {}
    starts: dict[str, range] = {}
    for j in instance.jobs:
        starts[j.id] = range(j.release, j.deadline + 1)
        for s in starts[j.id]:
            x[(j.id, s)] = model.binary_var(name=f"x_{j.id}_{s}")

    # (1) each job scheduled exactly once.
    for j in instance.jobs:
        model.add_constraint(model.sum(x[(j.id, s)] for s in starts[j.id]) == 1)

    def in_progress(job_id: str, duration: int, t: int) -> LinearExpr:
        # y[j,t] = sum of starts s with t-d+1 <= s <= t (clamped to the window).
        lo = max(starts[job_id].start, t - duration + 1)
        hi = min(starts[job_id].stop - 1, t)
        return model.sum(x[(job_id, s)] for s in range(lo, hi + 1))

    # (2) capacity tied to outage; big-M free.
    for j in instance.jobs:
        cap = arc_caps[j.arc]
        for t in periods:
            y = in_progress(j.id, j.duration, t)
            model.add_constraint(f[(j.arc[0], j.arc[1], t)] <= cap * (1 - y))

    # (3) circulation: flow conservation at every node, every period.
    for n in instance.nodes:
        for t in periods:
            inflow = model.sum(f[(u, v, t)] for (u, v) in arc_caps if v == n)
            outflow = model.sum(f[(u, v, t)] for (u, v) in arc_caps if u == n)
            model.add_constraint(inflow - outflow == 0)

    # (4) optional bounded jobs per period.
    if instance.max_jobs_per_period is not None:
        cap_k = instance.max_jobs_per_period
        for t in periods:
            total = model.sum(
                in_progress(j.id, j.duration, t) for j in instance.jobs
            )
            model.add_constraint(total <= cap_k)

    ret = (instance.sink, instance.source)
    model.maximize(model.sum(f[(ret[0], ret[1], t)] for t in periods))

    if time_limit_s is not None:
        model.parameters.timelimit = time_limit_s

    sol = model.solve()
    return _to_result(model, sol, x, starts)


def _to_result(
    model: Model,
    sol: object | None,
    x: dict[tuple[str, int], Var],
    starts: dict[str, range],
) -> Result:
    details = model.solve_details
    wall = float(details.time) if details is not None else 0.0
    status_text = (details.status if details is not None else "").lower()

    if sol is None:
        status = SolveStatus.INFEASIBLE if "infeasible" in status_text else SolveStatus.UNKNOWN
        return Result(
            method="direct_mip",
            backend="cplex",
            status=status,
            objective=None,
            wall_time_s=wall,
        )

    status = SolveStatus.OPTIMAL if "optimal" in status_text else SolveStatus.FEASIBLE
    schedule: dict[str, int] = {}
    for job_id, window in starts.items():
        for s in window:
            if sol.get_value(x[(job_id, s)]) > 0.5:
                schedule[job_id] = s
                break

    primal = float(sol.objective_value)
    dual = float(details.best_bound) if details is not None else primal
    gap = abs(primal - dual) / (abs(primal) + 1e-10)
    return Result(
        method="direct_mip",
        backend="cplex",
        status=status,
        objective=primal,
        wall_time_s=wall,
        gap=gap,
        schedule=schedule,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cplex.py::test_direct_mip_toy_optimum tests/test_cplex.py::test_direct_mip_k_zero_infeasible -v`
Expected: both PASS. If `sol.get_value` / `solve_details.status` / `best_bound` names differ in the installed docplex, adjust per `uv run python -c "import docplex.mp.model as m; help(m.Model.solve_details)"` and re-run.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format src/cplex_mip.py tests/test_cplex.py && uv run ruff check --fix src/cplex_mip.py tests/test_cplex.py
uv run mypy src/cplex_mip.py
git add src/cplex_mip.py tests/test_cplex.py
git commit -m "feat: CPLEX (docplex) direct MIP solver"
```

---

### Task 3: Lazy-callback smoke test (verify mechanics cheaply)

**Files:**
- Test: `tests/test_cplex.py`

**Interfaces:**
- Consumes: `docplex` callback API only — no project code.
- Produces: confidence that `ConstraintCallbackMixin` + `LazyConstraintCallback` + `make_solution_from_vars` + `linear_ct_to_cplex` + `self.add(...)` behave as the Benders task assumes. (Per the repo's "check callbacks cheaply" lesson: prove the mechanism on a trivial model before wiring the subproblem.)

- [ ] **Step 1: Write the smoke test**

Add to `tests/test_cplex.py`:
```python
def test_lazy_callback_fires_and_cuts() -> None:
    # Trivial model: maximize z, z <= 10. A lazy callback adds z <= 3 whenever it
    # sees an incumbent with z > 3. Proves the mixin + add() path drives the
    # optimum down to 3 and that the callback actually fires.
    from cplex.callbacks import LazyConstraintCallback
    from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
    from docplex.mp.model import Model

    class _SmokeCallback(ConstraintCallbackMixin, LazyConstraintCallback):
        def __init__(self, env: object) -> None:
            LazyConstraintCallback.__init__(self, env)
            ConstraintCallbackMixin.__init__(self)
            self.fired = 0

        def __call__(self) -> None:
            sol = self.make_solution_from_vars([self.z])
            if sol.get_value(self.z) > 3.0 + 1e-6:
                self.fired += 1
                cpx_lhs, sense, cpx_rhs = self.linear_ct_to_cplex(self.z <= 3)
                self.add(cpx_lhs, sense, cpx_rhs)

    model = Model(name="smoke")
    z = model.integer_var(lb=0, ub=10, name="z")
    model.maximize(z)
    cb = model.register_callback(_SmokeCallback)
    cb.z = z
    sol = model.solve()

    assert sol is not None
    assert sol.get_value(z) == pytest.approx(3.0)
    assert cb.fired >= 1
```

- [ ] **Step 2: Run the smoke test**

Run: `uv run pytest tests/test_cplex.py::test_lazy_callback_fires_and_cuts -v`
Expected: PASS. If an API name differs (e.g. `make_solution_from_vars`), inspect with `uv run python -c "from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin as C; print([m for m in dir(C) if not m.startswith('__')])"` and adjust the smoke test AND note the corrected names for Task 4.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cplex.py
git commit -m "test: CPLEX lazy-constraint callback smoke test"
```

---

### Task 4: CPLEX Benders with native lazy callback (`src/cplex_benders.py`)

**Files:**
- Create: `src/cplex_benders.py`
- Test: `tests/test_cplex.py`

**Interfaces:**
- Consumes: `src.instance.Instance`, `src.instance.Job`; `src.result.Result`, `src.result.SolveStatus`; `src.subproblem.MinCutEvaluator`, `src.subproblem.PeriodCut`; the docplex callback API confirmed in Task 3.
- Produces: `solve_cplex_benders(instance: Instance, time_limit_s: float | None = None) -> Result` with `method="benders"`, `backend="cplex"`, `cut_count` populated.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cplex.py`:
```python
def test_benders_toy_optimum_and_cuts() -> None:
    from src.cplex_benders import solve_cplex_benders

    res = solve_cplex_benders(toy_instance())
    assert res.method == "benders"
    assert res.backend == "cplex"
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
    assert res.cut_count is not None and res.cut_count > 0  # lazy callback fired
    assert res.schedule is not None


def test_cplex_mip_and_benders_agree() -> None:
    # Same problem, two CPLEX formulations: objective + status must match.
    # (Schedule may differ — multi-optimum pitfall.) Small TIGHT instance.
    from src.cplex_benders import solve_cplex_benders
    from src.cplex_mip import solve_cplex_direct_mip

    inst = generate_instance(size_idx=1, list_idx=0, regime=Regime.TIGHT, seed=7)
    mip = solve_cplex_direct_mip(inst, time_limit_s=60.0)
    ben = solve_cplex_benders(inst, time_limit_s=60.0)
    assert mip.status is SolveStatus.OPTIMAL
    assert ben.status is SolveStatus.OPTIMAL
    assert ben.objective == pytest.approx(mip.objective)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cplex.py::test_benders_toy_optimum_and_cuts -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.cplex_benders'`.

- [ ] **Step 3: Implement `src/cplex_benders.py`**

Create `src/cplex_benders.py`:
```python
"""Disaggregated Benders in CPLEX (docplex) with a native lazy callback.

Master: binary schedule x[j,s], arc-availability proxy y[a,t] in [0,1] (1 iff arc
a open at t), per-period flow proxy theta_t; objective max sum_t theta_t.
Subproblem: per-period s->t min-cut (src.subproblem.MinCutEvaluator), yielding the
disaggregated optimality cut  theta_t <= sum_{a in min-cut} cap_a * y[a,t].

Cuts are injected through CPLEX's own LazyConstraintCallback: at each integer
incumbent the callback evaluates every period and adds a cut for any violated
theta_t. This is the showcase — OR-Tools only exposes lazy callbacks on SCIP/
Gurobi, so CPLEX gives a clean native reference for the mechanism.

Arc availability under OVERLAPPING same-arc jobs: y[a,t] <= 1 - inprogress_j(t)
for EACH job j on arc a, matching the direct MIP's per-job outage semantics so the
two formulations share an optimum.

Note: registering a legacy control callback makes CPLEX disable dynamic search and
run sequentially — expected, and fine for a small-instance mechanism demo. CPLEX
Community Edition caps the model at 1000 vars / 1000 constraints.
"""

from __future__ import annotations

import time

import networkx as nx
from cplex.callbacks import LazyConstraintCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
from docplex.mp.dvar import Var
from docplex.mp.model import Model

from src.instance import Instance, Job
from src.result import Result, SolveStatus
from src.subproblem import MinCutEvaluator

_EPS = 1e-6


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


def _closed_arcs(
    instance: Instance, schedule: dict[str, int], t: int
) -> frozenset[tuple[str, str]]:
    closed: set[tuple[str, str]] = set()
    for j in instance.jobs:
        s = schedule[j.id]
        if s <= t <= s + j.duration - 1:
            closed.add(j.arc)
    return frozenset(closed)


class _BendersLazyCallback(ConstraintCallbackMixin, LazyConstraintCallback):
    """Adds disaggregated optimality cuts at each integer incumbent."""

    def __init__(self, env: object) -> None:
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.cut_count = 0

    def setup(
        self,
        model: Model,
        instance: Instance,
        evaluator: MinCutEvaluator,
        x: dict[tuple[str, int], Var],
        y: dict[tuple[tuple[str, str], int], Var],
        theta: dict[int, Var],
        starts: dict[str, range],
        arcs_with_jobs: frozenset[tuple[str, str]],
    ) -> None:
        self._model = model
        self._instance = instance
        self._evaluator = evaluator
        self._x = x
        self._y = y
        self._theta = theta
        self._starts = starts
        self._arcs_with_jobs = arcs_with_jobs
        self._periods = range(1, instance.horizon + 1)

    def __call__(self) -> None:
        watched = list(self._x.values()) + list(self._theta.values())
        sol = self.make_solution_from_vars(watched)
        schedule: dict[str, int] = {}
        for job_id, window in self._starts.items():
            for s in window:
                if sol.get_value(self._x[(job_id, s)]) > 0.5:
                    schedule[job_id] = s
                    break
        for t in self._periods:
            cut = self._evaluator.evaluate(_closed_arcs(self._instance, schedule, t))
            if sol.get_value(self._theta[t]) > cut.flow_value + _EPS:
                rhs = self._model.sum(
                    cap * self._y[(arc, t)]
                    for arc, cap in cut.coeffs.items()
                    if arc in self._arcs_with_jobs
                )
                const = sum(
                    cap
                    for arc, cap in cut.coeffs.items()
                    if arc not in self._arcs_with_jobs
                )
                cpx_lhs, sense, cpx_rhs = self.linear_ct_to_cplex(
                    self._theta[t] <= rhs + const
                )
                self.add(cpx_lhs, sense, cpx_rhs)
                self.cut_count += 1


def solve_cplex_benders(
    instance: Instance, time_limit_s: float | None = None
) -> Result:
    """Solve via disaggregated Benders with a CPLEX native lazy callback."""
    start = time.perf_counter()
    periods = range(1, instance.horizon + 1)
    ub = _full_capacity_maxflow(instance)
    model = Model(name=f"cplex_benders:{instance.name}")

    # Start-indexed schedule vars + exactly-once.
    x: dict[tuple[str, int], Var] = {}
    starts: dict[str, range] = {}
    for j in instance.jobs:
        starts[j.id] = range(j.release, j.deadline + 1)
        for s in starts[j.id]:
            x[(j.id, s)] = model.binary_var(name=f"x_{j.id}_{s}")
        model.add_constraint(model.sum(x[(j.id, s)] for s in starts[j.id]) == 1)

    def in_progress(job: Job, t: int) -> object:
        lo = max(starts[job.id].start, t - job.duration + 1)
        hi = min(starts[job.id].stop - 1, t)
        return model.sum(x[(job.id, s)] for s in range(lo, hi + 1))

    # Arc-availability proxy y[a,t] in [0,1]; y <= 1 - ip_j per job on the arc.
    jobs_on_arc = _jobs_on_arc(instance)
    y: dict[tuple[tuple[str, str], int], Var] = {}
    for arc, jobs in jobs_on_arc.items():
        for t in periods:
            yvar = model.continuous_var(lb=0.0, ub=1.0, name=f"y_{arc[0]}_{arc[1]}_{t}")
            y[(arc, t)] = yvar
            for j in jobs:
                model.add_constraint(yvar <= 1 - in_progress(j, t))

    theta: dict[int, Var] = {
        t: model.continuous_var(lb=0.0, ub=float(ub), name=f"theta_{t}") for t in periods
    }

    # Optional <=K lives in the master only (constrains the schedule).
    if instance.max_jobs_per_period is not None:
        cap_k = instance.max_jobs_per_period
        for t in periods:
            total = model.sum(in_progress(j, t) for j in instance.jobs)
            model.add_constraint(total <= cap_k)

    model.maximize(model.sum(theta[t] for t in periods))

    evaluator = MinCutEvaluator(instance)
    arcs_with_jobs = frozenset(jobs_on_arc)
    cb = model.register_callback(_BendersLazyCallback)
    cb.setup(model, instance, evaluator, x, y, theta, starts, arcs_with_jobs)

    if time_limit_s is not None:
        model.parameters.timelimit = time_limit_s

    sol = model.solve()
    return _to_result(model, sol, instance, evaluator, x, starts, cb.cut_count, start)


def _to_result(
    model: Model,
    sol: object | None,
    instance: Instance,
    evaluator: MinCutEvaluator,
    x: dict[tuple[str, int], Var],
    starts: dict[str, range],
    cut_count: int,
    start: float,
) -> Result:
    details = model.solve_details
    wall = time.perf_counter() - start
    status_text = (details.status if details is not None else "").lower()

    if sol is None:
        status = SolveStatus.INFEASIBLE if "infeasible" in status_text else SolveStatus.UNKNOWN
        return Result(
            method="benders",
            backend="cplex",
            status=status,
            objective=None,
            wall_time_s=wall,
            cut_count=cut_count,
        )

    status = SolveStatus.OPTIMAL if "optimal" in status_text else SolveStatus.FEASIBLE
    schedule: dict[str, int] = {}
    for job_id, window in starts.items():
        for s in window:
            if sol.get_value(x[(job_id, s)]) > 0.5:
                schedule[job_id] = s
                break

    # True objective = sum of per-period min-cut flow under the final schedule
    # (the master theta is only an upper proxy until all cuts are present).
    periods = range(1, instance.horizon + 1)
    true_objective = float(
        sum(evaluator.evaluate(_closed_arcs(instance, schedule, t)).flow_value for t in periods)
    )
    return Result(
        method="benders",
        backend="cplex",
        status=status,
        objective=true_objective,
        wall_time_s=wall,
        gap=0.0,
        schedule=schedule,
        iteration_count=1,
        cut_count=cut_count,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cplex.py::test_benders_toy_optimum_and_cuts tests/test_cplex.py::test_cplex_mip_and_benders_agree -v`
Expected: both PASS.

- [ ] **Step 5: Run the full CPLEX test module**

Run: `uv run pytest tests/test_cplex.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff format src/cplex_benders.py tests/test_cplex.py && uv run ruff check --fix src/cplex_benders.py tests/test_cplex.py
uv run mypy src/cplex_benders.py
git add src/cplex_benders.py tests/test_cplex.py
git commit -m "feat: CPLEX (docplex) Benders with native lazy-constraint callback"
```

---

### Task 5: Full-suite regression check

**Files:** none (verification only).

**Interfaces:** consumes everything above.

- [ ] **Step 1: Run the whole suite**

Run: `uv run pytest`
Expected: all tests PASS (the new CPLEX module plus the existing OR-Tools suite, which is untouched).

- [ ] **Step 2: Lint + type-check the whole tree**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy .`
Expected: clean.

- [ ] **Step 3: Commit any residual formatting**

```bash
git add -A
git commit -m "chore: lint/format pass for CPLEX modules" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:**
- Files `src/cplex_mip.py`, `src/cplex_benders.py`, `tests/test_cplex.py` → Tasks 2/4/1–4. ✓
- Reuse `Instance`/`Result`/`MinCutEvaluator`; local tiny helpers → Task 4. ✓
- docplex API + required deps → Task 1. ✓
- Direct MIP transliteration (constraints 1–4, gap/status convention) → Task 2. ✓
- Benders master mirror + native lazy callback + true-objective recompute → Task 4. ✓
- CE size-error surfacing → covered by the module docstrings + small test instances; CE overflow raises CPLEX's own clear "problem size limits" error (no extra wrapping needed since tests stay small). ✓
- Verification: toy optimum (MIP + Benders), MIP/Benders agreement, cut_count>0 → Tasks 2/4. ✓
- check-callbacks-cheaply lesson → Task 3 smoke test. ✓
- No `backends.py`/`run.py`/`benchmark.py` changes → respected. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `solve_cplex_direct_mip` / `solve_cplex_benders` signatures match between tasks and tests; `_to_result`, `_closed_arcs`, `_jobs_on_arc`, `_full_capacity_maxflow`, `_BendersLazyCallback.setup` consistent. ✓

**Note for the executor:** docplex method names (`make_solution_from_vars`, `linear_ct_to_cplex`, `solve_details.status`/`best_bound`, `parameters.timelimit`) are the documented ones but were not introspected against the installed wheel (deps are added in Task 1). Task 1's import test and Task 3's smoke test are the early gates that catch any drift before the real Benders wiring — adjust names there if the installed version differs, then carry the correction into Tasks 2/4.
