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

from docplex.mp.dvar import Var
from docplex.mp.linear import LinearExpr
from docplex.mp.model import Model

from src.instance import Instance
from src.result import Result, SolveStatus


def solve_cplex_direct_mip(instance: Instance, time_limit_s: float | None = None) -> Result:
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
            total = model.sum(in_progress(j.id, j.duration, t) for j in instance.jobs)
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
            # docplex ships no stubs; sol is narrowed from object (not None) above
            if sol.get_value(x[(job_id, s)]) > 0.5:  # type: ignore[attr-defined]
                schedule[job_id] = s
                break

    # docplex ships no stubs; sol is narrowed from object (not None) above
    primal = float(sol.objective_value)  # type: ignore[attr-defined]
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
