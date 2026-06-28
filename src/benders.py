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
