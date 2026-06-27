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

Backend note: MathOpt backends here use continuous flow vars; the native CP-SAT
integer-flow build is Stage 2.1.
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


def _solve_mathopt(instance: Instance, backend: Backend, time_limit_s: float | None) -> Result:
    # every MathOpt backend has a SolverType; this guards the type narrowing below
    if backend.solver_type is None:
        raise ValueError(f"backend {backend.name!r} has no MathOpt SolverType")
    periods = range(1, instance.horizon + 1)
    model = mathopt.Model(name=f"direct_mip:{instance.name}")

    # Flow vars for every real arc plus the return arc (sink -> source). The
    # return arc is uncapacitated in effect: its cap is an upper bound on any
    # feasible throughput (total capacity leaving the source). Variable upper
    # bounds enforce the plain capacity limit f[a,t] <= cap_a for all arcs.
    return_cap = sum(a.capacity for a in instance.arcs if a.u == instance.source)
    arc_caps: dict[tuple[str, str], int] = {(a.u, a.v): a.capacity for a in instance.arcs}
    if (instance.sink, instance.source) in arc_caps:
        raise ValueError(
            "instance declares a real (sink, source) arc; that key is reserved for the return arc"
        )
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
        params = mathopt.SolveParameters(time_limit=datetime.timedelta(seconds=time_limit_s))
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
    # gap denominates by incumbent |primal| rather than |dual|, for divide-by-zero safety
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
