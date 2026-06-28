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

Backend note: the MathOpt backends build the continuous-flow transliteration above
(also reachable as the cp-sat-m backend). The native CP-SAT backend (_solve_cp_sat)
builds a CP-idiomatic model — presence literals + optional intervals, exactly-one,
enforcement-literal outage, and add_cumulative for <=K — with integer flow vars.
Integer is exact here (integral capacities give an integral max flow), so every
backend reaches the same optimum; the cross-backend agreement tests enforce it.
"""

from __future__ import annotations

import datetime

from ortools.math_opt.python import mathopt
from ortools.sat.python import cp_model

from src.backends import NUM_THREADS, ApiFamily, Backend
from src.instance import Instance
from src.result import Result, SolveStatus


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

    params = mathopt.SolveParameters(
        # HiGHS rejects the threads param via MathOpt; use NUM_THREADS only where supported.
        threads=NUM_THREADS if backend.supports_threads_param else None,
        time_limit=datetime.timedelta(seconds=time_limit_s) if time_limit_s is not None else None,
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
