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

import time
from dataclasses import dataclass

import networkx as nx
from ortools.math_opt.python import callback as cb
from ortools.math_opt.python import mathopt
from ortools.sat.python import cp_model

from src.backends import NUM_THREADS, ApiFamily, Backend
from src.instance import Instance, Job
from src.result import Result, SolveStatus
from src.subproblem import MinCutEvaluator, PeriodCut


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

    # Optional <=K constraint: at most K jobs in progress simultaneously per period.
    # This lives in the master only (it constrains the schedule, not the subproblem).
    if instance.max_jobs_per_period is not None:
        cap_k = instance.max_jobs_per_period
        for t in periods:
            total = sum(
                (in_progress(j, t) for j in instance.jobs), start=mathopt.LinearExpression()
            )
            model.add_linear_constraint(total <= cap_k)

    model.maximize(sum((theta[t] for t in periods), start=mathopt.LinearExpression()))
    return _Master(model=model, x=x, y=y, theta=theta, starts=starts)


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
        for s, lit in zip(starts[j.id], lits, strict=True):
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

    # Optional <=K constraint: at most K jobs in progress simultaneously per period.
    # This lives in the master only (it constrains the schedule, not the subproblem).
    if instance.max_jobs_per_period is not None:
        cap_k = instance.max_jobs_per_period
        for t in periods:
            model.add(cp_model.LinearExpr.sum([in_progress(j, t) for j in instance.jobs]) <= cap_k)

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
                method="benders",
                backend=backend.name,
                status=SolveStatus.UNKNOWN,
                objective=None,
                wall_time_s=time.perf_counter() - start,
                iteration_count=iterations,
                cut_count=cut_count,
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
        method="benders",
        backend=backend.name,
        status=SolveStatus.OPTIMAL,
        objective=float(true_objective),
        wall_time_s=time.perf_counter() - start,
        gap=0.0,
        schedule=schedule,
        iteration_count=iterations,
        cut_count=cut_count,
    )


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
    result: mathopt.SolveResult,
    starts: dict[str, range],
    x: dict[tuple[str, int], mathopt.Variable],
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
    cut: PeriodCut,
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
                method="benders",
                backend=backend.name,
                status=SolveStatus.UNKNOWN,
                objective=None,
                wall_time_s=time.perf_counter() - start,
                iteration_count=iterations,
                cut_count=cut_count,
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
        method="benders",
        backend=backend.name,
        status=SolveStatus.OPTIMAL,
        objective=true_objective,
        wall_time_s=time.perf_counter() - start,
        gap=0.0,
        schedule=schedule,
        iteration_count=iterations,
        cut_count=cut_count,
    )


def _solve_lazy_mathopt(
    instance: Instance, backend: Backend, time_limit_s: float | None, *, pre_cuts: bool
) -> Result:
    """Benders via MathOpt lazy-constraint callback (SCIP / Gurobi).

    At each MIP incumbent the callback evaluates all periods and injects a
    disaggregated cut for every violated theta_t. The solver never returns an
    infeasible schedule — it only terminates when no violated cut exists at the
    optimal incumbent.

    time_limit_s is accepted for API parity with the loop paths but is NOT
    wired into mathopt.solve; time-limit wiring is deferred to Stage 4.
    """
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

    reg = cb.CallbackRegistration(events={cb.Event.MIP_SOLUTION}, add_lazy_constraints=True)
    result = mathopt.solve(master.model, backend.solver_type, callback_reg=reg, cb=on_incumbent)
    if result.termination.reason is not mathopt.TerminationReason.OPTIMAL:
        return Result(
            method="benders",
            backend=backend.name,
            status=SolveStatus.UNKNOWN,
            objective=None,
            wall_time_s=time.perf_counter() - start,
            cut_count=cut_count,
        )
    schedule = _schedule_from_mathopt(result, master.starts, master.x)
    true_objective = sum(
        evaluator.evaluate(_closed_arcs(instance, schedule, t)).flow_value for t in periods
    )
    return Result(
        method="benders",
        backend=backend.name,
        status=SolveStatus.OPTIMAL,
        objective=float(true_objective),
        wall_time_s=time.perf_counter() - start,
        gap=0.0,
        schedule=schedule,
        iteration_count=1,
        cut_count=cut_count,
    )


def solve_benders(
    instance: Instance,
    backend: Backend,
    time_limit_s: float | None = None,
    *,
    pre_cuts: bool = False,
) -> Result:
    """Solve via disaggregated Benders.

    Dispatch:
    - CP-SAT family → iterative cut loop (CP-SAT has no lazy callbacks).
    - MathOpt + supports_lazy → lazy-constraint callback (SCIP / Gurobi).
    - MathOpt + not supports_lazy → iterative re-solve loop (HiGHS).
    """
    if backend.family is ApiFamily.CP_SAT:
        return _solve_loop_cpsat(instance, backend, time_limit_s, pre_cuts=pre_cuts)
    if backend.supports_lazy:
        return _solve_lazy_mathopt(instance, backend, time_limit_s, pre_cuts=pre_cuts)
    return _solve_loop_mathopt(instance, backend, time_limit_s, pre_cuts=pre_cuts)
