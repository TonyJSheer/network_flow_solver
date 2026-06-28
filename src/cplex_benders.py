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
                    cap for arc, cap in cut.coeffs.items() if arc not in self._arcs_with_jobs
                )
                cpx_lhs, sense, cpx_rhs = self.linear_ct_to_cplex(self._theta[t] <= rhs + const)
                self.add(cpx_lhs, sense, cpx_rhs)
                self.cut_count += 1


def solve_cplex_benders(instance: Instance, time_limit_s: float | None = None) -> Result:
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
