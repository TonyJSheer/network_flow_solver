"""CPLEX (docplex) solver tests. Deps are required, so no importorskip.

Toy instance known optimum is 8 (see tests/test_direct_mip.py): horizon 6 x
bottleneck capacity 2 = 12, minus the unavoidable 2-period outage of job j0 on
arc (a, t) (2 x 2) = 8.
"""

from __future__ import annotations

import pytest

from src.generator import toy_instance
from src.instance import Arc, Instance, Job
from src.result import SolveStatus


def test_cplex_imports() -> None:
    import cplex  # noqa: F401
    from cplex.callbacks import LazyConstraintCallback  # noqa: F401
    from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin  # noqa: F401
    from docplex.mp.model import Model  # noqa: F401


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


def test_benders_toy_optimum_and_cuts() -> None:
    from src.cplex_benders import solve_cplex_benders

    res = solve_cplex_benders(toy_instance())
    assert res.method == "benders"
    assert res.backend == "cplex"
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
    assert res.cut_count is not None and res.cut_count > 0  # lazy callback fired
    assert res.schedule is not None


def _small_two_arc_instance() -> Instance:
    # Two parallel s->*->t paths, each carrying one maintenance job. The scheduler
    # must stagger the two outages to keep flow up, so the per-period min-cut (and
    # thus Benders) genuinely varies across periods — a non-trivial agreement check
    # that still stays well under CPLEX Community Edition's 1000/1000 limit.
    return Instance(
        name="small-two-arc",
        horizon=6,
        source="s",
        sink="t",
        nodes=("s", "a", "b", "t"),
        arcs=(
            Arc("s", "a", 3),
            Arc("s", "b", 2),
            Arc("a", "t", 2),
            Arc("b", "t", 4),
        ),
        jobs=(
            Job("ja", ("a", "t"), 2, 1, 3),
            Job("jb", ("b", "t"), 2, 1, 3),
        ),
        seed=7,
        max_jobs_per_period=None,
        known_optimum=None,
    )


def test_cplex_mip_and_benders_agree() -> None:
    # Same problem, two CPLEX formulations: objective + status must match.
    # (Schedule may differ — multi-optimum pitfall.)
    #
    # NOTE (plan deviation): the plan used generate_instance(size_idx=1, ...), but
    # the generator's durations (10-30) across 5-15 sequential jobs per arc force a
    # horizon of hundreds of periods, so EVERY generated instance blows past CPLEX
    # Community Edition's 1000-var/1000-constraint cap. We use a hand-built small
    # instance instead, satisfying the plan's own global constraint #21.
    from src.cplex_benders import solve_cplex_benders
    from src.cplex_mip import solve_cplex_direct_mip

    inst = _small_two_arc_instance()
    mip = solve_cplex_direct_mip(inst, time_limit_s=60.0)
    ben = solve_cplex_benders(inst, time_limit_s=60.0)
    assert mip.status is SolveStatus.OPTIMAL
    assert ben.status is SolveStatus.OPTIMAL
    assert ben.objective == pytest.approx(mip.objective)
