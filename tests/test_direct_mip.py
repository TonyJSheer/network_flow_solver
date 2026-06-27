"""Direct MIP baseline tests. The toy instance has a hand-checked optimum of 8:
6 periods x bottleneck capacity 2 = 12, minus the unavoidable 2-period outage of
the single job on arc (a, t) (2 periods x 2) = 8.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.backends import resolve
from src.direct_mip import solve_direct_mip
from src.generator import Regime, generate_instance, toy_instance
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
