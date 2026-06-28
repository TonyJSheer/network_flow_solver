from __future__ import annotations

from dataclasses import replace

import pytest

from src.backends import resolve
from src.benders import _full_capacity_maxflow, solve_benders
from src.direct_mip import solve_direct_mip
from src.generator import Regime, generate_instance, toy_instance
from src.result import SolveStatus


def test_full_capacity_maxflow_toy() -> None:
    assert _full_capacity_maxflow(toy_instance()) == 2


def test_master_without_cuts_is_relaxation_bound() -> None:
    from ortools.math_opt.python import mathopt

    from src.benders import _build_master_mathopt

    master = _build_master_mathopt(toy_instance(), ub=2)
    result = mathopt.solve(master.model, resolve("highs").solver_type)
    assert result.objective_value() == pytest.approx(12.0)


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


def test_loop_reaches_toy_optimum_cpsat() -> None:
    res = solve_benders(toy_instance(), resolve("cp-sat"))
    assert res.backend == "cp-sat"
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
    assert res.schedule is not None and res.schedule["j0"] in range(1, 6)


def test_lazy_reaches_toy_optimum_scip() -> None:
    res = solve_benders(toy_instance(), resolve("scip"))
    assert res.backend == "scip"
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
    assert res.cut_count is not None and res.cut_count >= 1


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


def test_benders_k_zero_makes_instance_infeasible() -> None:
    # Every job must be scheduled exactly once => in progress at some period;
    # K=0 forbids any in-progress period, so the master is infeasible.
    # This BINDS: without the <=K constraint the model would be feasible/optimal.
    inst = replace(toy_instance(), max_jobs_per_period=0)
    res = solve_benders(inst, resolve("highs"))
    assert res.status is SolveStatus.INFEASIBLE
    assert res.objective is None


def test_benders_k_zero_infeasible_cpsat() -> None:
    # Same binding K=0 test, covering the CP-SAT iterative-loop path.
    inst = replace(toy_instance(), max_jobs_per_period=0)
    res = solve_benders(inst, resolve("cp-sat"))
    assert res.status is SolveStatus.INFEASIBLE
    assert res.objective is None


def test_precuts_give_bottleneck_and_reach_optimum() -> None:
    from src.benders import _bottleneck_precuts

    cuts = _bottleneck_precuts(toy_instance())
    # the single bottleneck cut-set is arc (a,t) cap 2
    assert any(c.coeffs == {("a", "t"): 2} for c in cuts)
    res = solve_benders(toy_instance(), resolve("highs"), pre_cuts=True)
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(8.0)
