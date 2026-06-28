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
