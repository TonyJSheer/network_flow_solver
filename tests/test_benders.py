from __future__ import annotations

import pytest

from src.backends import resolve
from src.benders import _full_capacity_maxflow, solve_benders
from src.direct_mip import solve_direct_mip
from src.generator import toy_instance
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
