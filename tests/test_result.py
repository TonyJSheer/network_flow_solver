import dataclasses

import pytest

from src.result import Result, SolveStatus


def test_result_minimal_construction_defaults_optionals_to_none() -> None:
    r = Result(
        method="direct_mip",
        backend="cp-sat",
        status=SolveStatus.OPTIMAL,
        objective=17.0,
        wall_time_s=0.42,
    )
    assert r.objective == 17.0
    assert r.gap is None
    assert r.schedule is None
    assert r.node_count is None
    assert r.iteration_count is None
    assert r.cut_count is None


def test_result_is_frozen() -> None:
    r = Result(
        method="benders",
        backend="scip",
        status=SolveStatus.FEASIBLE,
        objective=None,
        wall_time_s=1.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.objective = 5.0  # type: ignore[misc]  # asserting frozen behavior


def test_solve_status_has_exactly_four_members() -> None:
    assert {s.name for s in SolveStatus} == {
        "OPTIMAL",
        "FEASIBLE",
        "INFEASIBLE",
        "UNKNOWN",
    }
