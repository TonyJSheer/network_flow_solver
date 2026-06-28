from __future__ import annotations

import pytest

from src.backends import resolve
from src.benders import _full_capacity_maxflow, solve_benders
from src.generator import toy_instance
from src.result import SolveStatus


def test_full_capacity_maxflow_toy() -> None:
    assert _full_capacity_maxflow(toy_instance()) == 2


def test_master_without_cuts_is_relaxation_bound() -> None:
    # No cuts yet: each theta_t hits its UB (=2), summed over horizon 6 => 12.
    # (The true optimum is 8; cuts in later tasks bring it down.)
    res = solve_benders(toy_instance(), resolve("highs"))
    assert res.method == "benders"
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(12.0)
