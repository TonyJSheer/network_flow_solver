"""Direct MIP baseline tests. The toy instance has a hand-checked optimum of 8:
6 periods x bottleneck capacity 2 = 12, minus the unavoidable 2-period outage of
the single job on arc (a, t) (2 periods x 2) = 8.
"""

from __future__ import annotations

import pytest

from src.backends import resolve
from src.direct_mip import solve_direct_mip
from src.generator import toy_instance
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
