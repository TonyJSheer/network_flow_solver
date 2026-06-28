"""CPLEX (docplex) solver tests. Deps are required, so no importorskip.

Toy instance known optimum is 8 (see tests/test_direct_mip.py): horizon 6 x
bottleneck capacity 2 = 12, minus the unavoidable 2-period outage of job j0 on
arc (a, t) (2 x 2) = 8.
"""

from __future__ import annotations

import pytest

from src.generator import Regime, generate_instance, toy_instance
from src.result import SolveStatus


def test_cplex_imports() -> None:
    import cplex  # noqa: F401
    from cplex.callbacks import LazyConstraintCallback  # noqa: F401
    from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin  # noqa: F401
    from docplex.mp.model import Model  # noqa: F401
