"""Backend selection — the one seam that knows about specific solvers.

Resolves a ``--backend`` name to its OR-Tools API family plus two capability
flags the formulation code branches on:
- ``continuous_flow``: MathOpt MIP backends use continuous flow vars; CP-SAT is
  integer-only (exact here, since integral capacities give an integral max flow).
- ``supports_lazy``: whether lazy-constraint callbacks are available, deciding
  the Benders cut-injection path (lazy callback vs iterative re-solve loop).

This module builds no optimization models; the formulation packages do.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ortools.math_opt.python import mathopt


class BackendError(ValueError):
    """Raised for an unknown or unavailable backend name."""


class ApiFamily(Enum):
    CP_SAT = "cp_sat"
    MATH_OPT = "math_opt"


@dataclass(frozen=True)
class Backend:
    name: str
    family: ApiFamily
    solver_type: mathopt.SolverType | None  # MathOpt SolverType; None for CP-SAT
    continuous_flow: bool
    supports_lazy: bool
    # HiGHS rejects the MathOpt threads= param (must be set via globals); others accept it.
    supports_threads_param: bool = True


# Worker count for all backends — held equal so solve-time comparisons stay fair.
# 6 (not 1): CP-SAT's speed relies on its multi-worker search portfolio.
NUM_THREADS: int = 6

# Static registry. supports_lazy: CP-SAT and HiGHS always use the iterative
# cut loop (no lazy-callback support). SCIP and Gurobi support MIP_SOLUTION
# lazy-constraint callbacks via ortools.math_opt.python.callback — confirmed
# in Stage 3a Task 5. HiGHS and CP-SAT keep supports_lazy=False.
# Confirmed SolverType names from installed ortools: GSCIP, HIGHS, GUROBI.
_REGISTRY: dict[str, Backend] = {
    "cp-sat": Backend("cp-sat", ApiFamily.CP_SAT, None, False, False),
    "cp-sat-m": Backend("cp-sat-m", ApiFamily.MATH_OPT, mathopt.SolverType.CP_SAT, False, False),
    "scip": Backend("scip", ApiFamily.MATH_OPT, mathopt.SolverType.GSCIP, True, False),
    # HiGHS rejects the MathOpt threads param; supports_threads_param=False skips it.
    "highs": Backend(
        "highs", ApiFamily.MATH_OPT, mathopt.SolverType.HIGHS, True, False, False,
    ),
    "gurobi": Backend("gurobi", ApiFamily.MATH_OPT, mathopt.SolverType.GUROBI, True, True),
}


def resolve(name: str) -> Backend:
    """Look up a backend by ``--backend`` name."""
    try:
        backend = _REGISTRY[name]
    except KeyError:
        raise BackendError(f"unknown backend {name!r}; choose from {sorted(_REGISTRY)}") from None
    if backend.name == "gurobi" and not _gurobi_available():
        raise BackendError("backend 'gurobi' is unavailable: no usable license found")
    return backend


def available_backends() -> list[Backend]:
    """Backends usable at runtime. CP-SAT and bundled SCIP/HiGHS are always
    present; Gurobi only if a license resolves."""
    available: list[Backend] = []
    for backend in _REGISTRY.values():
        if backend.name == "gurobi" and not _gurobi_available():
            continue
        available.append(backend)
    return available


def _gurobi_available() -> bool:
    """Probe for a usable Gurobi license without making it a hard dependency."""
    model = mathopt.Model(name="probe")
    try:
        mathopt.solve(model, mathopt.SolverType.GUROBI)
    except Exception:  # noqa: BLE001 -- any failure means Gurobi is unusable here
        return False
    return True
