"""Shared result record returned by every formulation/backend.

A single rich-superset dataclass: each method+backend fills the fields it can
produce and leaves the rest ``None``. The benchmark harness flattens this to a
CSV row; tests compare the (objective, status, schedule) triple for agreement
across methods and backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SolveStatus(Enum):
    """Normalized solve outcome — backend-native codes map onto these four.

    TIME_LIMIT is intentionally absent: the time limit is a solve *input*; the
    status reports only what came back (an incumbent -> FEASIBLE, nothing ->
    UNKNOWN).
    """

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"  # incumbent found, not proven optimal
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"  # no incumbent (incl. time limit with nothing found)


@dataclass(frozen=True)
class Result:
    """Outcome of one solve. Optional fields are ``None`` when not applicable."""

    method: str  # "direct_mip" | "benders"
    backend: str  # "cp-sat" | "scip" | "highs" | "gurobi"
    status: SolveStatus
    objective: float | None  # None if no incumbent
    wall_time_s: float
    gap: float | None = None
    schedule: dict[str, int] | None = None  # job id -> start period
    node_count: int | None = None
    iteration_count: int | None = None  # Benders only
    cut_count: int | None = None  # Benders only
