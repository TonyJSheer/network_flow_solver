"""Quick method+backend probe — run with: uv run python probe_sizes.py

Solves each instance with BOTH the direct MIP and disaggregated Benders, across
every backend, and checks they agree on the optimum. The headline correctness
claim for Stage 3 is that Benders returns the SAME optimal flow as the direct
MIP and that the value is stable across backends (cp-sat / scip / highs /
cp-sat-m); this probe makes that visible at a glance.

Trivially-easy instances by default, so every backend (including native cp-sat)
solves to optimality and the columns are directly comparable. Bump SIZES /
REGIMES to explore harder instances; cp-sat will hit the time limit on
wide/large ones, at which point its objective may be an incumbent (FEASIBLE),
not a proven optimum, and the agreement check skips it.
"""

from __future__ import annotations

import logging
import math

from src.backends import resolve
from src.benders import solve_benders
from src.direct_mip import solve_direct_mip
from src.generator import Regime, generate_instance
from src.result import Result, SolveStatus

TIME_LIMIT = 10.0
BACKENDS = ["cp-sat-m", "cp-sat"]
# Tight 8 is slow enough to show a difference in solvers.
# First few med should be fine too, up to 5.
SIZES = [1]  # [8], [1,2,3,4,5]
REGIMES = [Regime.MEDIUM]  # [TIGHT, MEDIUM, WIDE]

# Surface the Benders master-vs-subproblem timing breakdown emitted by src.benders
# (one indented line per Benders solve, printed just before that solve's table row).
logging.basicConfig(level=logging.INFO, format="    %(message)s")

# Methods to compare. Each is (label, solver-callable).
METHODS = [
    ("direct", solve_direct_mip),
    ("benders", solve_benders),
]

header = (
    f"{'instance':<30} {'method':<8} {'backend':<10} {'status':<10}"
    f" {'obj':>10} {'gap':>8} {'detail':>16} {'time':>7}"
)
print(header)
print("-" * len(header))


def _detail(result: Result) -> str:
    """Per-method progress metric: search nodes (direct) or iters/cuts (Benders)."""
    if result.method == "benders":
        return f"it={result.iteration_count} cuts={result.cut_count}"
    return f"nodes={result.node_count}" if result.node_count is not None else "nodes=n/a"


for size_idx in SIZES:
    for regime in REGIMES:
        inst = generate_instance(size_idx=size_idx, list_idx=0, regime=regime, seed=42)
        label = (
            f"size={size_idx} {regime.name.lower()}"
            f" ({len(inst.arcs)}a {len(inst.jobs)}j T={inst.horizon})"
        )

        # Collect proven optima (method, backend) -> objective for the agreement check.
        optima: dict[tuple[str, str], float] = {}
        for method_name, solve in METHODS:
            for backend_name in BACKENDS:
                backend = resolve(backend_name)
                result = solve(inst, backend, time_limit_s=TIME_LIMIT)
                if result.status is SolveStatus.OPTIMAL and result.objective is not None:
                    optima[(method_name, backend_name)] = result.objective
                gap_str = f"{result.gap:.4f}" if result.gap is not None else "n/a"
                obj_str = f"{result.objective:.1f}" if result.objective is not None else "n/a"
                print(
                    f"{label:<30} {method_name:<8} {backend_name:<10} {result.status.value:<10}"
                    f" {obj_str:>10} {gap_str:>8} {_detail(result):>16}"
                    f" {result.wall_time_s:>6.1f}s"
                )
            print()

        # Agreement verdict: every proven optimum on this instance must match.
        values = list(optima.values())
        if not values:
            verdict = "?? no proven optimum (all hit time limit?)"
        elif all(math.isclose(v, values[0], rel_tol=1e-6, abs_tol=1e-6) for v in values):
            verdict = f"OK  all {len(values)} proven optima agree (obj={values[0]:.1f})"
        else:
            spread = ", ".join(f"{m}/{b}={v:.1f}" for (m, b), v in sorted(optima.items()))
            verdict = f"!!  MISMATCH: {spread}"
        print(f"  -> {verdict}\n")
