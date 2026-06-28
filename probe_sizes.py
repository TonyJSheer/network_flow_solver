"""Quick backend probe — run with: uv run python probe_sizes.py

Trivially-easy instances only, so every backend (including native cp-sat) solves
to optimality and the four columns are directly comparable. Bump SIZES / REGIMES
to explore harder instances; cp-sat will hit the time limit on wide/large ones.
"""

from src.backends import resolve
from src.direct_mip import solve_direct_mip
from src.generator import Regime, generate_instance

TIME_LIMIT = 10.0
BACKENDS = ["cp-sat-m", "cp-sat"]  # "highs", "scip", 
# Tight 8 is slow enough to show a difference in solvers
# First few med should be fine too, up to 5
SIZES = [8]  # [8], [1,2,3,4,5], 
REGIMES = [Regime.TIGHT] # [TIGHT, MEDIUM, WIDE]

header = (
    f"{'instance':<34} {'backend':<10} {'status':<10}"
    f" {'obj':>14} {'gap':>8} {'nodes':>10} {'time':>7}"
)
print(header)
print("-" * len(header))

for size_idx in SIZES:
    for regime in REGIMES:
        inst = generate_instance(size_idx=size_idx, list_idx=0, regime=regime, seed=42)
        label = (
            f"size={size_idx} {regime.name.lower()}"
            f" ({len(inst.arcs)}a {len(inst.jobs)}j T={inst.horizon})"
        )
        for backend_name in BACKENDS:
            backend = resolve(backend_name)
            result = solve_direct_mip(inst, backend, time_limit_s=TIME_LIMIT)
            gap_str = f"{result.gap:.4f}" if result.gap is not None else "n/a"
            nodes_str = str(result.node_count) if result.node_count is not None else "n/a"
            obj_str = f"{result.objective:.1f}" if result.objective is not None else "n/a"
            print(
                f"{label:<34} {backend_name:<10} {result.status.value:<10}"
                f" {obj_str:>14} {gap_str:>8}"
                f" {nodes_str:>10} {result.wall_time_s:>6.1f}s"
            )
        print()
