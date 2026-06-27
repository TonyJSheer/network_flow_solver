"""Random instance generator for the maintenance-scheduling problem.

Networks are layered DAGs (s -> L layers of width W -> t). Each arc carries
5-15 maintenance jobs laid out sequentially so a non-overlapping schedule always
exists. Window regimes are the *number of legal start times* per job (paper
1603.02378v2 sec 4.1): width = deadline - release + 1 (our `deadline` is the
latest start). All randomness derives from (seed, size_idx, list_idx, regime).
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from src.instance import Arc, Job


class Regime(Enum):
    """Window regime = number of legal start times per job (lo, hi)."""

    TIGHT = (1, 10)
    MEDIUM = (1, 35)
    WIDE = (25, 35)

    @property
    def width_range(self) -> tuple[int, int]:
        lo, hi = self.value
        return lo, hi


# (num_layers, layer_width) per size index 1..8.
# Base arc count 2W+(L-1)W^2 = 4,6,8,10,12,15,24,35 (strictly increasing).
SIZE_SCHEDULE: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (2, 3),
    (2, 4),
    (2, 5),
)

CAP_LO, CAP_HI = 10, 100
MIN_JOBS_PER_ARC, MAX_JOBS_PER_ARC = 5, 15
DUR_LO, DUR_HI = 10, 30
NUM_SIZES = 8
NUM_LISTS = 10


def _derive_rng(
    seed: int, size_idx: int, list_idx: int, regime: Regime
) -> np.random.Generator:
    """Independent, reproducible RNG stream keyed by the instance coordinates."""
    regime_code = list(Regime).index(regime)
    return np.random.default_rng([seed, size_idx, list_idx, regime_code])


def _build_network(
    size_idx: int, rng: np.random.Generator
) -> tuple[tuple[str, ...], tuple[Arc, ...]]:
    """Layered DAG: s -> L layers of width W -> t, full bipartite between layers."""
    num_layers, width = SIZE_SCHEDULE[size_idx - 1]
    layers: list[list[str]] = [
        [f"L{i}_{w}" for w in range(width)] for i in range(num_layers)
    ]
    nodes: list[str] = ["s"]
    for layer in layers:
        nodes.extend(layer)
    nodes.append("t")

    def cap() -> int:
        return int(rng.integers(CAP_LO, CAP_HI + 1))

    arcs: list[Arc] = []
    for node in layers[0]:  # source into first layer
        arcs.append(Arc("s", node, cap()))
    for i in range(num_layers - 1):  # full bipartite between adjacent layers
        for u in layers[i]:
            for v in layers[i + 1]:
                arcs.append(Arc(u, v, cap()))
    for node in layers[-1]:  # last layer into sink
        arcs.append(Arc(node, "t", cap()))
    for i in range(num_layers - 2):  # a few skip arcs for irregular cuts
        arcs.append(Arc(layers[i][0], layers[i + 2][0], cap()))

    return tuple(nodes), tuple(arcs)


def _build_jobs_for_arc(
    arc: Arc, regime: Regime, rng: np.random.Generator
) -> tuple[list[Job], int]:
    """Lay jobs out sequentially so even all-latest starts never overlap.

    Job k gets `release_k`; `deadline_k = release_k + width_k - 1` so it has
    exactly `width_k` legal start positions. The next job's release sits past
    the current job's latest completion (`deadline + duration - 1`), guaranteeing
    the no-overlap-per-arc assumption is satisfiable for any choice of starts.
    """
    lo, hi = regime.width_range
    n_jobs = int(rng.integers(MIN_JOBS_PER_ARC, MAX_JOBS_PER_ARC + 1))
    jobs: list[Job] = []
    cursor = 1  # earliest release (periods are 1-indexed)
    for k in range(n_jobs):
        duration = int(rng.integers(DUR_LO, DUR_HI + 1))
        width = int(rng.integers(lo, hi + 1))  # number of legal start times
        release = cursor
        deadline = release + width - 1  # latest start
        jobs.append(
            Job(
                id=f"{arc.u}->{arc.v}#{k}",
                arc=(arc.u, arc.v),
                duration=duration,
                release=release,
                deadline=deadline,
            )
        )
        latest_completion = deadline + duration - 1
        gap = int(rng.integers(0, 4))  # small slack between consecutive jobs
        cursor = latest_completion + 1 + gap
    last = jobs[-1]
    arc_horizon = last.deadline + last.duration - 1
    return jobs, arc_horizon
