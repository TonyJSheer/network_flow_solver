import numpy as np

from src.generator import (
    NUM_SIZES,
    SIZE_SCHEDULE,
    Regime,
    _derive_rng,
)


def test_regime_width_ranges() -> None:
    assert Regime.TIGHT.width_range == (1, 10)
    assert Regime.MEDIUM.width_range == (1, 35)
    assert Regime.WIDE.width_range == (25, 35)


def test_size_schedule_has_eight_monotone_rows() -> None:
    assert len(SIZE_SCHEDULE) == NUM_SIZES
    # base arc count of a layered full-bipartite DAG: 2W + (L-1)*W*W
    counts = [2 * w + (layers - 1) * w * w for layers, w in SIZE_SCHEDULE]
    assert counts == sorted(counts)
    assert len(set(counts)) == len(counts)  # strictly increasing


def test_derive_rng_is_repeatable_and_index_sensitive() -> None:
    a = _derive_rng(0, 1, 0, Regime.WIDE).integers(0, 1_000_000, size=5)
    b = _derive_rng(0, 1, 0, Regime.WIDE).integers(0, 1_000_000, size=5)
    c = _derive_rng(0, 1, 1, Regime.WIDE).integers(0, 1_000_000, size=5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
