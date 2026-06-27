import networkx as nx
import numpy as np

from src.generator import (
    CAP_HI,
    CAP_LO,
    NUM_SIZES,
    SIZE_SCHEDULE,
    Regime,
    _build_network,
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


def test_build_network_is_a_legal_st_dag() -> None:
    for size_idx in range(1, NUM_SIZES + 1):
        rng = _derive_rng(0, size_idx, 0, Regime.MEDIUM)
        nodes, arcs = _build_network(size_idx, rng)
        assert nodes[0] == "s" and nodes[-1] == "t"
        # no parallel arcs under (u, v) identity
        keys = [(a.u, a.v) for a in arcs]
        assert len(set(keys)) == len(keys)
        # every endpoint is a declared node; capacities in range
        node_set = set(nodes)
        for a in arcs:
            assert a.u in node_set and a.v in node_set
            assert CAP_LO <= a.capacity <= CAP_HI
        # s reaches t
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(keys)
        assert nx.has_path(g, "s", "t")


def test_network_arc_count_is_nondecreasing() -> None:
    counts = []
    for size_idx in range(1, NUM_SIZES + 1):
        _, arcs = _build_network(size_idx, _derive_rng(0, size_idx, 0, Regime.MEDIUM))
        counts.append(len(arcs))
    assert counts == sorted(counts)
