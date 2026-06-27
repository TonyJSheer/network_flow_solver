from pathlib import Path

import networkx as nx
import numpy as np

from src.generator import (
    CAP_HI,
    CAP_LO,
    DUR_HI,
    DUR_LO,
    MAX_JOBS_PER_ARC,
    MIN_JOBS_PER_ARC,
    NUM_LISTS,
    NUM_SIZES,
    SIZE_SCHEDULE,
    Regime,
    _build_jobs_for_arc,
    _build_network,
    _derive_rng,
    generate_instance,
    generate_suite,
    main,
)
from src.instance import Arc, Instance
from src.instance import load as load_instance


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


def test_jobs_respect_paper_parameter_ranges() -> None:
    arc = Arc("u", "v", 50)
    for regime in Regime:
        lo, hi = regime.width_range
        rng = _derive_rng(0, 5, 2, regime)
        jobs, _ = _build_jobs_for_arc(arc, regime, rng)
        assert MIN_JOBS_PER_ARC <= len(jobs) <= MAX_JOBS_PER_ARC
        for j in jobs:
            assert j.arc == ("u", "v")
            assert DUR_LO <= j.duration <= DUR_HI
            width = j.deadline - j.release + 1  # number of legal start times
            assert lo <= width <= hi


def test_jobs_admit_a_nonoverlapping_schedule() -> None:
    # Even the worst case (every job at its LATEST start) must not overlap.
    arc = Arc("u", "v", 50)
    jobs, arc_horizon = _build_jobs_for_arc(arc, Regime.WIDE, _derive_rng(0, 8, 0, Regime.WIDE))
    for prev, nxt in zip(jobs, jobs[1:], strict=False):
        prev_latest_completion = prev.deadline + prev.duration - 1
        assert prev_latest_completion < nxt.release
    last = jobs[-1]
    assert arc_horizon == last.deadline + last.duration - 1
    assert jobs[0].release >= 1


def _assert_legal(inst: Instance, tmp_path: Path) -> None:
    # load() runs the full structural validator and raises on any illegality.
    out = tmp_path / f"{inst.name}.json"
    inst.save(out)
    reloaded = load_instance(out)
    assert reloaded == inst


def test_every_generated_instance_is_legal(tmp_path: Path) -> None:
    for size_idx in range(1, NUM_SIZES + 1):
        for regime in Regime:
            inst = generate_instance(size_idx, 0, regime, seed=0)
            _assert_legal(inst, tmp_path)
            assert inst.source == "s" and inst.sink == "t"
            assert inst.known_optimum is None
            assert inst.jobs  # non-empty
            assert inst.horizon == max(j.deadline + j.duration - 1 for j in inst.jobs)


def test_quick_small_sizes_are_legal_across_lists(tmp_path: Path) -> None:
    # The --quick demo uses the small sizes; exercise them broadly.
    for size_idx in (1, 2, 3):
        for list_idx in range(NUM_LISTS):
            for regime in Regime:
                _assert_legal(generate_instance(size_idx, list_idx, regime, 0), tmp_path)


def test_suite_yields_full_sweep() -> None:
    names = [inst.name for inst in generate_suite(seed=0)]
    assert len(names) == NUM_SIZES * NUM_LISTS * 3
    assert len(set(names)) == len(names)  # unique names


def test_toy_instance_matches_committed_fixture() -> None:
    from src.generator import toy_instance

    fixture = Path(__file__).parent / "fixtures" / "toy.json"
    inst = toy_instance()
    assert inst == load_instance(fixture)
    assert inst.known_optimum == 8


def test_cli_writes_filtered_instances(tmp_path: Path) -> None:
    out = tmp_path / "instances"
    main(["--seed", "0", "--out", str(out), "--size", "1", "--regime", "wide"])
    files = sorted(p.name for p in out.glob("*.json"))
    assert files == [f"net1-list{i}-wide.json" for i in range(NUM_LISTS)]
    # written files are valid instances
    for p in out.glob("*.json"):
        load_instance(p)
