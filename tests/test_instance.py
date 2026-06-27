# tests/test_instance.py
from pathlib import Path

import pytest

from src.instance import Arc, InstanceError, Job, load

FIXTURE = Path(__file__).parent / "fixtures" / "toy.json"


def test_load_toy_fixture_parses_all_fields() -> None:
    inst = load(FIXTURE)
    assert inst.name == "toy"
    assert inst.horizon == 6
    assert inst.source == "s"
    assert inst.sink == "t"
    assert inst.nodes == ("s", "a", "t")
    assert inst.arcs == (Arc("s", "a", 3), Arc("a", "t", 2))
    assert inst.jobs == (Job("j0", ("a", "t"), 2, 1, 5),)
    assert inst.seed == 42
    assert inst.max_jobs_per_period is None
    assert inst.known_optimum == 8


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    inst = load(FIXTURE)
    out = tmp_path / "rt.json"
    inst.save(out)
    assert load(out) == inst


def test_to_digraph_has_capacitated_edges() -> None:
    g = load(FIXTURE).to_digraph()
    assert set(g.edges()) == {("s", "a"), ("a", "t")}
    assert g["s"]["a"]["capacity"] == 3
    assert g["a"]["t"]["capacity"] == 2


def test_parallel_arcs_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "parallel.json"
    bad.write_text(
        '{"name":"b","horizon":3,"source":"s","sink":"t","nodes":["s","t"],'
        '"arcs":[{"u":"s","v":"t","capacity":1},{"u":"s","v":"t","capacity":2}],'
        '"jobs":[]}'
    )
    with pytest.raises(InstanceError, match="parallel"):
        load(bad)


def test_job_referencing_unknown_arc_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "badjob.json"
    bad.write_text(
        '{"name":"b","horizon":3,"source":"s","sink":"t","nodes":["s","t"],'
        '"arcs":[{"u":"s","v":"t","capacity":1}],'
        '"jobs":[{"id":"j0","arc":["a","t"],"duration":1,"release":1,"deadline":1}]}'
    )
    with pytest.raises(InstanceError, match="arc"):
        load(bad)


def test_job_not_fitting_horizon_rejected(tmp_path: Path) -> None:
    # latest start 3 + duration 2 - 1 = period 4 > horizon 3 -> invalid
    bad = tmp_path / "overflow.json"
    bad.write_text(
        '{"name":"b","horizon":3,"source":"s","sink":"t","nodes":["s","t"],'
        '"arcs":[{"u":"s","v":"t","capacity":1}],'
        '"jobs":[{"id":"j0","arc":["s","t"],"duration":2,"release":1,"deadline":3}]}'
    )
    with pytest.raises(InstanceError, match="horizon"):
        load(bad)
