# Instance Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/generator.py` — a reproducible random-instance generator for the maintenance-scheduling problem (Pearce-Forbes regime), emitting JSON via the existing `Instance` model, plus a hand-checkable toy instance.

**Architecture:** Layered-DAG networks on a monotone size schedule; per-arc sequential job layout that guarantees a feasible non-overlapping schedule exists; window regimes expressed as *number of legal start times*. All randomness flows through a numpy `Generator` derived from `(seed, size_idx, list_idx, regime)`. Reuses `src/instance.py` (`Arc`, `Job`, `Instance`, `load`, `save`).

**Tech Stack:** Python 3.12+, numpy (RNG), the existing `src.instance` module, pytest. Run everything through `uv`.

## Global Constraints

- Python 3.12+; run Python only via `uv` (never bare `python3`).
- Type annotations on every function; `mypy --strict` clean; `ruff` clean (line-length 100, rules E/F/I/UP/B/ANN).
- Reuse `src.instance` types verbatim; do not modify the `Instance` data model.
- Confirmed paper params (do not change): 8 sizes × 10 job-lists; jobs/arc ∈ `[5,15]`; duration ∈ `[10,30]`; window-width regimes TIGHT `(1,10)`, MEDIUM `(1,35)`, WIDE `(25,35)` = number of legal start times where width = `deadline − release + 1`.
- Our-choice demo params: capacities integer `U[10,100]`; per-instance horizon = max over arcs of that arc's last-job latest completion.
- No two jobs on the same arc may overlap — guaranteed by construction (this is also a Stage 2/3 formulation dependency, out of scope here).

---

## File Structure

- Create: `src/generator.py` — the whole generator (enum, size table, network builder, per-arc job builder, assembly, CLI). One module, focused responsibility.
- Create: `tests/test_generator.py` — all generator tests.
- Modify: `.gitignore` — add `instances/`.

---

### Task 1: Regime enum, size schedule, RNG derivation

**Files:**
- Create: `src/generator.py`
- Test: `tests/test_generator.py`

**Interfaces:**
- Consumes: nothing (entry module).
- Produces:
  - `class Regime(Enum)` with members `TIGHT=(1,10)`, `MEDIUM=(1,35)`, `WIDE=(25,35)` and property `width_range -> tuple[int,int]`.
  - `SIZE_SCHEDULE: tuple[tuple[int,int], ...]` — 8 rows of `(num_layers, layer_width)`.
  - Constants `CAP_LO=10`, `CAP_HI=100`, `MIN_JOBS_PER_ARC=5`, `MAX_JOBS_PER_ARC=15`, `DUR_LO=10`, `DUR_HI=30`, `NUM_SIZES=8`, `NUM_LISTS=10`.
  - `_derive_rng(seed: int, size_idx: int, list_idx: int, regime: Regime) -> np.random.Generator`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.generator'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/generator.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generator.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/generator.py tests/test_generator.py
uv run mypy src/generator.py
git add src/generator.py tests/test_generator.py
git commit -m "feat: generator regime enum, size schedule, rng derivation"
```

---

### Task 2: Layered-DAG network builder

**Files:**
- Modify: `src/generator.py`
- Test: `tests/test_generator.py`

**Interfaces:**
- Consumes: `SIZE_SCHEDULE`, `CAP_LO`, `CAP_HI`, `_derive_rng` from Task 1; `Arc` from `src.instance`.
- Produces: `_build_network(size_idx: int, rng: np.random.Generator) -> tuple[tuple[str, ...], tuple[Arc, ...]]` returning `(nodes, arcs)` with `s` first and `t` last.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_generator.py
import networkx as nx

from src.generator import _build_network


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generator.py -k network -q`
Expected: FAIL — `ImportError: cannot import name '_build_network'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/generator.py imports
from src.instance import Arc

# add at end of src/generator.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generator.py -k network -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/generator.py tests/test_generator.py
uv run mypy src/generator.py
git add src/generator.py tests/test_generator.py
git commit -m "feat: layered-DAG network builder"
```

---

### Task 3: Per-arc job builder (non-overlap by construction)

**Files:**
- Modify: `src/generator.py`
- Test: `tests/test_generator.py`

**Interfaces:**
- Consumes: `MIN_JOBS_PER_ARC`, `MAX_JOBS_PER_ARC`, `DUR_LO`, `DUR_HI`, `Regime`, `_derive_rng` from Task 1; `Arc`, `Job` from `src.instance`.
- Produces: `_build_jobs_for_arc(arc: Arc, regime: Regime, rng: np.random.Generator) -> tuple[list[Job], int]` returning the arc's jobs (in start order) and the arc's required horizon (last job's latest completion).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_generator.py
from src.generator import (
    DUR_HI,
    DUR_LO,
    MAX_JOBS_PER_ARC,
    MIN_JOBS_PER_ARC,
    _build_jobs_for_arc,
)
from src.instance import Arc


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
    for prev, nxt in zip(jobs, jobs[1:]):
        prev_latest_completion = prev.deadline + prev.duration - 1
        assert prev_latest_completion < nxt.release
    last = jobs[-1]
    assert arc_horizon == last.deadline + last.duration - 1
    assert jobs[0].release >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generator.py -k jobs -q`
Expected: FAIL — `ImportError: cannot import name '_build_jobs_for_arc'`.

- [ ] **Step 3: Write minimal implementation**

```python
# extend the src.instance import in src/generator.py
from src.instance import Arc, Job

# add at end of src/generator.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generator.py -k jobs -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/generator.py tests/test_generator.py
uv run mypy src/generator.py
git add src/generator.py tests/test_generator.py
git commit -m "feat: per-arc job builder with non-overlap by construction"
```

---

### Task 4: Instance assembly + suite (legality centerpiece)

**Files:**
- Modify: `src/generator.py`
- Test: `tests/test_generator.py`

**Interfaces:**
- Consumes: `_build_network`, `_build_jobs_for_arc`, `_derive_rng`, `NUM_SIZES`, `NUM_LISTS`, `Regime` from earlier tasks; `Instance`, `load` from `src.instance`.
- Produces:
  - `generate_instance(size_idx: int, list_idx: int, regime: Regime, seed: int) -> Instance`
  - `generate_suite(seed: int) -> Iterator[Instance]` (yields all `NUM_SIZES * NUM_LISTS * 3` instances).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_generator.py
from src.generator import generate_instance, generate_suite
from src.instance import Instance


def _assert_legal(inst: Instance, tmp_path) -> None:
    # load() runs the full structural validator and raises on any illegality.
    out = tmp_path / f"{inst.name}.json"
    inst.save(out)
    from src.instance import load

    reloaded = load(out)
    assert reloaded == inst


def test_every_generated_instance_is_legal(tmp_path) -> None:
    for size_idx in range(1, NUM_SIZES + 1):
        for regime in Regime:
            inst = generate_instance(size_idx, 0, regime, seed=0)
            _assert_legal(inst, tmp_path)
            assert inst.source == "s" and inst.sink == "t"
            assert inst.known_optimum is None
            assert inst.jobs  # non-empty
            assert inst.horizon == max(j.deadline + j.duration - 1 for j in inst.jobs)


def test_quick_small_sizes_are_legal_across_lists(tmp_path) -> None:
    # The --quick demo uses the small sizes; exercise them broadly.
    for size_idx in (1, 2, 3):
        for list_idx in range(NUM_LISTS):
            for regime in Regime:
                _assert_legal(generate_instance(size_idx, list_idx, regime, 0), tmp_path)


def test_suite_yields_full_sweep() -> None:
    names = [inst.name for inst in generate_suite(seed=0)]
    assert len(names) == NUM_SIZES * NUM_LISTS * 3
    assert len(set(names)) == len(names)  # unique names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generator.py -k "generated or suite or quick" -q`
Expected: FAIL — `ImportError: cannot import name 'generate_instance'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/generator.py imports
from collections.abc import Iterator

from src.instance import Arc, Instance, Job  # extend existing instance import

# add at end of src/generator.py
def generate_instance(
    size_idx: int, list_idx: int, regime: Regime, seed: int
) -> Instance:
    """Assemble one instance: layered network + per-arc jobs + fitted horizon."""
    rng = _derive_rng(seed, size_idx, list_idx, regime)
    nodes, arcs = _build_network(size_idx, rng)
    jobs: list[Job] = []
    horizon = 0
    for arc in arcs:
        arc_jobs, arc_horizon = _build_jobs_for_arc(arc, regime, rng)
        jobs.extend(arc_jobs)
        horizon = max(horizon, arc_horizon)
    return Instance(
        name=f"net{size_idx}-list{list_idx}-{regime.name.lower()}",
        horizon=horizon,
        source="s",
        sink="t",
        nodes=nodes,
        arcs=arcs,
        jobs=tuple(jobs),
        seed=seed,
        max_jobs_per_period=None,
        known_optimum=None,
    )


def generate_suite(seed: int) -> Iterator[Instance]:
    """Yield the full 8 sizes x 10 lists x 3 regimes sweep."""
    for size_idx in range(1, NUM_SIZES + 1):
        for list_idx in range(NUM_LISTS):
            for regime in Regime:
                yield generate_instance(size_idx, list_idx, regime, seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generator.py -k "generated or suite or quick" -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/generator.py tests/test_generator.py
uv run mypy src/generator.py
git add src/generator.py tests/test_generator.py
git commit -m "feat: instance assembly and full-suite generator"
```

---

### Task 5: Toy instance

**Files:**
- Modify: `src/generator.py`
- Test: `tests/test_generator.py`

**Interfaces:**
- Consumes: `Arc`, `Job`, `Instance` from `src.instance`.
- Produces: `toy_instance() -> Instance` returning the hand-checkable instance matching `tests/fixtures/toy.json` (`known_optimum=8`).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_generator.py
from pathlib import Path

from src.generator import toy_instance
from src.instance import load as load_instance

FIXTURE = Path(__file__).parent / "fixtures" / "toy.json"


def test_toy_instance_matches_committed_fixture() -> None:
    inst = toy_instance()
    assert inst == load_instance(FIXTURE)
    assert inst.known_optimum == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generator.py -k toy -q`
Expected: FAIL — `ImportError: cannot import name 'toy_instance'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add at end of src/generator.py
def toy_instance() -> Instance:
    """Tiny hand-checkable instance (known optimum 8); matches fixtures/toy.json.

    s->a cap 3, a->t cap 2; one 2-period job on a->t with start window [1,5],
    horizon 6. Max flow is min-cut 2 per period x 6 periods minus the 2 periods
    the a->t arc is out for maintenance => 2*6 - 2*2 = 8.
    """
    return Instance(
        name="toy",
        horizon=6,
        source="s",
        sink="t",
        nodes=("s", "a", "t"),
        arcs=(Arc("s", "a", 3), Arc("a", "t", 2)),
        jobs=(Job("j0", ("a", "t"), 2, 1, 5),),
        seed=42,
        max_jobs_per_period=None,
        known_optimum=8,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generator.py -k toy -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/generator.py tests/test_generator.py
uv run mypy src/generator.py
git add src/generator.py tests/test_generator.py
git commit -m "feat: toy instance constructor matching fixture"
```

---

### Task 6: CLI entry point + gitignore

**Files:**
- Modify: `src/generator.py`
- Modify: `.gitignore`
- Test: `tests/test_generator.py`

**Interfaces:**
- Consumes: `generate_instance`, `Regime`, `NUM_SIZES`, `NUM_LISTS` from earlier tasks.
- Produces: `main(argv: list[str] | None = None) -> None` plus a `__main__` guard; writes `<name>.json` files into `--out`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_generator.py
from src.generator import main


def test_cli_writes_filtered_instances(tmp_path) -> None:
    out = tmp_path / "instances"
    main(["--seed", "0", "--out", str(out), "--size", "1", "--regime", "wide"])
    files = sorted(p.name for p in out.glob("*.json"))
    assert files == [f"net1-list{i}-wide.json" for i in range(NUM_LISTS)]
    # written files are valid instances
    for p in out.glob("*.json"):
        load_instance(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generator.py -k cli -q`
Expected: FAIL — `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/generator.py imports
import argparse
from pathlib import Path


def _parse_regime(text: str) -> Regime:
    return Regime[text.upper()]


def main(argv: list[str] | None = None) -> None:
    """CLI: emit instance JSON files into --out (default ./instances)."""
    parser = argparse.ArgumentParser(description="Generate scheduling instances.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("instances"))
    parser.add_argument("--regime", type=_parse_regime, default=None)
    parser.add_argument("--size", type=int, default=None)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    sizes = [args.size] if args.size is not None else range(1, NUM_SIZES + 1)
    regimes = [args.regime] if args.regime is not None else list(Regime)
    for size_idx in sizes:
        for list_idx in range(NUM_LISTS):
            for regime in regimes:
                inst = generate_instance(size_idx, list_idx, regime, args.seed)
                inst.save(args.out / f"{inst.name}.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generator.py -k cli -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Add gitignore entry**

Add this line to `.gitignore` under the "Generated experiment artefacts" section:

```
instances/
```

- [ ] **Step 6: Full suite, lint, type-check, commit**

```bash
uv run pytest tests/test_generator.py -q
uv run ruff check . && uv run ruff format --check .
uv run mypy .
git add src/generator.py tests/test_generator.py .gitignore
git commit -m "feat: generator CLI entry point; gitignore instances/"
```

Expected: all generator tests pass; ruff and mypy clean.

---

## Self-Review

- **Spec coverage:** Regime-as-window-width (Task 1); layered DAG + capacities (Task 2); jobs/arc, durations, non-overlap construction, fitted horizon (Tasks 3–4); seeding/reproducibility (Task 1 RNG + Task 4 names carry indices); toy matches fixture (Task 5); CLI + `instances/` gitignore (Task 6); legality-focused testing with `--quick` small sizes emphasized (Task 4). Forward dependency (no-overlap as Stage 2/3 constraint) recorded in spec, intentionally out of this plan's scope.
- **Placeholder scan:** none — every step ships real code/commands.
- **Type consistency:** `_build_network -> (nodes, arcs)`, `_build_jobs_for_arc -> (jobs, arc_horizon)`, `generate_instance`/`generate_suite`/`toy_instance`/`main` signatures are used identically across tasks; `Regime.width_range` referenced consistently.
