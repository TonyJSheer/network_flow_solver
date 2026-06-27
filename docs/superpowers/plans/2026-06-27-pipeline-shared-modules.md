# Pipeline Shared Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the three shared "spine" modules — `instance.py`, `result.py`, `backends.py` — that every later build stage depends on, plus the project scaffold to run them.

**Architecture:** Three independent, dependency-light modules behind stable contracts. `instance.py` is the data model (frozen dataclasses + JSON load/save + `nx.DiGraph` builder); `result.py` is the normalized result record (`Result` + `SolveStatus`); `backends.py` resolves a `--backend` name to an API family + two capability flags, building no models. No solver formulations in this plan — those are later stages that consume these three modules through a uniform `solve(instance, backend, *, time_limit_s) -> Result` signature.

**Tech Stack:** Python 3.11+, `uv`, `ortools` (CP-SAT + MathOpt), `networkx`, `numpy`, `matplotlib`, `pytest`, `ruff`, `mypy`.

**Source spec:** `docs/superpowers/specs/2026-06-27-pipeline-data-model-and-interfaces-design.md`

## Global Constraints

- Python 3.11+. Run Python only through `uv` — never bare `python3`.
- Dependencies limited to the declared set: `ortools`, `networkx`, `numpy`, `matplotlib`, `pytest` (+ `ruff`, `mypy` dev). No additions without justification.
- Gurobi is **optional** — never a hard dependency; never hardcode any solver license/keys.
- Type annotations on every function; `mypy --strict` clean; `ruff check` + `ruff format --check` clean.
- No bare `except:`; no `# type: ignore` without a justifying comment.
- All domain dataclasses are `@dataclass(frozen=True)`.
- Backend-specific code (OR-Tools solver selection) lives **only** in `backends.py`.
- Commit after each task with `<type>: <message>` on branch `docs/pipeline-data-model-design` (already checked out).

---

### Task 1: Project scaffold & tooling

**Files:**
- Create: `pyproject.toml`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_scaffold.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a working `uv` environment where `import ortools`, `import networkx` succeed and `uv run pytest` runs; `ruff` + `mypy` configured.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "network-flow-solver"
version = "0.1.0"
description = "Network maintenance scheduling: direct MIP vs disaggregated Benders (OR-Tools)."
requires-python = ">=3.11"
dependencies = [
    "ortools>=9.10",
    "networkx>=3.2",
    "numpy>=1.26",
    "matplotlib>=3.8",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.6",
    "mypy>=1.11",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "ANN"]

[tool.mypy]
python_version = "3.11"
strict = true

[[tool.mypy.overrides]]
module = ["networkx.*", "ortools.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create empty package markers**

`src/__init__.py` and `tests/__init__.py` are empty files (touch them).

- [ ] **Step 3: Write a scaffold smoke test**

```python
# tests/test_scaffold.py
def test_core_imports_available() -> None:
    import networkx  # noqa: F401
    import numpy  # noqa: F401
    from ortools.sat.python import cp_model  # noqa: F401

    assert cp_model.CpModel() is not None
```

- [ ] **Step 4: Sync and run the smoke test**

Run: `uv sync && uv run pytest tests/test_scaffold.py -v`
Expected: PASS (1 passed). If `uv sync` fails on `ortools`, stop and report — the rest of the plan depends on it.

- [ ] **Step 5: Verify lint + type tooling run**

Run: `uv run ruff check . && uv run mypy .`
Expected: no errors (an empty `src/` is clean).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/__init__.py tests/__init__.py tests/test_scaffold.py
git commit -m "chore: project scaffold, deps, ruff/mypy/pytest config"
```

---

### Task 2: Result record (`result.py`)

**Files:**
- Create: `src/result.py`
- Create: `tests/test_result.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class SolveStatus(Enum)` with members `OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `UNKNOWN`.
  - `@dataclass(frozen=True) class Result` with fields `method: str`, `backend: str`, `status: SolveStatus`, `objective: float | None`, `wall_time_s: float`, `gap: float | None = None`, `schedule: dict[str, int] | None = None`, `node_count: int | None = None`, `iteration_count: int | None = None`, `cut_count: int | None = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_result.py
import dataclasses

import pytest

from src.result import Result, SolveStatus


def test_result_minimal_construction_defaults_optionals_to_none() -> None:
    r = Result(
        method="direct_mip",
        backend="cp-sat",
        status=SolveStatus.OPTIMAL,
        objective=17.0,
        wall_time_s=0.42,
    )
    assert r.objective == 17.0
    assert r.gap is None
    assert r.schedule is None
    assert r.node_count is None
    assert r.iteration_count is None
    assert r.cut_count is None


def test_result_is_frozen() -> None:
    r = Result(
        method="benders",
        backend="scip",
        status=SolveStatus.FEASIBLE,
        objective=None,
        wall_time_s=1.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.objective = 5.0  # type: ignore[misc]  # asserting frozen behavior


def test_solve_status_has_exactly_four_members() -> None:
    assert {s.name for s in SolveStatus} == {
        "OPTIMAL",
        "FEASIBLE",
        "INFEASIBLE",
        "UNKNOWN",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_result.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.result'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/result.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_result.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Lint + type check**

Run: `uv run ruff check src/result.py tests/test_result.py && uv run mypy src/result.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/result.py tests/test_result.py
git commit -m "feat: shared Result record and normalized SolveStatus"
```

---

### Task 3: Instance data model (`instance.py`)

**Files:**
- Create: `src/instance.py`
- Create: `tests/fixtures/toy.json`
- Create: `tests/test_instance.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `@dataclass(frozen=True) class Arc` with `u: str`, `v: str`, `capacity: int`.
  - `@dataclass(frozen=True) class Job` with `id: str`, `arc: tuple[str, str]`, `duration: int`, `release: int`, `deadline: int`.
  - `@dataclass(frozen=True) class Instance` with `name: str`, `horizon: int`, `source: str`, `sink: str`, `nodes: tuple[str, ...]`, `arcs: tuple[Arc, ...]`, `jobs: tuple[Job, ...]`, `seed: int | None = None`, `max_jobs_per_period: int | None = None`, `known_optimum: int | None = None`.
  - Methods/functions: `Instance.to_digraph() -> networkx.DiGraph`, `Instance.save(path: pathlib.Path) -> None`, module-level `load(path: pathlib.Path) -> Instance`.
  - `class InstanceError(ValueError)` raised by `load`/validation.

- [ ] **Step 1: Write the toy fixture**

```json
{
  "name": "toy",
  "seed": 42,
  "horizon": 6,
  "source": "s",
  "sink": "t",
  "max_jobs_per_period": null,
  "nodes": ["s", "a", "t"],
  "arcs": [
    {"u": "s", "v": "a", "capacity": 3},
    {"u": "a", "v": "t", "capacity": 2}
  ],
  "jobs": [
    {"id": "j0", "arc": ["a", "t"], "duration": 2, "release": 1, "deadline": 5}
  ],
  "known_optimum": 8
}
```

(`known_optimum` here is a round-trip placeholder, not a solver-verified value — no solver exists yet. Note: latest start 5 + duration 2 - 1 = period 6 = horizon, so this job is schedulable, exercising the boundary validation.)

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_instance.py
from pathlib import Path

import pytest

from src.instance import Arc, Instance, InstanceError, Job, load

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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_instance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.instance'`.

- [ ] **Step 4: Write minimal implementation**

```python
# src/instance.py
"""Instance data model for the maintenance-scheduling problem.

The on-disk JSON is the single source of truth shared by the generator and both
solvers. Arc identity is the (u, v) tuple (per the design): this maps to a plain
``networkx.DiGraph`` and forbids parallel arcs (asserted on load).

Resolved spec assumptions (made explicit here):
- ``deadline`` is the latest *start* period, not completion. A job occupies
  periods [start, start + duration - 1] with release <= start <= deadline.
- A job must complete within the horizon: deadline + duration - 1 <= horizon.
- ``known_optimum`` is populated only for small, hand-checkable toy instances.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


class InstanceError(ValueError):
    """Raised when an instance fails structural validation."""


@dataclass(frozen=True)
class Arc:
    u: str
    v: str
    capacity: int


@dataclass(frozen=True)
class Job:
    id: str
    arc: tuple[str, str]
    duration: int
    release: int  # earliest start period
    deadline: int  # latest start period


@dataclass(frozen=True)
class Instance:
    name: str
    horizon: int
    source: str
    sink: str
    nodes: tuple[str, ...]
    arcs: tuple[Arc, ...]
    jobs: tuple[Job, ...]
    seed: int | None = None
    max_jobs_per_period: int | None = None  # K; None means unbounded
    known_optimum: int | None = None

    def to_digraph(self) -> nx.DiGraph:
        """Capacitated directed graph; each arc carries a ``capacity`` attr."""
        g: nx.DiGraph = nx.DiGraph()
        g.add_nodes_from(self.nodes)
        for arc in self.arcs:
            g.add_edge(arc.u, arc.v, capacity=arc.capacity)
        return g

    def save(self, path: Path) -> None:
        payload = {
            "name": self.name,
            "seed": self.seed,
            "horizon": self.horizon,
            "source": self.source,
            "sink": self.sink,
            "max_jobs_per_period": self.max_jobs_per_period,
            "nodes": list(self.nodes),
            "arcs": [{"u": a.u, "v": a.v, "capacity": a.capacity} for a in self.arcs],
            "jobs": [
                {
                    "id": j.id,
                    "arc": [j.arc[0], j.arc[1]],
                    "duration": j.duration,
                    "release": j.release,
                    "deadline": j.deadline,
                }
                for j in self.jobs
            ],
            "known_optimum": self.known_optimum,
        }
        path.write_text(json.dumps(payload, indent=2))


def load(path: Path) -> Instance:
    """Parse and validate an instance JSON file."""
    raw = json.loads(path.read_text())
    arcs = tuple(Arc(a["u"], a["v"], a["capacity"]) for a in raw["arcs"])
    jobs = tuple(
        Job(j["id"], (j["arc"][0], j["arc"][1]), j["duration"], j["release"], j["deadline"])
        for j in raw["jobs"]
    )
    inst = Instance(
        name=raw["name"],
        horizon=raw["horizon"],
        source=raw["source"],
        sink=raw["sink"],
        nodes=tuple(raw["nodes"]),
        arcs=arcs,
        jobs=jobs,
        seed=raw.get("seed"),
        max_jobs_per_period=raw.get("max_jobs_per_period"),
        known_optimum=raw.get("known_optimum"),
    )
    _validate(inst)
    return inst


def _validate(inst: Instance) -> None:
    node_set = set(inst.nodes)
    if inst.source not in node_set or inst.sink not in node_set:
        raise InstanceError("source/sink must be declared nodes")

    arc_keys = [(a.u, a.v) for a in inst.arcs]
    if len(set(arc_keys)) != len(arc_keys):
        raise InstanceError("parallel arcs are not allowed under (u, v) identity")

    arc_set = set(arc_keys)
    for a in inst.arcs:
        if a.u not in node_set or a.v not in node_set:
            raise InstanceError(f"arc {(a.u, a.v)} references an undeclared node")

    for j in inst.jobs:
        if j.arc not in arc_set:
            raise InstanceError(f"job {j.id} references unknown arc {j.arc}")
        if j.deadline < j.release:
            raise InstanceError(f"job {j.id} has deadline < release")
        if j.deadline + j.duration - 1 > inst.horizon:
            raise InstanceError(f"job {j.id} cannot complete within horizon")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_instance.py -v`
Expected: PASS (6 passed).

- [ ] **Step 6: Lint + type check**

Run: `uv run ruff check src/instance.py tests/test_instance.py && uv run mypy src/instance.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/instance.py tests/fixtures/toy.json tests/test_instance.py
git commit -m "feat: Instance data model with JSON load/save and validation"
```

---

### Task 4: Backend interface (`backends.py`)

**Files:**
- Create: `src/backends.py`
- Create: `tests/test_backends.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class ApiFamily(Enum)` with members `CP_SAT`, `MATH_OPT`.
  - `@dataclass(frozen=True) class Backend` with `name: str`, `family: ApiFamily`, `solver_type: object | None`, `continuous_flow: bool`, `supports_lazy: bool`.
  - `class BackendError(ValueError)`.
  - `resolve(name: str) -> Backend` — raises `BackendError` on unknown/unavailable names.
  - `available_backends() -> list[Backend]` — runtime-available backends (powers `--solver-check`).

- [ ] **Step 1: Verify the OR-Tools MathOpt API surface before coding**

The exact `SolverType` member names and the Gurobi-availability probe are backend specifics that must be confirmed against the installed `ortools`, not guessed. Use the `inspect-package` skill (or the commands below) to confirm:

Run: `uv run python -c "from ortools.math_opt.python import mathopt; print([s for s in dir(mathopt.SolverType) if not s.startswith('_')])"`

Record the exact member names for **SCIP** (commonly `GSCIP`), **HiGHS** (`HIGHS`), **Gurobi** (`GUROBI`), and **CP-SAT** (`CP_SAT`). If a name differs from the assumption in Step 3, use the confirmed name there.

Run: `uv run python -c "from ortools.math_opt.python import mathopt; print(type(mathopt.SolverType.GSCIP))"`
Expected: confirms `SolverType` is the import path used below.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_backends.py
import pytest

from src.backends import ApiFamily, Backend, BackendError, available_backends, resolve


def test_resolve_cpsat_is_integer_no_lazy() -> None:
    b = resolve("cp-sat")
    assert b.family is ApiFamily.CP_SAT
    assert b.continuous_flow is False
    assert b.supports_lazy is False  # CP-SAT uses the iterative cut loop
    assert b.solver_type is None


def test_resolve_mathopt_backends_are_continuous() -> None:
    for name in ("scip", "highs"):
        b = resolve(name)
        assert b.family is ApiFamily.MATH_OPT
        assert b.continuous_flow is True
        assert b.solver_type is not None


def test_resolve_unknown_name_raises() -> None:
    with pytest.raises(BackendError, match="unknown backend"):
        resolve("glpk")


def test_available_backends_always_includes_cpsat_scip_highs() -> None:
    names = {b.name for b in available_backends()}
    assert {"cp-sat", "scip", "highs"} <= names


def test_available_backends_returns_backend_objects() -> None:
    assert all(isinstance(b, Backend) for b in available_backends())
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_backends.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.backends'`.

- [ ] **Step 4: Write minimal implementation**

Use the `SolverType` member names confirmed in Step 1. The snippet assumes `GSCIP`/`HIGHS`/`GUROBI`/`CP_SAT`; correct them if Step 1 showed otherwise.

```python
# src/backends.py
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
    solver_type: object | None  # mathopt.SolverType for MATH_OPT; None for CP-SAT
    continuous_flow: bool
    supports_lazy: bool


# Static registry. supports_lazy defaults conservatively to False everywhere:
# CP-SAT has no lazy callbacks (always the iterative loop); whether MathOpt
# exposes SCIP lazy callbacks is verified when Stage 3 (Benders) is built.
_REGISTRY: dict[str, Backend] = {
    "cp-sat": Backend("cp-sat", ApiFamily.CP_SAT, None, False, False),
    "scip": Backend("scip", ApiFamily.MATH_OPT, mathopt.SolverType.GSCIP, True, False),
    "highs": Backend("highs", ApiFamily.MATH_OPT, mathopt.SolverType.HIGHS, True, False),
    "gurobi": Backend("gurobi", ApiFamily.MATH_OPT, mathopt.SolverType.GUROBI, True, False),
}


def resolve(name: str) -> Backend:
    """Look up a backend by ``--backend`` name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise BackendError(
            f"unknown backend {name!r}; choose from {sorted(_REGISTRY)}"
        ) from None


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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_backends.py -v`
Expected: PASS (5 passed). If `test_resolve_mathopt_backends_are_continuous` errors on a `SolverType` attribute, the name from Step 1 differs — fix the registry and re-run.

- [ ] **Step 6: Lint + type check**

Run: `uv run ruff check src/backends.py tests/test_backends.py && uv run mypy src/backends.py`
Expected: no errors.

- [ ] **Step 7: Full suite + commit**

Run: `uv run pytest -v`
Expected: all tests pass (scaffold + result + instance + backends).

```bash
git add src/backends.py tests/test_backends.py
git commit -m "feat: backend-selection interface with capability flags"
```

---

## Self-Review

**Spec coverage** (against `2026-06-27-pipeline-data-model-and-interfaces-design.md`):
- Module layout — Task 1 scaffolds `src/`; formulation submodules deferred (no logic to build yet, correctly out of scope).
- Instance data model, `(u,v)` identity, `to_digraph`, three resolved assumptions, `known_optimum` toy-only — Task 3 (assumptions encoded as validation + comments).
- Backend interface, `ApiFamily`, two capability flags, conservative `supports_lazy`, `available_backends` for `--solver-check` — Task 4.
- Result record, four-member `SolveStatus`, no `TIME_LIMIT`, optionals `None` — Task 2.
- Uniform `solve(...)` signature — documented as the produced contract; no implementation here because no formulation exists yet (out of scope, flagged).

**Deferred-but-flagged (not gaps):** `run.py` wiring of `--backend`/`--solver-check`/`--quick` belongs with the benchmark stage; the formulation packages and generator are later stages. This plan delivers only the shared spine, which is independently testable.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; the one runtime-API uncertainty (`SolverType` names) has an explicit verification step rather than a guess.

**Type consistency:** `Backend`, `ApiFamily`, `SolveStatus`, `Result`, `Instance`/`Arc`/`Job`, and the `resolve`/`load`/`available_backends` signatures match between their "Produces" blocks, the implementations, and the tests that consume them.
