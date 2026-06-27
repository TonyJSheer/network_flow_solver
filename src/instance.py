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
