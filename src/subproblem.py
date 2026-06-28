"""Per-period subproblem evaluators for Benders decomposition.

Given a fixed schedule, each period is an independent s->t max-flow on the
network with arcs-under-maintenance set to capacity 0. The LP dual of max-flow
is min-cut, so the analytic evaluator solves the flow combinatorially with
networkx and reads the min-cut straight off the result -- no LP in the loop.

The returned cut is the disaggregated Benders optimality cut for that period:

    theta_t <= sum_{a in min-cut} cap_a * y[a,t]

where cap_a is the arc's ORIGINAL capacity and y[a,t] is the master's
arc-availability proxy. Coefficients are keyed by arc; the evaluator never sees
master variables.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from src.instance import Instance


@dataclass(frozen=True)
class PeriodCut:
    """One disaggregated optimality cut. ``coeffs`` maps each min-cut arc to its
    original capacity; ``flow_value`` is the period's max flow under the outage."""

    flow_value: int
    coeffs: dict[tuple[str, str], int]


class MinCutEvaluator:
    """Analytic evaluator: networkx max-flow, read the source-side min-cut.

    Caches results by the frozenset of closed arcs, so identical outage patterns
    across periods/iterations are recalled rather than re-solved.
    """

    def __init__(self, instance: Instance) -> None:
        self._instance = instance
        self._caps: dict[tuple[str, str], int] = {(a.u, a.v): a.capacity for a in instance.arcs}
        self._cache: dict[frozenset[tuple[str, str]], PeriodCut] = {}
        self.total_calls = 0
        self.distinct_solves = 0

    def evaluate(self, closed_arcs: frozenset[tuple[str, str]]) -> PeriodCut:
        self.total_calls += 1
        cached = self._cache.get(closed_arcs)
        if cached is not None:
            return cached
        self.distinct_solves += 1
        cut = self._solve(closed_arcs)
        self._cache[closed_arcs] = cut
        return cut

    def _solve(self, closed_arcs: frozenset[tuple[str, str]]) -> PeriodCut:
        g: nx.DiGraph = nx.DiGraph()
        for (u, v), cap in self._caps.items():
            g.add_edge(u, v, capacity=0 if (u, v) in closed_arcs else cap)
        value, (reachable, _) = nx.minimum_cut(g, self._instance.source, self._instance.sink)
        coeffs = {
            (u, v): cap
            for (u, v), cap in self._caps.items()
            if u in reachable and v not in reachable
        }
        return PeriodCut(flow_value=int(value), coeffs=coeffs)
