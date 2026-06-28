from __future__ import annotations

from src.generator import toy_instance
from src.subproblem import MinCutEvaluator


def test_open_network_mincut_is_bottleneck_arc() -> None:
    # toy: s->a cap 3, a->t cap 2. Min cut = {(a,t)} value 2.
    ev = MinCutEvaluator(toy_instance())
    cut = ev.evaluate(frozenset())
    assert cut.flow_value == 2
    assert cut.coeffs == {("a", "t"): 2}


def test_closed_bottleneck_gives_zero_flow_but_keeps_original_cap() -> None:
    ev = MinCutEvaluator(toy_instance())
    cut = ev.evaluate(frozenset({("a", "t")}))
    assert cut.flow_value == 0
    # the cut arc still reports its ORIGINAL capacity (master's y handles closure)
    assert cut.coeffs == {("a", "t"): 2}


def test_cache_counts_distinct_configs() -> None:
    ev = MinCutEvaluator(toy_instance())
    ev.evaluate(frozenset())
    ev.evaluate(frozenset())
    ev.evaluate(frozenset({("a", "t")}))
    assert ev.total_calls == 3
    assert ev.distinct_solves == 2
