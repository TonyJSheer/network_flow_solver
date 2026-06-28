# Benders Performance Findings

Where the disaggregated Benders decomposition actually spends its time on the open
OR-Tools stack, and why the headline "Benders beats the direct MIP" result is hard to
reproduce here without a solver that has efficient lazy constraints.

Measured with `probe_sizes.py` and focused micro-probes on **ortools 9.15.6755**,
instance `generate_instance(size_idx=8, regime=TIGHT, seed=42)` =
**35 arcs, 354 jobs, horizon 438** unless noted. `NUM_THREADS = 6` for every backend.

---

## TL;DR

- **The subproblems are cheap.** Per-period min-cut evaluation is negligible and fully
  cached — never the bottleneck.
- **The master re-solve is the bottleneck.** Classic iterative-loop Benders re-proves an
  ever-larger master MIP from scratch every iteration; cost escalates as cuts accumulate.
- **On size=8 tight the iterative loop does not beat the direct MIP.** The direct
  time-indexed MIP solves to a proven optimum in ~7s; the loop spends more than that by
  iteration 4 without converging.
- **The algorithmic fix is lazy constraints (branch-and-Benders-cut)** — one search tree,
  cuts injected during the search, proof cost paid once. But on this stack lazy is blocked:
  GSCIP is broken/slow, CP-SAT/HiGHS have no lazy callbacks, and Gurobi is unlicensed.

---

## Finding 1 — subproblems are not the bottleneck

The `MinCutEvaluator` solves each period's min-cut with networkx and caches by the
frozenset of closed arcs. Cumulative subproblem time over 5 loop iterations at size=8:

| iter | distinct min-cuts solved | cumulative min-cut time | subproblem phase (this iter) |
|-----:|-------------------------:|------------------------:|-----------------------------:|
| 1 | 299 | 0.076s | 0.10s |
| 2 | 558 | 0.110s | 0.05s |
| 3 | 807 | 0.158s | 0.06s |
| 4 | 1000 | 0.195s | 0.05s |
| 5 | 1184 | 0.221s | 0.04s |

**~0.22s total** for 1184 distinct min-cuts. Optimizing the subproblem would not move the
needle.

## Finding 2 — the master re-solve is the bottleneck

Same run, master solve time per iteration (each round adds hundreds of disaggregated cuts,
and the loop re-proves master optimality from scratch):

| iter | master solve | cuts added | termination |
|-----:|-------------:|-----------:|-------------|
| 1 | 0.12s | 391 | OPTIMAL |
| 2 | 0.21s | 316 | OPTIMAL |
| 3 | 1.22s | 202 | OPTIMAL |
| 4 | **10.0s** | 144 | FEASIBLE (hit 10s cap, not proven) |
| 5 | 4.92s | 100 | OPTIMAL |

The master MIP gets harder every round as cuts pile up. This is the classic-Benders
pathology: repeated re-proof of a growing master.

## Finding 3 — loop Benders vs direct MIP (size=8 tight)

| method | backend | result | time |
|--------|---------|--------|-----:|
| direct | cp-sat-m | optimal 56090 | 6.6s |
| direct | cp-sat | optimal 56090 (nodes=3299) | 7.7s |
| benders (loop) | cp-sat | not converged in 6 iters | >40s (one master solve hit cap → FEASIBLE) |

Hint warm-starts (feeding the previous iteration's schedule to the master) **do not help** —
they are a slight net negative on easy instances and a no-op on hard ones, because CP-SAT's
difficulty here is proving the master's bound, not finding an incumbent.

## Finding 4 — lazy constraints are the fix, but blocked on this stack

Branch-and-Benders-cut injects cuts as lazy constraints inside a single branch-and-bound
tree, so the search/proof cost is paid once instead of once-per-iteration. Availability:

| backend | lazy support | status here |
|---------|--------------|-------------|
| Gurobi | native, first-class | **unlicensed** (`_gurobi_available()` → False) |
| SCIP (GSCIP) | via MathOpt constraint handler | **broken/slow** — see below |
| HiGHS | none (no runtime callbacks) | iterative loop only |
| CP-SAT | none (no user lazy constraints mid-search) | iterative loop only |

**GSCIP lazy is correct but pathological.** On size=4 tight the lazy callback path takes
~16s vs ~2.3s for the *same SCIP solver* run through the iterative loop. Every solve also
emits:

```
[scip_event.c:305] ERROR: SCIPcatchEvent does not support variable or row change events...
[gscip_event_handler.cc:125] ERROR: Error <-9> in function call
```

This error fires once per solve at event-handler init **regardless of whether lazy
constraints are added** (reproduced with a 2-variable observe-only callback). Correctness is
unaffected — MathOpt enforces lazy constraints through a SCIP *constraint handler*, not the
dead event handler. But the broken event handler appears to break incumbent-gating: the
MIP_SOLUTION callback fires far more often than there are distinct incumbents (5× on a
single-incumbent toy model; ~392 enforcement rounds on size=4 tight vs the loop's 10), which
is what makes the lazy path slower than just re-solving.

---

## Reproduction note — the breakdown logging goes to stderr

`probe_sizes.py` emits the master-vs-subproblem breakdown via `logging` (configured at
INFO). The configuration is correct, but Python's default `logging` handler writes to
**stderr**, while the result table is `print()`ed to **stdout**. Consequences when running
`uv run probe_sizes.py`:

- If stdout is piped/redirected (block-buffered) and stderr is not, the breakdown lines
  clump at the top instead of appearing before each table row.
- If stderr is dropped (`2>/dev/null`) or only stdout is captured, the breakdown lines
  disappear entirely.

To make the breakdown interleave reliably with the table, point logging at stdout:
`logging.basicConfig(level=logging.INFO, format="    %(message)s", stream=sys.stdout)`.

---

## Open levers (not yet pursued)

1. **Gap-limit / time-cap early master solves.** Early cuts are valid from any violating
   integer master solution; only the final termination check needs a proven master bound.
   This attacks the Finding 2 blowup without needing lazy.
2. **Cut management.** ~1150 cuts accumulate over 5 iterations; pruning dominated cuts would
   keep the master smaller.
3. **Test wide-window instances.** The spec's claim is specifically about *wide-window*
   instances; all data here is TIGHT, which gives the direct MIP an easy time (few schedule
   choices, compact flow model) and is plausibly Benders' worst case.
