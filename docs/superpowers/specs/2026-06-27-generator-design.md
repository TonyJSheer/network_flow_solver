# Stage 1 — Instance Generator Design (`src/generator.py`)

Date: 2026-06-27
Status: approved (brainstorming complete)

## Purpose

Emit random maintenance-scheduling instances matching the Pearce-Forbes / Boland regime,
serialised to JSON via the existing `Instance` data model. Plus a tiny hand-checkable toy
instance for unit tests. This is the first build stage; downstream solvers and the benchmark
consume the JSON.

## Grounding in the paper (`1603.02378v2.pdf`, §2, §3.2, §4.1)

Confirmed-from-paper parameters:

- 8 networks of increasing size; 10 randomly-generated job-lists each.
- Every arc carries `m ~ U[5, 15]` maintenance jobs.
- Job duration `~ U[10, 30]` time steps.
- **Window regimes are *number of possible start times* (window width), not absolute time
  ranges.** In our data model `deadline` is the latest *start*, so window width =
  `deadline - release + 1`:
  - `TIGHT`  = 1–10 start times (instance set 3, easy)
  - `MEDIUM` = 1–35 start times (instance set 1)
  - `WIDE`   = 25–35 start times (instance set 2, **hard**)
  `WIDE` is hard precisely because the window often exceeds the job duration, which weakens
  the LP relaxation (§3.2) — this is the crux the demo reproduces.
- **No two jobs on the same arc may overlap** (Boland assumption, §2, line ~131). The
  generator guarantees a feasible non-overlapping schedule exists by construction.

Deferred by the paper to Boland et al. [1] (not in repo) — **we choose, demo-scaled**:

- The 8 network sizes and the time horizon `H`. Decision: modest sizes (net 1 ≈ 4 arcs …
  net 8 ≈ 30 arcs); per-instance horizon sized to fit the busiest arc's non-overlapping jobs.
- Arc capacity range. Decision: integer `U[10, 100]`.

## Public API

```python
class Regime(Enum):          # carries (lo, hi) = number-of-start-times range
    TIGHT  = (1, 10)
    MEDIUM = (1, 35)
    WIDE   = (25, 35)

def generate_instance(size_idx: int, list_idx: int, regime: Regime, seed: int) -> Instance
def generate_suite(seed: int) -> Iterator[Instance]   # full 8 × 10 × 3 = 240 sweep
def toy_instance() -> Instance                         # hand-checkable, known_optimum = 8
```

CLI: `uv run python -m src.generator --seed 0 --out instances/ [--regime wide] [--size 3]`.
Serialises each `Instance` via the existing `instance.save()`. `instances/` is gitignored;
tests use the API directly, not files.

## Network generation (layered DAG)

A monotone size schedule indexed 1–8 scales `(num_layers, layer_width)` from tiny to ~30
arcs, held in one whiteboard-readable table constant at the top of the module.

Construction:
- Nodes: `s`, then `L` layers of `W` nodes, then `t`.
- Arcs: full bipartite between adjacent layers; `s` → all of layer 1; last layer → `t`;
  plus a few random skip arcs for irregular cuts. Guarantees `s`-reaches-`t` and a
  non-trivial min-cut.
- Capacities: integer `U[10, 100]`.

## Job generation (per arc, the part the paper pins)

Every arc gets `m ~ U[5, 15]` jobs. For each: `duration ~ U[10, 30]`, window width
`n ~ U[regime.lo, regime.hi]`.

Per-arc sequential layout guarantees the no-overlap assumption is satisfiable:
- Walk along the horizon placing jobs in order. Job `k` gets `release_k`, and
  `deadline_k = release_k + n_k - 1` (exactly `n_k` legal start positions).
- The next job's earliest release starts after the previous job's *latest completion*
  (`deadline_k + duration_k - 1 + 1`), so even worst-case starts never overlap. A small
  random gap adds slack/variety.
- The arc's required horizon = the final job's latest completion. **Instance horizon
  `H = max over all arcs`** of that.

This makes both the all-earliest and all-latest schedules valid and non-overlapping, so every
generated instance is feasible, while the solver still chooses starts to maximise flow.

With 5–15 non-overlapping jobs/arc, `H` realistically lands in the low hundreds for the larger
nets (busiest arc dominates). `--quick` uses only the small sizes.

## Forward dependency (flagged, not silent)

**No-overlap-per-arc is also a Stage 2/3 formulation constraint.** `direct_mip.py` and
`benders.py` must add a per-arc no-overlap constraint on job schedules. The `Instance` model
needs no change — windows stay independent; feasibility is guaranteed by the generator.

## Reproducibility & toy

- Single master `--seed`. Each instance derives an independent `numpy.random.default_rng`
  stream from a stable hash of `(seed, size_idx, list_idx, regime)`, and records that derived
  seed in `Instance.seed` so a single instance regenerates without replaying the suite.
- `toy_instance()` builds the existing `tests/fixtures/toy.json` in code
  (`known_optimum = 8`). Random instances leave `known_optimum = None`.

## Testing (TDD)

Focus: **every generated instance is a legal graph/instance**, especially the `--quick`
(small) sizes. Determinism gets light coverage only.

- **Legality:** every generated instance passes `instance._validate` without
  raising — for all 8 sizes and all three regimes, with emphasis on the small `--quick`
  sizes. Source reaches sink; no parallel arcs; all jobs reference real arcs; every job
  fits the horizon.
- **Feasibility invariant:** each arc admits a non-overlapping schedule (assert the
  construction invariant directly).
- **Parameter ranges:** per-job window width ∈ regime range; duration ∈ [10, 30];
  jobs/arc ∈ [5, 15].
- **Toy:** `toy_instance()` round-trips to the committed fixture and has `known_optimum = 8`.
- **Determinism (light):** same seed reproduces an instance; no exhaustive byte-equality
  sweep.
- **Monotonicity (light):** arc count non-decreasing across the 8 sizes.
```
