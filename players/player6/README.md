# Player6: grid search, beam search, and local 45-degree refinement

All implementation, tests, benchmark code, and benchmark outputs for this change
are inside `players/player6/`.

## Running the tests

From the repository root:

```bash
uv run python -m unittest discover -s players/player6 -p 'test_*.py'
uv run ruff check players/player6
uv run python -m scripts.check_submission 6 --no-run
```

The tests cover scarce/fixed-seed inputs, small rooms, insufficient inventory,
equal-length walls and gates, real gate-face bonuses, CPU-budget fallback,
rotated rooms, a construction starting in the middle of a straight face,
strict score improvements, and preservation of both input inventory and seed.
Budget tests also cover active simulator timers, proportional allocation,
programmatic limits, repeated GUI-style runs, and safe beam pruning.

## Dynamic CPU budget

The shared Player API supplies room, inventory, and weights, but no time limit.
Player6 observes the active framework `CpuLimit` on Unix without changing its
timer or signal handler. This respects `--cpu-limit`, programmatic Game limits,
constructor CPU time already spent, and a fresh budget for each GUI run.
When that timer is unavailable it reads `--cpu-limit` from the command line,
falling back to the framework's default (currently 300 seconds). Direct API
callers can use `Player6(room, inventory, weights, cpu_limit=60)`; this is also
needed for programmatic custom limits on platforms without the Unix timer.

Reserve 3% of the configured limit for returning safely (minimum 0.05 seconds,
maximum 5 seconds, and never more than 20% of the remaining time). Allocate the
remaining CPU time as an initial policy:

| Stage | Share | 60-second limit | 300-second limit |
|---|---:|---:|---:|
| Orthogonal candidates and placement | 45% | 26.19 s | 132.75 s |
| Beam search with 45-degree turns | 50% | 29.10 s | 147.50 s |
| Local 45-degree refinement | 5% | 2.91 s | 14.75 s |
| Safety reserve | | 1.80 s | 5.00 s |

These nominal figures assume negligible constructor time. Within the first
stage, grid construction gets 30% of usable CPU and rectangle/L placement gets
15%. Beam reclaims unused earlier time while reserving the unspent local quota.
Local search can run before beam using spare orthogonal time, and after beam;
both passes share one quota. Stages stop when their candidates are exhausted,
so the limit need not be consumed. With insufficient diagonal connectors, the
unavailable stages receive no budget and orthogonal work gets more time.
Explicit stage caps are available through `stage_limits={"grid": 20, "beam": 20,
"local": 2}`; overlarge caps scale down to respect the total budget.

Beam still has a 12-face limit. Player6 uses an optimistic area/score upper
bound from its maximum perimeter, room area, inventory, and gate bonus. If even
that bound cannot beat the incumbent, beam is skipped. This does not skip a
potential improvement. In particular, extra CPU alone cannot make this small
beam neighborhood competitive with the large spiral grid solution.

Partial beam states distinguish consumed right, diagonal, and straight
connectors as well as endpoint, direction, and edge lengths. Each state retains
at most two distinct exact boundary prefixes before the existing global beam
selection. This preserves some geometric diversity without increasing the
global beam width; the per-state cap remains heuristic.

Material assignment rejects outlines whose total length exceeds the longest
stock pieces usable with the available straight connectors: `n` faces and `s`
straight connectors can use at most `n+s` pieces. Only proven failures enter
the permanent feasibility cache. Queries stopped by a node/CPU cutoff are
deferred for the current beam pass and retried after widening, instead of
restarting for every outline with the same face lengths. Closed outlines with
deferred assignments remain eligible for evaluation in the next pass.

Closed beam candidates must have at least one gate face. After placement, a
validated baseline is retained before retrying material assignment for two
distinct gate faces. Gate-constrained queries have separate caches from prefix
feasibility checks. Each optional retry gets up to 0.01 CPU seconds; all such
retries share at most 5% of the existing beam budget. A cutoff or impossible
bonus keeps the original plan. The current Player6 score is also passed to
beam search, so closed candidates whose optimistic score cannot improve it
are skipped before placement.

The percentages are starting values, not a measured optimum across all rooms.
Increasing time cannot improve a stage whose candidate set is exhausted.
A single native geometry call may exceed a cooperative deadline until it returns;
the simulator's hard CPU guard remains authoritative.

Each run also reports actual CPU time for face-library generation, the grid,
orthogonal placement, local refinement, and beam search. `phase_seconds` keeps
these measurements; `beam_history` records improvements as `(CPU seconds since
the start of this run, full simulator score)`. Both appear in benchmark JSON/CSV,
and the run note retains the budget explanation along with the measurements.

## Benchmarking

```bash
# Comparison on nine cases, including spiral, diagonal hexagon, and two scarce seeds.
uv run python -m players.player6.benchmark

# Repeat selected cases to assess variation under CPU deadlines.
uv run python -m players.player6.benchmark --case spiral --case scarce_seed1 --repeats 3

# Allocate automatically from a different total limit.
uv run python -m players.player6.benchmark --case scarce_seed1 --cpu-limit 60

# A short smoke comparison with smaller, explicitly split budgets.
uv run python -m players.player6.benchmark --case tiny_room --case insufficient_stock --search-seconds 1 --grid-seconds 0.3 --beam-seconds 0.4 --local-seconds 0.1
```

The default outputs are `benchmark_results.json`, `benchmark_results.csv`, and
`benchmark_results.md` in this directory. `--output PATH` changes their location.
Each row records validity/no-solution/timeout/error, score, area, perimeter,
gate count and actual gate bonus, CPU time, wall time, grid CPU time, local-search
statistics, allocated CPU budgets, beam pruning, and whether the input inventory was preserved. Returning no solution
is expected for the deliberately insufficient-stock case and scores zero.

`baseline` runs the current grid + rectangle/L + beam strategy with local moves
disabled. `local45` enables the local pass. Both have the same overall CPU limit
(45 seconds by default for this benchmark command) and use automatic stage
allocation unless caps are supplied. The simulator guard uses that exact total
limit, including player construction. Each row records the actual allocation.

This is an ablation of local refinement, not a comparison against the historical
pre-grid player. Deadline-limited searches can vary between runs, and improving
an incumbent locally does not establish that every complete run will beat a
separate baseline run.

The checked-in `benchmark_results.*` files are historical results from the fixed
45/20/20/2-second implementation, before dynamic budgets and beam upper-bound
pruning. They contain 16 runs across eight cases: 14 valid
solutions and two expected no-solutions for insufficient stock, with no invalid
solutions, crashes, timeouts, or inventory mutations. Final scores were equal
between variants on this set. The diagonal-room local pass accepted an earlier
improvement that beam search also found; spiral found no local improvement and
took about 1.5 extra CPU seconds. These results do not show an end-to-end score
gain yet. Use `stage_limits={"local": 0}` to disable the pass, or compare different
budgets with the benchmark CLI. See [benchmark_results.md](benchmark_results.md)
and the JSON/CSV files for all scores, timings, and local-search statistics.

## How the local move works

Replacing an integer-grid corner with one diagonal usually requires a noninteger
piece length. Instead, `local_search.py` translates a chain of two to four faces
and inserts equal-length, opposite diagonal faces at its ends. Their
`sqrt(2)/2` coordinate contributions cancel exactly. The old faces keep their
integer lengths and assigned pieces. Only spare pieces/straight connectors are
used for the new faces; all right/diagonal connector counts are recomputed.

The new outline is tried at its current position and a few bounding-box-based
translations without changing its orientation. The official validator checks
closure, self-intersections, containment, all inventories, the 30-unit face limit,
and gates. A move is accepted only if the official score strictly increases.
On timeout, the best validated construction is retained. The original seed and
inventory are never modified.

For example, a gated 10-by-10 square in a 15-by-15 room can use two spare
5-unit walls and four diagonal connectors. Under `A=2, C=-3, G=100`, its area
increases from 100 to approximately 170.711 and its score increases from 1080
to approximately 1191.421, with an exactly closed integer-length construction.

The pass is intentionally limited: it keeps existing material assignments,
uses only this paired-diagonal neighborhood and a few translations, and may
find no move when stock, connectors, placement, or remaining time are tight.
The general beam search remains necessary.

## Experimental staircase replacement

`inscribed_diagonal.py` provides a second local neighborhood: replace two
opposite staircases with equal-length diagonal faces and reassign the pieces.
It remains disabled by default (`Player6.INSCRIBED_DIAGONALS = False`). The
original branch did not observe a final-score improvement from enabling it.

For experiments, set `player.INSCRIBED_DIAGONALS = True` before running. It shares
the existing local-refinement CPU quota, including time used by enumeration,
material assignment, and placement. Only official-validator-approved score
improvements are retained. Separate `inscribed_diagonal_stats` track attempts,
feasible candidates, accepted improvements, and CPU time. Deadline checks cover
staircase and pair enumeration too. The `diagonal_hexagon` benchmark case is a
small room with diagonal boundaries for testing these strategies.
