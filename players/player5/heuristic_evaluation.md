> Archived: this document describes a previous implementation. See [README.md](README.md) for the current strategy and benchmark procedure.

# Turn and final-wall pruning evaluation

These measurements used the scenario snapshot on `naive-exhaustive-search`,
including the former `tiny_exhaustive.json` fixture. That fixture is not included
in this PR, and upstream now contains additional or changed scenarios. The
tables preserve the original results; they are not a benchmark of the current
upstream scenario suite.

Historical experiment: these timed runs used the previous shared **CPU-time** budget. The current player uses a shared **elapsed-time** deadline instead; these timed scores and worker comparisons must not be read as results for that new limit. Unbounded exhaustive correctness/node-count results remain applicable.

Four configurations were run across all 29 player scenarios. All retain the existing Euclidean-distance pruning, fixed gate placement, and one-gate model. Each uses four processes sharing a five-second CPU budget, with 4.5 seconds allocated to workers and their startup. The full-suite comparison is one run per scenario/configuration; budget-limited results are sensitive to startup and execution costs and are not optimality claims.

## Full scenario suite

| Configuration | Valid | Budget / no polygon | Exhausted / no polygon | Search completed | Total score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Distance only (baseline) | 14 | 13 | 2 | 7 | 36028.07 |
| + Turn feasibility | 15 | 12 | 2 | 8 | 37310.07 |
| + Final-wall closure | 14 | 13 | 2 | 7 | 36028.07 |
| + Both | 14 | 13 | 2 | 8 | 36028.07 |

No configuration produced invalid polygons, crashes, or simulator timeouts. The full-suite score improvement with turn pruning came from player6/n_shaped_room.json (0 to 1282); all other scores were unchanged across configurations.

## Repeated borderline scenario

Each configuration was run two more times on player6/n_shaped_room.json. Including the full-suite trial:

| Configuration | Valid trials / 3 | Scores |
| --- | ---: | --- |
| Distance only (baseline) | 0/3 | 0, 0, 0 |
| + Turn feasibility | 3/3 | 1282.0, 1282.0, 1282.0 |
| + Final-wall closure | 1/3 | 0, 0, 1282.0 |
| + Both | 2/3 | 0, 1282.0, 1282.0 |

The turn check gave the most consistent improvement on this scenario. The final-wall check adds little here; both together remain near the budget boundary, so the single full-suite count understates their potential and does not establish that both are always worse than turn pruning alone.

## Small exhaustive benchmark

Inventory: one 5-unit gate; four 5-unit walls and one 10-unit wall; two straight and four right connectors. Room: 20 by 20, fixed gate start (0, 0), heading 0. Serial search has no CPU limit. Every configuration exhausts the tree, finds the same four valid polygons, and returns the same best score of 1070.

CPU measurements are the median of three batches of 100 searches, clearing the turn cache before each search. They exclude multiprocessing startup. This is a small test of algorithm work, not a speedup prediction for every room.

| Configuration | Nodes | Node reduction | CPU per search (ms) | CPU reduction |
| --- | ---: | ---: | ---: | ---: |
| Distance only (baseline) | 84 | 0.0% | 2.883 | 0.0% |
| + Turn feasibility | 57 | 32.1% | 2.214 | 23.2% |
| + Final-wall closure | 83 | 1.2% | 2.773 | 3.8% |
| + Both | 56 | 33.3% | 2.124 | 26.3% |

Both rules are enabled by default and can be disabled independently. Turn feasibility accounts for most of the measured reduction; final-wall pruning gives a smaller additional saving in the exhaustive benchmark. Neither changes the set of valid completions. The remaining unsolved large scenarios still spend most of their search budget in earlier branches.

## Per-scenario scores

| Scenario | Baseline | Turn | Final wall | Both |
| --- | ---: | ---: | ---: | ---: |
| scenarios/players/player10/i_room.json | 1534.00 | 1534.00 | 1534.00 | 1534.00 |
| scenarios/players/player10/triangle_room.json | 1390.00 | 1390.00 | 1390.00 | 1390.00 |
| scenarios/players/player10/zig_room.json | 0.00 | 0.00 | 0.00 | 0.00 |
| scenarios/players/player2/cool.json | 0.00 | 0.00 | 0.00 | 0.00 |
| scenarios/players/player2/hard.json | 0.00 | 0.00 | 0.00 | 0.00 |
| scenarios/players/player2/simple_room.json | 1142.07 | 1142.07 | 1142.07 | 1142.07 |
| scenarios/players/player3/house.json | 0.00 | 0.00 | 0.00 | 0.00 |
| scenarios/players/player3/trapezoid.json | 0.00 | 0.00 | 0.00 | 0.00 |
| scenarios/players/player3/triangle.json | 1060.00 | 1060.00 | 1060.00 | 1060.00 |
| scenarios/players/player4/easy_rectangle_generous.json | 2032.00 | 2032.00 | 2032.00 | 2032.00 |
| scenarios/players/player4/irregular_room_second_gate.json | 1498.00 | 1498.00 | 1498.00 | 1498.00 |
| scenarios/players/player4/l_shaped_room_scarce.json | 1424.00 | 1424.00 | 1424.00 | 1424.00 |
| scenarios/players/player5/angled_room.json | 0.00 | 0.00 | 0.00 | 0.00 |
| scenarios/players/player5/l_shaped_room.json | 0.00 | 0.00 | 0.00 | 0.00 |
| scenarios/players/player5/rectangle_room.json | 0.00 | 0.00 | 0.00 | 0.00 |
| scenarios/players/player5/tiny_exhaustive.json | 1030.00 | 1030.00 | 1030.00 | 1030.00 |
| scenarios/players/player6/hallway.json | 0.00 | 0.00 | 0.00 | 0.00 |
| scenarios/players/player6/n_shaped_room.json | 0.00 | 1282.00 | 0.00 | 0.00 |
| scenarios/players/player6/spiral.json | 0.00 | 0.00 | 0.00 | 0.00 |
| scenarios/players/player7/horseshoe_room.json | 1412.00 | 1412.00 | 1412.00 | 1412.00 |
| scenarios/players/player7/spiral_room.json | 0.00 | 0.00 | 0.00 | 0.00 |
| scenarios/players/player7/square_room_w_cutouts.json | 0.00 | 0.00 | 0.00 | 0.00 |
| scenarios/players/player8/gate_bonus.json | 2232.00 | 2232.00 | 2232.00 | 2232.00 |
| scenarios/players/player8/irregular_blob.json | 590.00 | 590.00 | 590.00 | 590.00 |
| scenarios/players/player8/octagon_shape.json | 0.00 | 0.00 | 0.00 | 0.00 |
| scenarios/players/player9/32_edge_concave_room.json | 0.00 | 0.00 | 0.00 | 0.00 |
| scenarios/players/player9/crown.json | 9400.00 | 9400.00 | 9400.00 | 9400.00 |
| scenarios/players/player9/hallways.json | 8390.00 | 8390.00 | 8390.00 | 8390.00 |
| scenarios/players/player9/lollipop_trap.json | 2894.00 | 2894.00 | 2894.00 | 2894.00 |

## Reproduction and validation

These historical timed runs used commit d241ed7 and an older search. The current
CPU-budget benchmark does not reproduce that implementation. The historical
records refer to `logs/player5-heuristics-*-5s.json`, with confirmation runs in
`*-repeat1.json` and `*-repeat2.json`; these logs are not included here. See
[README.md](README.md) for current commands.

Validation: 23 search tests, including exact turn-feasibility comparison with independent connector-subset enumeration and exhaustive before/after polygon-count checks; 19 simulator geometry tests; Group 5 submission structure/style/smoke checks at a five-second limit.
