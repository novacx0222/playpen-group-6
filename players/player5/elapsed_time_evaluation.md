> Archived: this document describes a previous implementation. See [README.md](README.md) for the current strategy and benchmark procedure.

# Five-second elapsed-time evaluation

These measurements used the scenario snapshot on `naive-exhaustive-search`,
including the former `tiny_exhaustive.json` fixture. That fixture is not included
in this PR, and upstream now contains additional or changed scenarios. The
tables preserve the original results; they are not a benchmark of the current
upstream scenario suite.

The player now uses a five-second elapsed-time deadline measured with time.monotonic(), starting before process startup. Every worker receives exactly the same absolute deadline. The deadline is not divided by worker count, and late-starting workers do not receive a fresh five seconds. All pruning rules are enabled. The simulator has a separate 300-second parent CPU guard.

Previous reports used a shared CPU budget and do not answer the intended elapsed-time question. These runs replace that comparison. One run per scenario/configuration, on a machine with 8 logical CPUs; scores near the deadline can vary.

## Four workers across all player scenarios

All 29 scenarios: {'valid': 20, 'deadline/no solution': 7, 'no solution': 2}. No invalid polygons, crashes, or simulator CPU timeouts.

Most deadline-limited runs returned at about 5.03 seconds. A native geometry call or process teardown can extend return time slightly beyond the cooperative cutoff. Searches that exhaust their tree return early.

## Corrected four-versus-64 comparison

| Scenario | Workers | Score | DFS nodes | Elapsed seconds | Combined CPU seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| scenarios/players/player5/tiny_exhaustive.json | 4 | 1030.00 | 7 | 0.216 | 0.832 |
| scenarios/players/player5/tiny_exhaustive.json | 64 | 1030.00 | 67 | 2.567 | 16.982 |
| scenarios/players/player4/easy_rectangle_generous.json | 4 | 2032.00 | 213980 | 5.030 | 20.071 |
| scenarios/players/player4/easy_rectangle_generous.json | 64 | 2032.00 | 93102 | 5.447 | 35.887 |
| scenarios/players/player6/n_shaped_room.json | 4 | 1422.00 | 159765 | 5.035 | 19.851 |
| scenarios/players/player6/n_shaped_room.json | 64 | 1282.00 | 122230 | 5.434 | 36.789 |

64 workers now visit nodes and return valid polygons in all three sampled scenarios. The old zero-node result came from splitting CPU time into 70 ms worker allocations and is not applicable here. However, spawning 64 processes still consumes substantial elapsed time and CPU on this 8-core machine. It did not improve the sampled scores: easy_rectangle_generous tied, and n_shaped_room scored lower than with four workers.

With 64 workers, process shutdown/result collection took about 0.4 seconds after the five-second search cutoff, so measured return time was about 5.45 seconds. Four workers remain the default. This implementation has a cooperative search deadline, not a hard process-level wall-time kill.

Combined CPU is measured from parent process CPU plus OS-reported CPU for completed child processes after shutdown on Unix. It is a reported measurement, not the stopping criterion.

## All four-worker scores

| Scenario | Result | Score | Search complete | Elapsed seconds |
| --- | --- | ---: | --- | ---: |
| scenarios/players/player10/i_room.json | valid | 1912.47 | False | 5.032 |
| scenarios/players/player10/triangle_room.json | valid | 1390.00 | True | 0.593 |
| scenarios/players/player10/zig_room.json | deadline/no solution | 0.00 | False | 5.033 |
| scenarios/players/player2/cool.json | valid | 8995.82 | False | 5.035 |
| scenarios/players/player2/hard.json | deadline/no solution | 0.00 | False | 5.030 |
| scenarios/players/player2/simple_room.json | valid | 1142.07 | True | 0.748 |
| scenarios/players/player3/house.json | no solution | 0.00 | True | 0.175 |
| scenarios/players/player3/trapezoid.json | no solution | 0.00 | True | 0.179 |
| scenarios/players/player3/triangle.json | valid | 1060.00 | True | 0.259 |
| scenarios/players/player4/easy_rectangle_generous.json | valid | 2032.00 | False | 5.030 |
| scenarios/players/player4/irregular_room_second_gate.json | valid | 1498.00 | False | 5.032 |
| scenarios/players/player4/l_shaped_room_scarce.json | valid | 1424.00 | True | 0.392 |
| scenarios/players/player5/angled_room.json | deadline/no solution | 0.00 | False | 5.032 |
| scenarios/players/player5/l_shaped_room.json | valid | 1414.00 | False | 5.029 |
| scenarios/players/player5/rectangle_room.json | valid | 1859.98 | False | 5.033 |
| scenarios/players/player5/tiny_exhaustive.json | valid | 1030.00 | True | 0.216 |
| scenarios/players/player6/hallway.json | deadline/no solution | 0.00 | False | 5.032 |
| scenarios/players/player6/n_shaped_room.json | valid | 1422.00 | False | 5.035 |
| scenarios/players/player6/spiral.json | deadline/no solution | 0.00 | False | 5.034 |
| scenarios/players/player7/horseshoe_room.json | valid | 1412.00 | False | 5.029 |
| scenarios/players/player7/spiral_room.json | deadline/no solution | 0.00 | False | 5.029 |
| scenarios/players/player7/square_room_w_cutouts.json | valid | 1799.89 | False | 5.034 |
| scenarios/players/player8/gate_bonus.json | valid | 2232.00 | True | 0.251 |
| scenarios/players/player8/irregular_blob.json | valid | 590.00 | False | 5.033 |
| scenarios/players/player8/octagon_shape.json | valid | 1288.00 | False | 5.031 |
| scenarios/players/player9/32_edge_concave_room.json | deadline/no solution | 0.00 | False | 5.031 |
| scenarios/players/player9/crown.json | valid | 9400.00 | False | 5.102 |
| scenarios/players/player9/hallways.json | valid | 8390.00 | False | 5.033 |
| scenarios/players/player9/lollipop_trap.json | valid | 2894.00 | False | 5.031 |

## Historical commands

These commands describe the older elapsed-time implementation. Their
`--time-limit` option is not accepted by the current benchmark. See
[README.md](README.md) for current CPU-budget commands.

```sh
uv run python -m players.player5.benchmark --workers 4 --time-limit 5 --output logs/player5-wall-4-workers.json
uv run python -m players.player5.benchmark --workers 64 --time-limit 5 --output logs/player5-wall-64-workers.json scenarios/players/player4/easy_rectangle_generous.json scenarios/players/player6/n_shaped_room.json
```

Validation: 25 search tests, including checks that all workers receive the same absolute deadline and that an already-expired deadline prevents search; 19 simulator geometry tests; Group 5 submission structure/style/smoke checks.
