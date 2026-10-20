# Player6 local 45-degree benchmark

Historical report: fixed budgets, before dynamic CPU allocation and beam upper-bound pruning.

Baseline disables local refinement; local45 enables it. All other CPU budgets are equal.
CPU time includes player construction and search; final verification is outside that budget.
Deadline-limited runs can vary. These cases do not establish a global improvement guarantee.

Budgets (CPU seconds): {"beam_seconds": 20.0, "grid_seconds": 20.0, "local_seconds": 2.0, "search_seconds": 45.0}

| Case               | Repeat |          Baseline |           Local45 | Score delta | CPU baseline/local | Local moves |
|--------------------|-------:|------------------:|------------------:|------------:|--------------------|------------:|
| tiny_room          |      1 |    990.00 (valid) |    990.00 (valid) |       +0.00 | 0.007 / 0.006      |           0 |
| insufficient_stock |      1 |       0.00 (none) |       0.00 (none) |       +0.00 | 0.037 / 0.036      |           0 |
| shared_lengths     |      1 |   1086.07 (valid) |   1086.07 (valid) |       +0.00 | 0.247 / 0.249      |           0 |
| diagonal_room      |      1 |   1191.42 (valid) |   1191.42 (valid) |       +0.00 | 0.044 / 0.044      |           1 |
| scarce_seed1       |      1 |   1490.00 (valid) |   1490.00 (valid) |       +0.00 | 45.004 / 45.003    |           0 |
| scarce_seed2       |      1 |   1262.00 (valid) |   1262.00 (valid) |       +0.00 | 24.742 / 24.768    |           0 |
| narrow             |      1 |   1076.00 (valid) |   1076.00 (valid) |       +0.00 | 2.133 / 2.165      |           0 |
| spiral             |      1 | 101390.00 (valid) | 101390.00 (valid) |       +0.00 | 29.331 / 30.834    |           0 |
