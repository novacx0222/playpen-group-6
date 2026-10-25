# Group 3: enclosure search and growth

## Current strategy

A valid enclosure scores `1000 + A * area + C * perimeter + gate_bonus`.
`C` is negative; the bonus is `G` when gates occupy at least two different faces.

1. **Find a starting enclosure.** Try rectangles at different positions and
   orientations, then use beam search to explore more general polygons. Keep
   promising partial paths while checking whether their faces can be assembled
   from the available walls, gates, and connectors.
2. **Map available space.** Build a coarse map of the room and select routes from
   up to four sides of the starting enclosure toward reachable unused space.
3. **Grow and look ahead.** Try whole-face and partial-face extensions. Passage
   proposals align narrower extensions with openings along a route and alternate
   with ordinary growth proposals. Lookahead can continue or turn; passage-guided
   candidates receive up to three additional layers. Compatible growth fronts
   can merge when their new territory meets.
4. **Validate every saved candidate.** Check closure, room containment, crossings,
   inventory, connector angles, the 30-unit maximum straight face, and the gate
   requirement. Joint packing checks prevent reusing the same stock across the
   current fronts' combined outline.
5. **Return the best actual score found.** Preserve the starting enclosure and
   better intermediate candidates or unions, even if later exploration scores
   worse. Return no construction if the seed search finds no positive-scoring
   valid enclosure. The search is bounded and can miss better solutions.

The final-deliverable branch uses fixed work limits for local search decisions:
up to 2,000 rectangle placements and 128 beam-state expansions per starting
point in each beam pass. A shared 150 CPU-second deadline still applies across
the stages. This reduces dependence on short timing windows; the overall deadline
can still truncate the search. The submission checker defaults to 60 seconds per
scenario, so it can interrupt a longer run before the player's own deadline.

## Scoring correction and observed results

Growth previously saved the largest-area candidate. The correction saves the
highest official score, including perimeter cost and the actual gate bonus.

| Regression case | Before correction | With correction |
| --- | ---: | ---: |
| Group 3 trapezoid | 670 | 750 |
| Same room and stock, `C = -30` | -370 | 30 |

An earlier merged revision, `9d3d167`, reproduced the spiral example on
`scenarios/players/player6/spiral.json`: **52,816.2 points**, area **28,918.1**,
perimeter **2,190**, both gates on different faces, 103 extensions and three
merges. This is a recorded baseline for that revision, not a new benchmark of
the final branch or evidence of optimality.

## Room-space discussion and next experiments

- **Several starting enclosures:** compare growth from distinct positions or
  orientations; the current growth fronts all share one selected seed.
- **Region planning:** evaluate wide regions and the narrow passages connecting
  them using expected score and required materials, beyond the current map.
- **Passage lookahead:** test whether longer approach-and-turn sequences reach
  useful space that the current bounded lookahead misses.
- **Boundary adjustments:** test small contractions or boundary shifts that
  could remove an awkward corner and enable a better expansion.
- **Boundary-first candidates:** compare approximating a useful section of the
  room outline with growing outward from an interior seed.
- **Connector allocation:** investigate whether simpler boundaries leave enough
  scarce connectors to reach a more valuable region elsewhere.
- **Expansion priorities:** compare score-aware proposal ordering with the
  current area-and-route heuristic. Under `A = 2, C = -3`, an extension adding
  100 area units and 40 perimeter units gains 80 points if the gate bonus stays
  unchanged. All resulting constructions must still pass the official validator.

These are discussion and evaluation ideas; no performance improvement is claimed
without comparison under the same scenario, inventory, and CPU allowance.

## Run and check

From the repository root:

```bash
uv run main.py --gui --player 3 --scenario scenarios/players/player6/spiral.json --seed 1
uv run python -m unittest -v players.player3.test_growth
uv run python -m scripts.check_submission 3
```

The regression tests cover preserving the seed and better intermediate shapes,
gate bonuses and validated unions, beneficial growth, inventory preservation,
and returning the seed when the growth deadline has expired.

Checked on final-branch revision `8dcea5f`: all five regression tests and the
default 60-second submission check passed, with no timeout or invalid warnings.
The eight smoke scores were 2492.0 (invalid-examples scenario), 0 (L-shaped room),
0 (tight room), 2807.0 (simple rectangle), 1063.0 (house), 3087.2 (star),
750.0 (trapezoid), and 1060.0 (triangle). A zero here means no solution found.

## Selected AI assistance

AI assisted with targeted review, the score-retention correction and regression
tests, and strategy brainstorming. Two condensed examples of requests:

1. "Check the player for scoring issues and make a minimal, tested correction
   while preserving the existing approach."
2. "Suggest ways to use more room space through starting locations, passages,
   and boundary choices, and explain how to evaluate them."
