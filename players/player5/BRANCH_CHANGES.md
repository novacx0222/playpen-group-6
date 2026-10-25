> Archived: this document describes a previous implementation. See [README.md](README.md) for the current strategy and benchmark procedure.

# Player 5 branch changes

This branch replaces Player 5's parallel depth-first enumeration with a
bounded-width beam search and adds area-focused face construction and second
gate handling.

## Search

- The search is serial. It expands partial enclosures one complete straight
  face at a time and retains at most 2048 states in the outer beam.
- The default search deadline is 20 seconds of elapsed time. The simulator's
  separate CPU limit is controlled by `--cpu-limit`.
- `workers` and the former pruning switches remain accepted by the constructor
  for compatibility, but the beam implementation does not use them.
- `search_complete` remains false because beam truncation does not prove that
  the best possible enclosure was found.

## Heuristic

For each partial chain, the search estimates the area it would enclose if its
endpoint connected directly back to the gate start. Lower heuristic values are
preferred:

```text
distance(endpoint, gate start) - 0.075 * estimated area
```

The coefficient balances length units against square-length area units. It was
chosen empirically on `easy_rectangle_generous.json`; it is a useful setting
for the current experiments, not a proven optimum for every room.

## Face construction

Each outer search step chooses a turn and builds a face from one or more walls
joined by straight connectors. The first step may also extend the gate's face.
A bounded knapsack over available wall lengths finds combinations that fit the
remaining face-length limit. For each turn, the search keeps a longest fit, the
candidate with the best area/distance heuristic, and a shorter alternative.
Geometry checks apply to the full straight run, while the construction retains
the individual wall pieces and connectors.

## Gate placement and bonus

By default, the first gate starts at the first endpoint of the longest room
boundary edge and points along that edge. An explicitly supplied `start` still
overrides this position.

After a valid one-gate construction is closed, Player 5 looks for a wall on a
different face that can be replaced by an available gate of the same length.
It validates the converted construction and keeps the replacement when the
simulator confirms the second-gate bonus. This preserves the geometry and uses
the existing inventory and face rules.

## Recorded comparisons

- `easy_rectangle_generous.json`, seed 1: a valid two-gate construction scored
  3439.06 (area 1388.53, perimeter 146) and returned in 12.44 seconds.
- `square_room_w_cutouts.json`, seed 1, 30-second simulator CPU cap: Player 5
  scored 2530 (area 1160, perimeter 180, two gates) in 18.21 CPU seconds.
  Player 4 scored 2592.2 (area 1201.45, perimeter 180, two gates) in 26.22 CPU
  seconds. Both runs returned valid constructions.

The Python source compiled and `git diff --check` passed during development.
The full Player 5 unit-test suite was not run on this branch.
