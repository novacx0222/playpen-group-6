# Player 5

Build a fence, then improve it by finding a new route between two of its corners.

## Start with one example

The input contains the room's corners, counts of walls/gates/connectors, and
three scoring weights. The output is a `Construction`: a starting point and
direction, an ordered list of pieces, and the connector before each piece.
Returning `None` earns zero.

Suppose the room is 20 by 20, and we have two 10-unit walls, two 10-unit gates,
and four right-angle connectors. A 10-by-10 square uses exactly that stock.
Its area is 100 and perimeter is 40. With weights `A=2`, `C=-3`, `G=100`,
its score is `1000 + 2*100 - 3*40 + 100 = 1180`. The bonus applies because
the gates occupy different faces.

A **face** is one straight run between corners. It can contain several pieces
joined by straight connectors, but its total length must stay at most 30.

## Follow the code in this order

1. **`Player5.build_enclosure` in `player.py`** runs the searches and gives each
   a CPU deadline. Searches report complete constructions as they find them.
2. **`PolygonPacker.pack` in `packing.py`** turns proposed corners into actual
   walls, gates and connectors. It checks stock across the whole fence, tries
   to obtain the gate bonus, and uses the official validator before returning.
3. **`Player5._keep_candidate`** saves the best positive official score. It also
   retains up to twelve different starting shapes for later improvements.
4. **`search_improvements` in `refinement.py`** takes starting shapes, tries
   geometric changes, allocates pieces, and keeps improvements. Both kinds
   of local change use this same loop.
5. **`_replacement_paths` in `path_refinement.py`** replaces one to three
   existing faces with two to five new faces. It chooses allowed turns and
   solves integer lengths so the new route meets the same endpoints.

The remaining functions generate geometry or make these operations cheaper.
They do not change the scoring rules or consume the original inventory.

## Where the shapes come from

| Search | What it proposes | File |
| --- | --- | --- |
| Room boundary | The room's existing outline, if buildable | `player.py` |
| Templates | Rectangles, hexagons, octagons, staggered rectangles, and corner cuts at different positions and angles | `templates.py` |
| Room grid | Joined interior cells that follow winding rooms; real notches split faces longer than 30 | `room_grid.py` |
| Face moves | Slide a side or add a rectangular extension | `refinement.py` |
| Short paths | Replace one to three faces with two to five faces, including 45-degree turns | `path_refinement.py` |
| Beam fallback | Extend partial chains when the other searches found no starting fence | `beam_fallback.py` |

Template placement finds translations that fit the room; nonconvex rooms also
use a stock check before shortlisting shapes. Grids vary cell sizes, offsets
and directions. Short-path moves choose early integer lengths and solve the
remaining signed direction sums so their endpoints reconnect.

The searches cache repeated stock problems and retain several promising
outlines. A failed proposal never replaces the best valid result. This is a
bounded heuristic search, with no claim of exhaustive coverage or optimality.

Starting shapes must score above zero. We ignore piece junctions on straight
faces and keep each outline once, updating its pieces if the same outline later
earns a better gate bonus. Whenever a new
outline joins the retained pool or an existing one improves, we rank by score
and first take one from each category: the centroid's cell in a 4-by-4 grid
over the room's bounding box, the right-angle corner count
divided by four and rounded down, and the diagonal corner count divided by four
and rounded down. Remaining slots take the highest scores. Short paths use the
starting shapes retained after face moves.

To bound route enumeration, four- and five-face replacements must use at least
three direction axes. Replacing one face with two is excluded because allowed
turns and integer lengths cannot reconnect those endpoints.

## What distinguishes the search

Other submissions also use templates, room grids, stock backtracking and local
improvement. Our main distinguishing operation is the general replacement of
a short boundary section, including changes to its number of faces and turns.
The closest reviewed alternatives use prescribed rectangular bumps, paired
diagonal insertions, translated runs or growing chains. Our search enumerates
turn patterns within the replacement window, solves their endpoint constraints,
then reallocates pieces across the whole fence. The grids and templates supply
starting fences for this operation.

## Time budget

The default budget is 240 CPU seconds under the official 300-second limit.
For a smaller active simulator limit, the player uses at most 80% of the
remaining CPU time as its internal target.

With the default budget, phase caps are 1 second for the boundary, 12 for
templates, 70 for grids, 40 for face moves and 50 for short paths: 173 seconds
in total. The extra budget is headroom; phases can also finish early. Smaller
budgets reduce the first four caps. Every phase, including short paths, is
clipped to the overall deadline.

If the first three searches produce a positive-score starting fence, we run
face moves and then short paths. Otherwise the beam fallback receives all
remaining time. No search runs without a usable gate and at least 360 degrees
of available corner turns. Checks are cooperative, so one geometry or validation
operation can cross a cap.

The official timer covers loading `player.py`, construction and search,
including imports inside the call and the player's validation. The final independent
validation runs after the timer stops.

## Run and check

```sh
uv run main.py --player 5 --scenario scenarios/players/player6/spiral.json --seed 1
uv run main.py --player 5 --scenario scenarios/simple_rectangle.json --seed 1 --gui
uv run python -m unittest discover -s players/player5 -t .
uv run python -m scripts.check_submission 5
```

`benchmark.py` runs scenarios in fresh processes and records inputs, source
hashes, CPU consumption and officially validated outputs:

```sh
uv run python -m players.player5.benchmark \
  --budget 10 --cpu-limit 30 --output /tmp/player5-results.json
```

Without paths, it runs all published scenario files. Team comparisons should
use identical inputs and official CPU allowances, preserve unsuccessful runs,
and treat score differences within `1e-6` as ties. Timed results can vary.
The older evaluation documents are historical records of earlier versions.
