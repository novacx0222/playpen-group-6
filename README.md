# Project 2, Playpen

Simulator scaffold for COMS 4444 Project 2: Playpen. You are given a room
footprint (polygon), a fixed inventory of walls, gates, and connectors, and a
set of scoring weights. Your job is to build a closed baby enclosure out of
that inventory that fits inside the room.

**Groups: your player goes in `players/player<N>/` (N is your group number, 1-10),
which is already set up for you with an empty player. Start with
[PLACEMENT_GUIDE.md](PLACEMENT_GUIDE.md).** The simulator itself -- geometry,
validation, scoring, and the GUI -- is complete. The repo also ships a worked
example strategy (`--player e`) and a demo that shows what invalid enclosures
look like (`--player i`).

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
brew install uv
```

### macOS (homebrew)

Python from homebrew doesn't include `tkinter`. If you're using a homebrew
Python interpreter, install it separately:

```bash
brew install python-tk@3.13
```

## Running the simulator

```bash
uv run main.py <CLI_ARGS>
```

### CLI Arguments

| Argument             | Default    | Description                                                                                   |
| :------------------- | :--------- | :---------------------------------------------------------------------------------------------- |
| `--gui`, `-g`         | `False`    | Launches the GUI. Without it, the simulator runs headless and prints results to the console.     |
| `--player`, `-p`      | *(required)* | Which player to run: `1`-`10` (that group's player in `players/player<N>/`), `e` (the worked example), `i` (invalid-examples demo, cycles), `i1`-`i10` (one specific invalid example). Not needed with `--sandbox`. |
| `--scenario`, `-i`    | `N/A`      | Path to a scenario JSON file (room + inventory + weights). If omitted, one is generated.         |
| `--export-scenario`, `-e` | `N/A`  | Save a freshly generated scenario to this path. Mutually exclusive with `--scenario`.            |
| `--seed`, `-s`        | `<random>` | Seed for the random number generator (also used for scenario generation).                        |
| `--difficulty`        | `generous` | `generous` or `scarce` -- how constrained a generated scenario's inventory is.                   |
| `--debug`, `-d`       | `False`    | Show extra debug info (piece lengths, etc.) in the GUI.                                          |
| `--sandbox`, `-x`     | `False`    | View a scenario's room/inventory without running a player. Implies `--gui`.                      |
| `--cpu-limit`         | `300`      | CPU seconds a player may use per run. Over it, the run is stopped and scores 0. See "Time limit and logs". |
| `--log-dir`           | `logs/`    | Where each run's log file is written. See "Time limit and logs".                                 |

### Usage examples

```bash
# group 3's player, headless, on a generated scenario
uv run main.py --player 3

# view the bundled example scenario in the GUI without a player
uv run main.py --sandbox --scenario scenarios/simple_rectangle.json

# run the worked example strategy with the GUI
uv run main.py --gui --player e --scenario scenarios/simple_rectangle.json

# step through examples of INVALID enclosures: click "Build enclosure" repeatedly,
# each click shows the next mistake and why the simulator rejects it
uv run main.py --gui --player i --scenario scenarios/invalid_examples.json

# ...or jump straight to one specific mistake (i1 ... i10, table below)
uv run main.py --gui --player i6 --scenario scenarios/invalid_examples.json

# generate a scenario and save it for reuse
uv run main.py --seed 123 --export-scenario scenarios/my_scenario.json

# a scarcer, more constrained generated scenario
uv run main.py --gui --player e --difficulty scarce
```

## Running the tournament

One command runs every player on every tournament scenario (`tournament/scenarios/`) and records each result in a CSV file:

```bash
uv run python -m tournament.run_tournament
```

To try only your own player (plus the example for comparison), name it, and add `--all` to also use everyone's scenarios:

```bash
uv run python -m tournament.run_tournament --players 3 e
```

Each run is its own process, so a player that crashes or runs out of time only costs its own row. Results go to a new `logs/tournament-<time>/` folder: `results.csv` (one row per run: player, scenario, status, score, area, perimeter, CPU seconds, ...), `summary.csv` (totals per player) and the output of every run. `--jobs N` sets how many run at once (default: your cores minus one); `--resume --out <folder>` carries on an interrupted run; `--dry-run` lists the runs without running them. Run `uv run python -m tournament.run_tournament --help` for the rest.

## Time limit and logs

**CPU time limit.** A player gets 300 seconds of CPU time per run
(`--cpu-limit SECONDS` changes it) for constructing the player *and*
`build_enclosure()` together. CPU time is the processor time the program
actually uses, not time on the clock: waiting costs nothing, and threads busy
at once add up. A player that goes over is stopped where it is and scores **0**
-- the same as returning no enclosure -- with a message such as
`Time limit exceeded: stopped after 300.0s of CPU time (limit 300s)`. Every run
prints how much it used (`player CPU time: 0.84s (limit 300s)`), and the GUI
shows it too. In the GUI each click on "Build enclosure" is its own run with a
fresh budget.

What the limit does not cover, so you know where it stops:
- Only the simulator's own process is counted. CPU used by processes a player
  starts (`multiprocessing`, `joblib`) is not.
- It interrupts Python code. One native call that itself runs past the limit is
  stopped only when it returns.
- On Windows nothing can interrupt a running player; the limit is checked when
  it returns, so an overrun is still reported, but a player that never returns
  would hang the run.

**Logs.** Everything a run prints is also written to a log file, in addition to
the terminal, whose output is unchanged. That includes a player's own
`print()`s and any error messages. Each run gets its own file in `logs/`:
`logs/20261006-162344_1_simple_rectangle_seed1.log` is the time, player,
scenario and seed. A log starts with a header (the command, player, scenario,
seed, CPU limit, Python version) and ends with a footer (how the run ended and
how long it took), plus the traceback if it crashed. The terminal says where
the log went (`logging to ...`). `--log-dir` picks another folder; `logs/` is
git-ignored.

## The GUI

- The left view shows the room and the enclosure that was built. **Hover a
  joint** (a corner where two walls meet) to see its coordinates and which
  connector is there; for a loop that doesn't close, the end of the last wall
  is hoverable too. Coordinates are in room units, as a player would compute
  them (the view draws y increasing downward).
- A wall, gate or connector the inventory **can't supply** is not drawn as a
  real piece: it appears as a translucent ghost with a label (e.g. `wall 13:
  none left in inventory`), the panel says `Walls placed (3 of 4)` and marks
  it, and the hover tooltip for a ghosted connector says `not in inventory`.
  If a construction needs two 13-unit walls and only one is in stock, the
  second is the ghost.
- The right panel shows the room size, a small **scrollable inventory box**, the
  enclosure's area and circumference, each wall/gate placed, and the **score calculation**: baseline, area term,
  perimeter term and second-gate bonus, added up to the score. An invalid
  enclosure shows the -1000 penalty instead; no solution shows why 0 is the
  safe choice. The terms come from `explain_score` in `src/enclosure.py`, the
  same function that produces the real score. **A negative score is shown in red,
  on a red band.**
- The inventory box lists every wall length, gate length and connector type, with
  how many there are. Once an enclosure is built, the rows it **uses are
  highlighted** (`using 2`), and the box scrolls to the first one. A row the
  enclosure wants more of than the inventory has (or an item that isn't in the
  inventory at all) is red, with how many it is short.
- The window is sized to fit your screen and everything is laid out from its
  actual size, so it can be resized. On a screen too short for the whole panel,
  the panel scrolls with the mouse wheel instead of cutting anything off.

## The rules, as implemented

- Walls/gates are integer-length pieces, minimum length 5 units.
- Connectors come in three types -- `straight` (180°), `right` (90° or,
  flipped, 270°), `diagonal` (135° or, flipped, 225°) -- and are placed one
  per vertex of the closed loop.
- A **linear face** is a maximal run of pieces joined only by `straight`
  connectors; every face must be at most 30 units long.
- The enclosure must close (the last piece connects back to the first,
  within a small floating-point tolerance) and must be a simple
  (non-self-intersecting) polygon.
- The enclosure must fit entirely within the room polygon. `Room.
  contains_enclosure(polygon)` is exposed to players so they can check this
  themselves.
- The enclosure needs at least one gate.
- Scoring: 1000 base points if valid, `+A * area`, `+C * perimeter` (`C` is
  negative), `+G` if a second gate sits on a different face than the first.
  An invalid construction scores -1000. No solution at all scores 0.

See `src/enclosure.py` for the full validation/scoring implementation and
`tests/test_enclosure.py` for worked examples of each rule.

## Scenario files

A scenario (`scenarios/*.json`) bundles everything a player is handed up
front:

```json
{
  "room": [[0, 0], [40, 0], [40, 30], [0, 30]],
  "inventory": {
    "walls": {"5": 6, "8": 4},
    "gates": {"6": 2},
    "connectors": {"straight": 12, "right": 12, "diagonal": 8}
  },
  "weights": {"A": 2.0, "C": -3.0, "G": 100.0}
}
```

Two example scenarios are included: `scenarios/simple_rectangle.json` (a
plain rectangular room, generous inventory -- good for quick sanity checks)
and `scenarios/l_shaped_room.json` (a non-convex room, scarcer inventory).
If no `--scenario` is given, one is generated from `--seed` using a
star-shaped random polygon generator (`src/room.py:generate_room`), which is
guaranteed to produce a simple polygon and is often non-convex.

## Writing a player

Groups put their player in `players/player<N>/player.py`, as `class Player<N>`
(each folder already holds an empty one). Nothing else needs registering: `--player 3`
finds `Player3` by itself. [PLACEMENT_GUIDE.md](PLACEMENT_GUIDE.md) is the place to
start: how a `Construction` becomes coordinates, and how to find somewhere in the room
to put it.

```python
from players.player0 import Player
from src.enclosure import Construction


class Player3(Player):
    def build_enclosure(self) -> Construction | None:
        # self.room, self.inventory, self.weights are available here.
        # Return None if no valid enclosure can be built.
        ...
```

A `Construction` is a closed loop: a starting point + heading for the first
piece, a list of `Piece`s (wall/gate + length), and a list of `Connector`s
(one per vertex, `connectors[i]` being the turn at vertex `i`). See
`src/pieces.py` for the `Piece`/`Connector` data model, `players/example_player.py`
for a complete working strategy, and `SIMULATOR_GUIDE.md` for a much deeper
walkthrough of every module and exactly how a `Construction` turns into a score.

**Submitting.** Work only inside `players/player<N>/` (and, if you like, your own test
scenarios in `scenarios/players/player<N>/`); a pull request that changes anything else
fails the automated check. Before opening it, run:

```bash
uv run ruff format players/player<N>
uv run python -m scripts.check_submission <N>
```

It must say `CHECK PASSED`. A player that crashes fails the check; a timeout or an
invalid enclosure is shown as a warning. Don't add dependencies (they're shared by all
groups): open an issue instead. `MAINTAINERS.md` describes how pull requests are reviewed.

- `demos/invalid_examples.py` (`--player i`) -- not a strategy: a
  teaching aid that returns a different deliberately *invalid* enclosure on
  each call (doesn't close, outside the room, walls cross, ...), each
  breaking exactly one rule. Read it to see how to build a `Construction`
  and what each mistake looks like. Use it with
  `scenarios/invalid_examples.json`.

Each kind of invalid enclosure has its own name, so it can be shown directly
(add `--gui` to see it drawn; without it the reason is printed):

| `--player` | What's wrong                                                          |
| :--------- | :-------------------------------------------------------------------- |
| `i1`       | wrong number of connectors: the attempt is drawn, the vertex with no connector is marked |
| `i2`       | doesn't close: last wall is too short, leaving a gap                  |
| `i3`       | closing joint has the wrong angle (walls meet, the angle doesn't)     |
| `i4`       | needs more of a piece than the inventory has                          |
| `i5`       | no gate                                                               |
| `i6`       | walls cross each other                                                |
| `i7`       | entirely outside the room                                             |
| `i8`       | sticks out through the room's right-hand wall                         |
| `i9`       | every corner inside the room, but one wall cuts across the outside    |
| `i10`      | a straight run of walls is 32 units long (limit is 30)                |

A player can set `self.note` to a one-line string; the GUI shows it in the
result panel and the console prints it.


## Tests

```bash
uv run python -m tests.test_enclosure          # rules, geometry, scoring
uv run python -m tests.test_players            # players' connector guard
uv run python -m tests.test_invalid_examples   # the invalid-enclosure demos
uv run python -m tests.test_limits             # the CPU time limit
uv run python -m tests.test_submissions        # group players, the submission checker
uv run python -m tests.test_placement_guide    # the docs: PLACEMENT_GUIDE.md's code runs, files mentioned exist
uv run python -m tests.test_runlog             # run logs, and the main.py command line
uv run python -m tests.test_gui_layout         # GUI layout (needs a display)
uv run python -m tests.test_gui_inventory      # the inventory box, red negative score (needs a display)
```

## Code Quality and Formatting

```bash
uv run ruff format --check
uv run ruff check
```
