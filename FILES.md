# What every file is for

A map of the repository, for whoever maintains it. (`MAINTAINERS.md` covers how to
review and merge pull requests; this is just *what is where*.)

## How a run flows

`uv run main.py --player 3 --scenario scenarios/l_shaped_room.json`:

1. **`main.py`** reads the command line (`src/args.py`), opens a log file
   (`src/runlog.py`), and creates a `Game` (`src/game.py`).
2. **`Game`** loads the scenario (`src/scenario.py`: room + inventory + weights).
3. When it plays, it looks the player up (`players/registry.py` finds
   `players/player3/player.py`), builds it and calls `build_enclosure()`, all inside the CPU
   time limit (`src/limits.py`).
4. The `Construction` it returns is checked and scored by **`src/enclosure.py`**, which is
   the one place that decides what is legal. With `--gui`, `Game` also draws it.

## Where do I change...

| I want to change | Look in |
| :-- | :-- |
| a rule (min wall length, max face length, tolerance, scoring constants) | `src/constants.py`, then `src/enclosure.py` if the logic changes |
| how an enclosure is checked or scored | `src/enclosure.py` |
| what a player is given or must return | `players/player0.py` (and tell the groups: it can break all ten) |
| the GUI | `src/game.py` (colors and sizes in `src/constants.py`) |
| the CPU time limit | `CPU_TIME_LIMIT_SECONDS` in `src/constants.py`, or `--cpu-limit` per run |
| what a log file contains | `src/runlog.py` and `main.py` (the header) |
| how many groups there are | `NUM_GROUPS` in `players/registry.py`, then add the folders |
| the checks run on a pull request | `scripts/check_submission.py`, `.github/workflows/ci.yml` |
| a test scenario | add a JSON file to `scenarios/` |

## Top level

| File | What it is |
| :-- | :-- |
| `main.py` | The entry point. Parses arguments, starts the run log, creates the `Game`. |
| `pyproject.toml` | Project name, Python version, dependencies (just `shapely`; `ruff` for development). |
| `uv.lock` | The exact versions of those dependencies, so everyone (and CI) gets the same ones. |
| `.python-version` | The Python version `uv` uses. |
| `.gitignore` | Keeps `.venv`, caches and `logs/` out of git. |

## Documentation

| File | For whom | What it is |
| :-- | :-- | :-- |
| `README.md` | everyone | How to run the simulator, every option, the rules in brief, how to write and submit a player. |
| `PLACEMENT_GUIDE.md` | students | Getting started with the geometry: coordinates, building shapes, testing whether they fit, searching for a placement. All its code runs (tested). |
| `SIMULATOR_GUIDE.md` | students | A deep walkthrough of every module: the data model, the validator step by step, scoring, the player contract. |
| `MAINTAINERS.md` | you | How the pull-request process is set up, how to review and merge, how to run all the players. |
| `FILES.md` | you | This file. |

## `src/`: the simulator

Nothing here is a group's to change; everything a group's player is judged by lives here.

| File | What it is |
| :-- | :-- |
| `src/enclosure.py` | **The rulebook.** `Construction` (what a player returns), turning it into corner points (`build_geometry`, `walk_points`, `sketch_points`), grouping walls into straight faces, `validate_construction` (every rule, in order, with a reason), `unavailable_items` (what the inventory can't supply), `explain_score` / `score_construction`. |
| `src/pieces.py` | The physical parts: `Piece` (wall or gate + length) and `Connector` (straight / right / diagonal, and flipped), with the angle each makes. |
| `src/inventory.py` | `Inventory`: the walls, gates and connectors available and their counts; `copy`, `take_piece`, `take_connector`; and the random inventory generator. |
| `src/room.py` | `Room`: the footprint polygon, `contains_enclosure` (the check players are given), drawing; and the random room generator. |
| `src/weights.py` | `Weights`: the scoring weights `A` (area), `C` (perimeter) and `G` (second gate). |
| `src/scenario.py` | `Scenario` = room + inventory + weights; reading and writing the JSON files in `scenarios/`; generating a random one. |
| `src/game.py` | `Game`: the tkinter window (room view, hover tooltips, result panel, score calculation), and `play()`, which builds the player under the CPU limit, validates and scores what it returns. The biggest file. |
| `src/args.py` | The command line: `--player`, `--scenario`, `--cpu-limit`, `--log-dir`, `--gui`, ... |
| `src/constants.py` | The tunable numbers: window and colors, the rules, tolerances, score constants, the default CPU limit. |
| `src/limits.py` | The CPU time limit (`CpuLimit`, `PlayerTimeout`). Counts CPU time, not clock time; interrupts a player that overruns. |
| `src/runlog.py` | `RunLog`: copies everything printed to a log file as well as the terminal. |

## `players/`: the players

| File | What it is |
| :-- | :-- |
| `players/player0.py` | The base class `Player` every player subclasses: it stores `room`, `inventory`, `weights`, offers `note` and `lacks_connectors`, and requires `build_enclosure`. This is the contract. |
| `players/registry.py` | Turns `--player <name>` into a class. Group players are found by naming convention (`players/playerN/player.py` defines `PlayerN`), so no group edits this file, and they are loaded only when selected. Also lists the demo and example players. |
| `players/example_player.py` | **The one worked example** (`--player e`): finds a rectangle the inventory can build, then searches for where it fits. |
| `players/player1/player.py` ... `player10/player.py` | One folder per group, each with an empty `PlayerN` for the group to fill in. A group changes only its own folder; the folder name and class name must stay. |

## `demos/`

| File | What it is |
| :-- | :-- |
| `demos/invalid_examples.py` | A teaching aid, not a strategy (`--player i`, or `i1`...`i10` for one example). Returns ten deliberately invalid constructions, one rule broken each: doesn't close, outside the room, walls cross, a wall cutting across a notch, ... Use it with `scenarios/invalid_examples.json`. |

## `scenarios/`

A scenario is a JSON file: `room` (polygon corners), `inventory`, `weights`.

| File | What it is |
| :-- | :-- |
| `scenarios/simple_rectangle.json` | A rectangular room with a generous inventory. The easy case. |
| `scenarios/l_shaped_room.json` | A non-convex L-shaped room with a scarcer inventory. |
| `scenarios/limits_tight_room_favors_small_shape.json` | A tight room where only the smallest of many possible rectangles fits; it caught a real bug in the example player's shape ranking. |
| `scenarios/invalid_examples.json` | The room and inventory the invalid-enclosure demo is built around (an L-shaped room, plus the odd pieces its examples need). |
| `tournament/scenarios/1a.json` | A regular 50-sided room (every side 2) with a plentiful inventory, all connector types. Weights A=1, C=-1, G=50. |
| `tournament/scenarios/1b.json` | The same 50-gon with a small but feasible inventory. |
| `tournament/scenarios/2a.json` | A 42 x 30 room whose whole border is a zigzag of length-2 segments, with a plentiful inventory. |
| `tournament/scenarios/2b.json` | The same zipper room with a small but feasible inventory. |
| `tournament/scenarios/3a.json` | A 100 x 100 square with a fixed set of walls, two gates, 10 straight and 12 right connectors, no diagonals. |
| `tournament/scenarios/3b.json` | The same square with every wall turned into a gate of the same length, so no walls. |
| `tournament/scenarios/4a.json` | An hourglass: a wide lower lobe and a smaller upper lobe joined by a 4-wide waist, with sharp corners (about 47 degrees). Plentiful inventory. |
| `tournament/scenarios/4b.json` | The same hourglass with a small but feasible inventory. |
| `tournament/scenarios/5.json` | A 40 x 25 rectangular room turned 36.87 degrees (corners are whole numbers), with a plentiful inventory. Needs the right start heading. |
| `tournament/scenarios/6.json` | A tight L-shaped room (23 x 21) with 11 pieces of different lengths and only 6 right-angle connectors. Exactly one enclosure is valid: the L that fills the room. |
| `scenarios/players/player1/` ... `player10/` | One folder per group for its own test scenarios. Each holds a `.gitkeep` so git keeps the empty folder. |

## `tournament/`

| File | What it is |
| :-- | :-- |
| `tournament/run_tournament.py` | Runs players on scenarios and writes one CSV row per run (player, scenario, score, validity, area, CPU time, ...). Every run is its own process, with a CPU limit and a real-time limit. Groups use it to try their own player; you use it for the real tournament. |
| `tournament/scenarios/` | The tournament's rooms, `1a.json` ... `6.json` (listed in the `scenarios/` table below). |

## `scripts/`

| File | What it is |
| :-- | :-- |
| `scripts/check_submission.py` | The check run on every group's pull request (and by groups before opening one): only one group's folders changed, the player loads and is formatted, then runs on every scenario. `uv run python -m scripts.check_submission N`. |
| `scripts/__init__.py` | Empty; lets `python -m scripts.check_submission` work. |

## `tests/`

Run any one with `uv run python -m tests.<name>`. None of them runs a group's player: they use
stand-ins, so a broken group player can't turn the simulator's tests red.

| File | What it covers |
| :-- | :-- |
| `tests/test_enclosure.py` | The rules: closing, inventory, gates, containment, straight-face length, scoring and its breakdown, the lenient drawing walk. |
| `tests/test_players.py` | The example player's connector guard. |
| `tests/test_invalid_examples.py` | Each invalid demo fails for exactly the reason it claims. |
| `tests/test_limits.py` | The CPU limit: interrupts a runaway player, ignores waiting, can't be swallowed, resets per run. |
| `tests/test_runlog.py` | Log files: mirrors stdout/stderr, header and footer, crashes, unique names, and `main.py` end to end. |
| `tests/test_submissions.py` | Finding group players, the ownership rules, the submission checker. |
| `tests/test_gui_layout.py` | The GUI: no overlapping text, nothing clipped at several window sizes, hover tooltips, the score panel. Needs a display. |
| `tests/test_gui_inventory.py` | The GUI's inventory box (every item listed, used ones highlighted, scrolling), the red negative score, and the panel scrolling on a short screen. Needs a display. |
| `tests/test_tournament.py` | The tournament runner: which scenarios and players it picks, the CSV rows, crashes and stuck players, resuming, the summary. Runs only the example player. |
| `tests/test_placement_guide.py` | Runs every code block in `PLACEMENT_GUIDE.md` and checks the numbers it claims; checks the docs only name files that exist, and that this file lists every file. |
| `tests/__init__.py` | Empty; makes `tests` a package. |

## `.github/`

| File | What it is |
| :-- | :-- |
| `.github/workflows/ci.yml` | CI. Job 1 formats, lints and tests the simulator. Job 2 runs the submission check on a pull request (skipped if it has the `maintainer` label). |
| `.github/pull_request_template.md` | The description every pull request starts with, with a checklist. |
| `.github/CODEOWNERS` | A template (all comments for now) for who is asked to review each group's folder. Fill it in. |

## Generated, not tracked by git

| Path | What it is |
| :-- | :-- |
| `logs/` | One log file per run: the command, the output, how it ended. |
| `.venv/` | The Python environment `uv sync` creates. |
| `.idea/` | PyCharm's project settings (kept out of git). |
| `__pycache__/`, `.ruff_cache/` | Python and `ruff` caches. |
