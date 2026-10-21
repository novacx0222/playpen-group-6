"""Compare Player6 with local 45-degree refinement disabled/enabled.

Run from the repository root: python -m players.player6.benchmark
Only files under players/player6 are written by default. Both variants use
the same overall CPU limit and fresh copies of each scenario. Stage caps are
optional; by default Player6 dynamically allocates the simulator CPU budget.
"""

import argparse
import csv
import json
import logging
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import shapely
from shapely.geometry import box

from players.player6.local_search import LocalSearchStats
from players.player6.player import Player6
from src.enclosure import score_construction, validate_construction
from src.inventory import Inventory
from src.limits import CpuLimit, PlayerTimeout
from src.pieces import ConnectorType as CT
from src.room import Room
from src.scenario import Scenario, generate_scenario, read_scenario
from src.weights import Weights

ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)


def scenarios():
    """Small exact cases plus existing difficult rooms and fixed scarce seeds."""
    weights = Weights(A=2, C=-3, G=100)
    cases = {
        "tiny_room": Scenario(
            Room(box(0, 0, 5, 5)),
            Inventory(walls={5: 3}, gates={5: 1}, connectors={CT.RIGHT: 4}),
            weights,
        ),
        "insufficient_stock": Scenario(
            Room(box(0, 0, 20, 20)),
            Inventory(
                walls={5: 2}, gates={5: 1}, connectors={CT.RIGHT: 4, CT.DIAGONAL: 4}
            ),
            weights,
        ),
        "shared_lengths": Scenario(
            Room(box(0, 0, 20, 20)),
            Inventory(
                walls={5: 8},
                gates={5: 1},
                connectors={CT.RIGHT: 8, CT.STRAIGHT: 8, CT.DIAGONAL: 4},
            ),
            weights,
        ),
        "diagonal_room": Scenario(
            Room(box(0, 0, 15, 15)),
            Inventory(
                walls={10: 3, 5: 2},
                gates={10: 1},
                connectors={CT.RIGHT: 4, CT.DIAGONAL: 4},
            ),
            weights,
        ),
        "scarce_seed1": generate_scenario(1, difficulty="scarce"),
        "scarce_seed2": generate_scenario(2, difficulty="scarce"),
    }
    for name in ("narrow", "spiral", "diagonal_hexagon"):
        cases[name] = read_scenario(
            str(ROOT / "scenarios" / "players" / "player6" / f"{name}.json")
        )
    return cases


class MeasuredPlayer(Player6):
    grid_cpu_seconds = 0.0

    def full_checks(self):
        started = time.process_time()
        try:
            super().full_checks()
        finally:
            self.grid_cpu_seconds = time.process_time() - started


def run_case(
    name,
    scenario,
    variant,
    *,
    search_seconds=45.0,
    grid_seconds=None,
    beam_seconds=None,
    local_seconds=None,
    repeat=1,
):
    player = None
    construction = None
    row = {
        "case": name,
        "repeat": repeat,
        "variant": variant,
        "status": "none",
        "score": 0.0,
        "area": 0.0,
        "perimeter": 0.0,
        "gate_count": 0,
        "second_gate_bonus": False,
        "reason": "no solution",
    }
    cpu = CpuLimit(search_seconds)
    started = time.perf_counter()
    try:
        with cpu:
            player = MeasuredPlayer(
                scenario.room.copy(),
                scenario.inventory.copy(),
                scenario.weights,
                cpu_limit=search_seconds,
                stage_limits={
                    "grid": grid_seconds,
                    "beam": beam_seconds,
                    "local": 0.0 if variant == "baseline" else local_seconds,
                },
            )
            construction = player.build_enclosure()
    except PlayerTimeout as error:
        row.update(status="timeout", reason=str(error))
    except Exception as error:
        row.update(status="error", reason=f"{type(error).__name__}: {error}")
        LOGGER.exception("Benchmark failed for %s (%s)", name, variant)
    row["cpu_seconds"] = cpu.used
    row["wall_seconds"] = time.perf_counter() - started
    if construction is not None and row["status"] not in ("timeout", "error"):
        result = validate_construction(construction, scenario.room, scenario.inventory)
        row.update(
            status="valid" if result.valid else "invalid",
            reason=result.reason,
            score=score_construction(result, scenario.weights),
            area=result.area,
            perimeter=result.perimeter,
            gate_count=result.gate_count,
            second_gate_bonus=result.second_gate_bonus,
        )
    row["inventory_unchanged"] = (
        player is None or player.inventory == scenario.inventory
    )
    row["grid_cpu_seconds"] = player.grid_cpu_seconds if player is not None else 0.0
    row["beam_skipped"] = getattr(player, "beam_skipped", False)
    row["beam_score_upper_bound"] = getattr(player, "beam_upper_bound", None)
    row["phase_seconds"] = dict(getattr(player, "phase_seconds", {}))
    history = getattr(player, "beam_history", [])
    row["beam_history"] = list(history)
    row["beam_improvement_count"] = len(history)
    staircase_stats = getattr(player, "inscribed_diagonal_stats", LocalSearchStats())
    row["staircase_stats"] = asdict(staircase_stats)
    stats = (
        asdict(player.local_stats)
        if player is not None and hasattr(player, "local_stats")
        else {}
    )
    for key in ("attempted", "feasible", "accepted", "cpu_seconds"):
        row[f"local_{key}"] = stats.get(key, 0)
    budget = getattr(player, "budget_info", None)
    for key in (
        "cpu_limit",
        "available_seconds",
        "usable_seconds",
        "safety_seconds",
        "grid_seconds",
        "beam_seconds",
        "local_seconds",
        "source",
    ):
        row[f"budget_{key}"] = getattr(budget, key, None)
    return row


def write_report(path, rows, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": config,
        "python": platform.python_version(),
        "shapely": shapely.__version__,
        "results": rows,
    }
    path.write_text(json.dumps(report, indent=2) + "\n")
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Player6 local 45-degree benchmark",
        "",
        "Baseline disables local refinement; local45 enables it. Both have the same overall CPU limit.",
        "Null stage caps mean automatic proportional allocation; unused stage time is reclaimed.",
        "CPU time includes player construction and search; final verification is outside that budget.",
        "Deadline-limited runs can vary. These cases do not establish a global improvement guarantee.",
        "",
        f"Budgets (CPU seconds): {json.dumps(config, sort_keys=True)}",
        "",
        "| Case | Repeat | Baseline | Local45 | Score delta | CPU baseline/local | Local moves |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        if row["variant"] != "baseline":
            continue
        other = next(
            r
            for r in rows
            if r["case"] == row["case"]
            and r["repeat"] == row["repeat"]
            and r["variant"] == "local45"
        )
        lines.append(
            f"| {row['case']} | {row['repeat']} | {row['score']:.2f} ({row['status']}) | "
            f"{other['score']:.2f} ({other['status']}) | {other['score'] - row['score']:+.2f} | "
            f"{row['cpu_seconds']:.3f} / {other['cpu_seconds']:.3f} | {other['local_accepted']} |"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    cases = scenarios()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=cases, dest="cases")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--search-seconds", "--cpu-limit", type=float, default=45.0)
    parser.add_argument("--grid-seconds", type=float, default=None)
    parser.add_argument("--beam-seconds", type=float, default=None)
    parser.add_argument("--local-seconds", type=float, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("benchmark_results.json"),
    )
    args = parser.parse_args(argv)
    if (
        args.repeats < 1
        or not math.isfinite(args.search_seconds)
        or args.search_seconds <= 0
        or any(
            value is not None and (not math.isfinite(value) or value < 0)
            for value in (args.grid_seconds, args.beam_seconds, args.local_seconds)
        )
    ):
        parser.error(
            "repeats/search budget must be positive; other budgets must be nonnegative"
        )
    config = {
        key: getattr(args, key)
        for key in ("search_seconds", "grid_seconds", "beam_seconds", "local_seconds")
    }
    rows = []
    for name in args.cases or cases:
        for repeat in range(1, args.repeats + 1):
            for variant in ("baseline", "local45"):
                row = run_case(name, cases[name], variant, repeat=repeat, **config)
                rows.append(row)
                print(
                    f"{name:20s} {variant:8s} {row['status']:7s} "
                    f"score={row['score']:.2f} cpu={row['cpu_seconds']:.3f}s "
                    f"local_moves={row['local_accepted']}",
                    flush=True,
                )
    write_report(args.output, rows, config)
    print(
        f"Reports: {args.output}, {args.output.with_suffix('.csv')}, {args.output.with_suffix('.md')}"
    )
    return int(
        any(
            row["status"] in ("invalid", "error", "timeout")
            or not row["inventory_unchanged"]
            for row in rows
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
