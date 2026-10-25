"""Run players against scenarios and write one CSV row per run.

    uv run python -m tournament.run_tournament                  # everyone, on tournament/scenarios/
    uv run python -m tournament.run_tournament --players 3 e    # only group 3 and the example
    uv run python -m tournament.run_tournament --all            # also the shared and the groups' own scenarios

Every (player, scenario) pair runs in its own process, so a player that crashes,
hangs or runs out of CPU time costs only its own row. Rows are written as they
finish, so a long run that is interrupted keeps what it has (`--resume` carries on).
Everything goes in one folder (default: logs/tournament-<time>/):

    results.csv   one row per run
    summary.csv   one row per player and scenario set (totals)
    logs/         what each run printed, one file per run
"""

import csv
import hashlib
import multiprocessing
import os
import re
import sys
import time
import traceback
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import connection
from pathlib import Path

import src.constants as c
from players.registry import GROUP_NAMES

ROOT = Path(__file__).resolve().parent.parent
TOURNAMENT_DIR = ROOT / "tournament" / "scenarios"
SHARED_DIR = ROOT / "scenarios"
GROUP_SCENARIOS_DIR = ROOT / "scenarios" / "players"

# the teaching room for the invalid-enclosure demo; never part of a tournament
EXCLUDED = {"invalid_examples.json"}

SEED = 0  # runs are reproducible; the scenario file fixes everything else

COLUMNS = [
    "set",  # tournament / shared / group<N>
    "player",  # example / player<N>
    "scenario",  # path of the scenario file, from the project folder
    "status",  # ok / invalid / no_solution / cpu_timeout / wall_timeout / crash
    "valid",
    "score",
    "area",
    "perimeter",
    "gate_count",
    "second_gate_bonus",
    "cpu_seconds",  # CPU time the player used (building it + build_enclosure)
    "wall_seconds",  # real time the run took, including start-up
    "reason",  # why it is invalid, or what went wrong
]


# ------------------------------------------------------------------ choosing


def player_label(name: str) -> str:
    return "example" if name == "e" else f"player{name}"


def set_of(path: Path) -> str:
    """Which collection a scenario belongs to, for the `set` column."""
    path = path.resolve()
    if path.is_relative_to(TOURNAMENT_DIR):
        return "tournament"
    if path.is_relative_to(GROUP_SCENARIOS_DIR):
        group = path.relative_to(GROUP_SCENARIOS_DIR).parts[0]
        return group.replace("player", "group", 1)
    if path.parent == SHARED_DIR:
        return "shared"
    return "other"


def relative(path: Path) -> str:
    path = path.resolve()
    return (path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).as_posix()


def _natural(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def _folder_files(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.glob("*.json") if p.name not in EXCLUDED), key=_natural
    )


def collect_scenarios(selected: list[Path] | None, everything: bool) -> list[Path]:
    """The scenario files to run, each once. A folder means its .json files;
    a file named on its own is always included."""
    paths: list[Path] = []
    for entry in selected or [TOURNAMENT_DIR]:
        if entry.is_dir():
            paths += _folder_files(entry)
        elif entry.is_file():
            paths.append(entry)
        else:
            raise SystemExit(f"no such scenario file or folder: {entry}")
    if everything:
        paths += _folder_files(TOURNAMENT_DIR) + _folder_files(SHARED_DIR)
        for folder in sorted(GROUP_SCENARIOS_DIR.glob("player*"), key=_natural):
            paths += _folder_files(folder)

    unique, seen = [], set()
    for path in paths:
        # the same room copied into two folders only counts once
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(path)
    return unique


def collect_players(selected: list[str] | None) -> list[str]:
    allowed = ["e"] + GROUP_NAMES
    if not selected:
        return allowed
    unknown = [p for p in selected if p not in allowed]
    if unknown:
        raise SystemExit(
            f"unknown player(s) {', '.join(unknown)}; choose from: {', '.join(allowed)}"
        )
    return list(dict.fromkeys(selected))


# ------------------------------------------------------------------- one run


def _summarize(game) -> dict:
    """The CSV fields for a run that finished (the player may still have failed)."""
    row = {"cpu_seconds": game.cpu_used, "score": game.score}
    if game.timeout_message:
        row.update(status="cpu_timeout", valid=False, reason=game.timeout_message)
    elif game.construction is None:
        row.update(status="no_solution", valid=False, reason="player returned None")
    elif game.result.valid:
        result = game.result
        row.update(
            status="ok",
            valid=True,
            area=result.area,
            perimeter=result.perimeter,
            gate_count=result.gate_count,
            second_gate_bonus=result.second_gate_bonus,
        )
    else:
        row.update(status="invalid", valid=False, reason=game.result.reason)
    return row


def _worker(conn, player: str, scenario: str, cpu_limit: float, log_path: str):
    """Runs in its own process: play one game, send the summary back."""
    from src.args import Args
    from src.game import Game

    started = time.process_time()
    with open(log_path, "w", buffering=1) as log:
        sys.stdout = sys.stderr = log
        try:
            game = Game(
                Args(
                    gui=False,
                    player=player,
                    scenario_path=scenario,
                    seed=SEED,
                    difficulty="generous",
                    export_scenario=None,
                    debug=False,
                    sandbox=False,
                    cpu_limit=cpu_limit,
                )
            )
            row = _summarize(game)
        except BaseException as e:  # noqa: BLE001 -- anything the player or scenario does
            traceback.print_exc()
            message = " ".join(f"{type(e).__name__}: {e}".split())
            row = {
                "status": "crash",
                "valid": False,
                "score": 0.0,
                "cpu_seconds": time.process_time() - started,
                "reason": message,
            }
        conn.send(row)


@dataclass
class Job:
    player: str
    scenario: Path

    @property
    def key(self) -> tuple[str, str]:
        return player_label(self.player), relative(self.scenario)

    def log_name(self) -> str:
        stem = f"{player_label(self.player)}__{set_of(self.scenario)}__{self.scenario.stem}"
        return re.sub(r"[^A-Za-z0-9_.-]", "_", stem) + ".log"


@dataclass
class Running:
    job: Job
    process: multiprocessing.Process
    conn: connection.Connection
    started: float


def _finish(
    running: Running, row: dict | None, killed: bool, wall_limit: float
) -> dict:
    """Turn what a process sent back (or didn't) into the final CSV row."""
    process = running.process
    process.join(5)
    wall = time.monotonic() - running.started
    if killed:
        row = {
            "status": "wall_timeout",
            "valid": False,
            "score": 0.0,
            "reason": f"killed after {wall_limit:g}s of real time",
        }
    elif row is None:
        row = {
            "status": "crash",
            "valid": False,
            "score": 0.0,
            "reason": f"process died without a result (exit code {process.exitcode})",
        }
    return {
        "set": set_of(running.job.scenario),
        "player": player_label(running.job.player),
        "scenario": relative(running.job.scenario),
        "wall_seconds": wall,
        **row,
    }


def run_jobs(jobs, jobs_at_once, cpu_limit, wall_limit, log_dir: Path, on_row):
    """Run every job, up to `jobs_at_once` at a time; call `on_row(row)` as each ends."""
    ctx = multiprocessing.get_context("spawn")  # same on every OS; no shared state
    pending = list(jobs)
    running: list[Running] = []
    while pending or running:
        while pending and len(running) < jobs_at_once:
            job = pending.pop(0)
            receiver, sender = ctx.Pipe(duplex=False)
            process = ctx.Process(
                target=_worker,
                args=(
                    sender,
                    job.player,
                    str(job.scenario),
                    cpu_limit,
                    str(log_dir / job.log_name()),
                ),
            )
            process.start()
            sender.close()
            running.append(Running(job, process, receiver, time.monotonic()))

        ready = connection.wait([r.conn for r in running], timeout=0.25)
        for r in list(running):
            row, killed = None, False
            if time.monotonic() - r.started > wall_limit:
                r.process.kill()  # over the limit counts, even if it just finished
                killed = True
            elif r.conn in ready:
                try:
                    row = r.conn.recv()
                except EOFError:
                    pass  # the process ended without sending anything
            else:
                continue
            running.remove(r)
            r.conn.close()
            on_row(_finish(r, row, killed, wall_limit))


# ------------------------------------------------------------------- results


def _cell(value):
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return "" if value is None else value


class ResultsFile:
    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        if path.exists():
            with open(path, newline="") as f:
                self.rows = list(csv.DictReader(f))
        else:
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(COLUMNS)

    def done(self) -> set[tuple[str, str]]:
        return {(r["player"], r["scenario"]) for r in self.rows}

    def add(self, row: dict):
        cells = {k: _cell(row.get(k)) for k in COLUMNS}
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, COLUMNS).writerow(cells)  # a row is on disk at once
        self.rows.append(cells)


SUMMARY_COLUMNS = [
    "set",
    "player",
    "runs",
    "valid_runs",
    "total_score",
    "mean_score",
    "failures",  # no solution, invalid, timeouts and crashes
    "timeouts",
    "crashes",
    "total_cpu_seconds",
    "max_cpu_seconds",
]


def summarize(rows: list[dict]) -> list[dict]:
    """Totals per scenario set and player, best total score first."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["set"], row["player"]), []).append(row)
    summary = []
    for (set_name, player), group in groups.items():
        scores = [float(r["score"] or 0) for r in group]
        cpu = [float(r["cpu_seconds"]) for r in group if r["cpu_seconds"] != ""]
        summary.append(
            {
                "set": set_name,
                "player": player,
                "runs": len(group),
                "valid_runs": sum(r["status"] == "ok" for r in group),
                "total_score": sum(scores),
                "mean_score": sum(scores) / len(scores),
                "failures": sum(r["status"] != "ok" for r in group),
                "timeouts": sum(
                    r["status"] in ("cpu_timeout", "wall_timeout") for r in group
                ),
                "crashes": sum(r["status"] == "crash" for r in group),
                "total_cpu_seconds": sum(cpu),
                "max_cpu_seconds": max(cpu, default=0.0),
            }
        )
    summary.sort(key=lambda s: (s["set"], -s["total_score"], s["player"]))
    return summary


def write_summary(path: Path, summary: list[dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summary:
            writer.writerow({k: _cell(row[k]) for k in SUMMARY_COLUMNS})


def print_summary(summary: list[dict]):
    print()
    for set_name in dict.fromkeys(s["set"] for s in summary):
        print(f"scenario set: {set_name}")
        print(
            f"  {'player':<10}{'runs':>5}{'valid':>7}{'total':>11}{'mean':>10}{'max cpu':>10}"
        )
        for s in (s for s in summary if s["set"] == set_name):
            print(
                f"  {s['player']:<10}{s['runs']:>5}{s['valid_runs']:>7}"
                f"{s['total_score']:>11.1f}{s['mean_score']:>10.1f}{s['max_cpu_seconds']:>9.2f}s"
            )
        print()


# ---------------------------------------------------------------------- main


def positive(text: str) -> float:
    value = float(text)
    if not value > 0:
        raise ValueError("must be greater than 0")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(
        description="Run players on scenarios and record every result in a CSV file."
    )
    parser.add_argument(
        "--players",
        nargs="+",
        metavar="PLAYER",
        help=f"which players: e (the example) and/or 1..{len(GROUP_NAMES)}. Default: all",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="scenario files or folders. Default: tournament/scenarios/",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="also run the shared scenarios and every group's own scenarios",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="runs at once; keep it below your number of cores. Default: %(default)s",
    )
    parser.add_argument(
        "--cpu-limit",
        type=positive,
        default=c.CPU_TIME_LIMIT_SECONDS,
        metavar="SECONDS",
        help="CPU time a player may use per run. Default: %(default)s",
    )
    parser.add_argument(
        "--wall-limit",
        type=positive,
        metavar="SECONDS",
        help="real time after which a run is killed, in case a player is stuck "
        "where the CPU limit cannot reach. Default: twice the CPU limit plus 60",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="folder for results.csv, summary.csv and logs/. "
        "Default: logs/tournament-<time>/",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="with --out: skip the runs already in its results.csv and carry on",
    )
    parser.add_argument("--dry-run", action="store_true", help="list the runs and stop")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.resume and not args.out:
        parser.error("--resume needs --out (the folder to carry on in)")

    players = collect_players(args.players)
    scenarios = collect_scenarios(args.scenarios, args.all)
    jobs = [Job(p, s) for s in scenarios for p in players]
    wall_limit = args.wall_limit or 2 * args.cpu_limit + 60

    if args.dry_run:
        for job in jobs:
            print(f"{job.key[0]:<10} {job.key[1]}")
        print(f"{len(jobs)} runs ({len(players)} players x {len(scenarios)} scenarios)")
        return 0

    out = (
        args.out
        or ROOT / "logs" / f"tournament-{datetime.now().astimezone():%Y%m%d-%H%M%S}"
    )
    (out / "logs").mkdir(parents=True, exist_ok=True)
    results = ResultsFile(out / "results.csv")
    if results.rows and not args.resume:
        raise SystemExit(f"{out / 'results.csv'} already has results; use --resume")
    done = results.done()
    todo = [j for j in jobs if j.key not in done]
    print(
        f"{len(players)} players x {len(scenarios)} scenarios = {len(jobs)} runs"
        f" ({len(jobs) - len(todo)} already done), {args.jobs} at a time, "
        f"cpu limit {args.cpu_limit:g}s"
    )
    print(f"results: {results.path}")

    finished = 0

    def on_row(row: dict):
        nonlocal finished
        finished += 1
        results.add(row)
        cpu = (
            f"{float(row['cpu_seconds']):.2f}s"
            if row.get("cpu_seconds") is not None
            else "-"
        )
        print(
            f"[{finished:>{len(str(len(todo)))}}/{len(todo)}] {row['player']:<9}"
            f" {row['scenario']:<58} {row['status']:<12}"
            f" score={float(row['score']):>8.1f} cpu={cpu}",
            flush=True,
        )

    try:
        run_jobs(todo, args.jobs, args.cpu_limit, wall_limit, out / "logs", on_row)
    except KeyboardInterrupt:
        print("\ninterrupted; rerun with --resume --out", out, "to carry on")
        return 130
    summary = summarize(results.rows)
    write_summary(out / "summary.csv", summary)
    print_summary(summary)
    print(f"wrote {results.path} and {out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
