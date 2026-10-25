"""Benchmark Player5 in fresh processes; run this module with --help for usage."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from src.limits import CpuLimit, PlayerTimeout

ROOT = Path(__file__).resolve().parents[2]


def _positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return seconds


def _worker_count(value):
    count = int(value)
    if not 1 <= count <= 4:
        raise argparse.ArgumentTypeError("must be between 1 and 4")
    return count


def _source_hashes():
    """Pin player and simulator code, excluding tests."""
    files = [ROOT / "players/player0.py", *sorted((ROOT / "src").glob("*.py"))]
    files += sorted((ROOT / "players/player5").glob("*.py"))
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
        if not path.name.startswith("test_")
    }


def _runtime():
    packages = {}
    for package in ("shapely", "numpy"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "process_id": os.getpid(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }


def _construction_dict(construction):
    if construction is None:
        return None
    return {
        "start": construction.start,
        "start_heading": construction.start_heading,
        "pieces": [
            {"piece_type": piece.piece_type.value, "length": piece.length}
            for piece in construction.pieces
        ],
        "connectors": [
            {"connector_type": joint.connector_type.value, "reflex": joint.reflex}
            for joint in construction.connectors
        ],
    }


def _run_scenario(path, budget, cpu_limit):
    """Run inside the isolated child; time constructor and search together."""
    started = time.perf_counter()
    row = {
        "scenario": str(path),
        "constructor_options": {"cpu_budget": budget},
        "cpu_limit_seconds": cpu_limit,
        "runtime": _runtime(),
        "source_hashes_before": _source_hashes(),
        "construction": None,
    }
    player = None
    cpu = CpuLimit(cpu_limit)
    try:
        from players.player5.player import Player5
        from src.enclosure import score_construction, validate_construction
        from src.scenario import Scenario

        raw_input = path.read_bytes()
        row["scenario_sha256"] = hashlib.sha256(raw_input).hexdigest()
        row["scenario_input"] = json.loads(raw_input)
        scenario = Scenario.from_dict(row["scenario_input"])
        with cpu:
            player = Player5(
                scenario.room.copy(),
                scenario.inventory.copy(),
                scenario.weights,
                cpu_budget=budget,
            )
            candidate = player.build_enclosure()
        row["construction"] = _construction_dict(candidate)
        if candidate is None:
            row.update(status="none", score=0.0)
        else:
            result = validate_construction(candidate, scenario.room, scenario.inventory)
            row.update(
                status="valid" if result.valid else "invalid",
                score=score_construction(result, scenario.weights),
                reason=result.reason,
                area=result.area,
                perimeter=result.perimeter,
                gate_count=result.gate_count,
                second_gate_bonus=result.second_gate_bonus,
            )
    except PlayerTimeout as error:
        row.update(status="timeout", score=0.0, error=str(error))
    except Exception as error:  # noqa: BLE001 -- record a failed run and continue the suite
        row.update(
            status="error",
            score=0.0,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    row["cpu_seconds"] = cpu.used
    row["wall_seconds"] = time.perf_counter() - started
    if player is not None:
        row["stats"] = asdict(player.stats)
        row["note"] = getattr(player, "note", "")
    row["source_hashes_after"] = _source_hashes()
    row["sources_changed_during_run"] = (
        row["source_hashes_before"] != row["source_hashes_after"]
    )
    return row


def _fresh_process(index, path, args, temporary_directory):
    result_path = temporary_directory / f"{index}.json"
    command = [
        sys.executable,
        "-m",
        "players.player5.benchmark",
        str(path),
        "--budget",
        str(args.budget),
        "--cpu-limit",
        str(args.cpu_limit),
        "--output",
        str(result_path),
        "--worker",
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    # This guard catches hangs outside timed Python work. Allow substantial
    # CPU contention when several children share one available processor.
    guard = max(60.0, 4 * args.workers * args.cpu_limit + 30)
    try:
        output, _ = process.communicate(timeout=guard)
    except subprocess.TimeoutExpired:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        output, _ = process.communicate()
        row = {"scenario": str(path), "status": "wall timeout", "score": 0.0}
    else:
        if process.returncode == 0 and result_path.exists():
            row = json.loads(result_path.read_text())
        else:
            row = {
                "scenario": str(path),
                "status": "worker error",
                "score": 0.0,
                "returncode": process.returncode,
            }
    if output.strip():
        row["process_output"] = output
    return index, row


def _save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        type=Path,
        help="scenario JSON paths; defaults to every published scenario",
    )
    parser.add_argument(
        "--budget",
        type=_positive_seconds,
        default=240.0,
        help="internal player CPU budget, in seconds",
    )
    parser.add_argument(
        "--cpu-limit",
        type=_positive_seconds,
        default=300.0,
        help="official CPU limit over constructor and search, in seconds",
    )
    parser.add_argument(
        "--workers",
        type=_worker_count,
        default=1,
        help="independent concurrent scenario processes, from 1 to 4 (default: 1)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    paths = [path.resolve() for path in args.scenarios] or sorted(
        (ROOT / "scenarios").rglob("*.json")
    )
    if not paths:
        parser.error("no scenarios found")
    if args.worker:
        if len(paths) != 1:
            parser.error("the internal worker requires exactly one scenario")
        _save(args.output, _run_scenario(paths[0], args.budget, args.cpu_limit))
        return

    report = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "constructor_options": {"cpu_budget": args.budget},
        "cpu_limit_seconds": args.cpu_limit,
        "workers": args.workers,
        "runtime": _runtime(),
        "protocol": (
            "Fresh process for each scenario. Official CPU limit covers the "
            "Player5 constructor and build_enclosure, including imports inside "
            "build_enclosure. Initial module imports, scenario loading, and "
            "independent output validation are outside that timer. "
            "None, CPU timeout, and process failure are separate statuses."
        ),
        "scenario_count": len(paths),
        "completed_count": 0,
        "results": [],
    }
    rows = {}
    _save(args.output, report)
    with (
        tempfile.TemporaryDirectory(prefix="player5-benchmark-") as directory,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        jobs = [
            executor.submit(_fresh_process, index, path, args, Path(directory))
            for index, path in enumerate(paths)
        ]
        for job in as_completed(jobs):
            index, row = job.result()
            rows[index] = row
            report["results"] = [rows[key] for key in sorted(rows)]
            report["completed_count"] = len(rows)
            report["status_counts"] = dict(
                Counter(item["status"] for item in rows.values())
            )
            _save(args.output, report)
            print(
                f"{len(rows)}/{len(paths)} {row['scenario']}: {row['status']}; "
                f"score {row['score']:g}; CPU {row.get('cpu_seconds', 0):.3f}s",
                flush=True,
            )
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
