"""The tournament runner: which scenarios and players it picks, the CSV it
writes, and how it records a crash or a stuck player. Run with
`uv run python -m tests.test_tournament`.

Only the example player is ever run here, so a group's broken player can't
make these tests fail."""

import csv
import shutil
import tempfile
from pathlib import Path

import tournament.run_tournament as t

PROJECT = Path(__file__).resolve().parent.parent
ONE_ROOM = PROJECT / "tournament" / "scenarios" / "1a.json"


def _run(*argv) -> tuple[int, list[dict], Path]:
    out = Path(tempfile.mkdtemp()) / "run"
    code = t.main([*map(str, argv), "--out", str(out), "--jobs", "2"])
    with open(out / "results.csv", newline="") as f:
        return code, list(csv.DictReader(f)), out


# ---------------------------------------------------------------- choosing


def test_the_default_is_the_tournament_folder_in_natural_order():
    names = [p.name for p in t.collect_scenarios(None, everything=False)]
    assert names == [
        "1a.json",
        "1b.json",
        "2a.json",
        "2b.json",
        "3a.json",
        "3b.json",
        "4a.json",
        "4b.json",
        "5.json",
        "6.json",
    ]


def test_everything_adds_the_shared_and_group_scenarios_but_never_the_invalid_demo():
    # a stand-in groups folder, so what the real groups commit can't matter here
    groups = Path(tempfile.mkdtemp()) / "players"
    (groups / "player2").mkdir(parents=True)
    shutil.copy(
        PROJECT / "scenarios" / "l_shaped_room.json", groups / "player2" / "mine.json"
    )
    (groups / "player2" / "other.json").write_text(
        (PROJECT / "scenarios" / "simple_rectangle.json").read_text() + " "
    )
    real, t.GROUP_SCENARIOS_DIR = t.GROUP_SCENARIOS_DIR, groups
    try:
        everything = t.collect_scenarios(None, everything=True)
    finally:
        t.GROUP_SCENARIOS_DIR = real
    names = {p.name for p in everything}
    assert {"1a.json", "simple_rectangle.json", "other.json"} <= names
    assert "invalid_examples.json" not in names
    # l_shaped_room.json is both shared and in the group folder: one run only
    assert "mine.json" not in names and "l_shaped_room.json" in names


def test_a_room_copied_into_two_places_runs_once():
    folder = Path(tempfile.mkdtemp())
    shutil.copy(ONE_ROOM, folder / "a.json")
    shutil.copy(ONE_ROOM, folder / "b.json")
    assert [p.name for p in t.collect_scenarios([folder], False)] == ["a.json"]


def test_a_file_named_on_its_own_is_included_even_if_it_is_usually_skipped():
    demo = PROJECT / "scenarios" / "invalid_examples.json"
    assert t.collect_scenarios([demo], False) == [demo]


def test_players_are_the_example_plus_the_ten_groups_and_unknown_ones_are_refused():
    assert t.collect_players(None) == ["e"] + [str(n) for n in range(1, 11)]
    assert t.collect_players(["3", "e", "3"]) == ["3", "e"]
    try:
        t.collect_players(["i"])  # the demos are not tournament players
    except SystemExit:
        return
    raise AssertionError("an unknown player should be refused")


def test_scenarios_are_labelled_by_where_they_live():
    assert t.set_of(ONE_ROOM) == "tournament"
    assert t.set_of(PROJECT / "scenarios" / "simple_rectangle.json") == "shared"
    assert (
        t.set_of(PROJECT / "scenarios" / "players" / "player4" / "x.json") == "group4"
    )


# ------------------------------------------------------------ running them


def test_a_run_writes_one_row_with_every_column():
    code, rows, out = _run("--players", "e", "--scenarios", ONE_ROOM)
    assert code == 0 and len(rows) == 1
    row = rows[0]
    assert list(row) == t.COLUMNS
    assert (row["set"], row["player"], row["scenario"]) == (
        "tournament",
        "example",
        "tournament/scenarios/1a.json",
    )
    assert (row["status"], row["valid"], row["score"]) == ("ok", "True", "1176")
    assert row["area"] == "240" and row["perimeter"] == "64"
    assert float(row["cpu_seconds"]) >= 0 and float(row["wall_seconds"]) > 0
    assert (out / "summary.csv").exists()
    assert list((out / "logs").glob("*.log")), "each run keeps its own log"


def test_no_solution_is_a_row_not_a_failure_of_the_run():
    _, rows, _ = _run(
        "--players", "e", "--scenarios", PROJECT / "tournament" / "scenarios" / "6.json"
    )
    assert (rows[0]["status"], rows[0]["valid"], rows[0]["score"]) == (
        "no_solution",
        "False",
        "0",
    )
    assert rows[0]["area"] == ""


def test_a_broken_scenario_is_recorded_as_a_crash_and_the_rest_still_run():
    folder = Path(tempfile.mkdtemp())
    (folder / "broken.json").write_text("{ this is not json")
    _, rows, out = _run("--players", "e", "--scenarios", folder, ONE_ROOM)
    by_name = {Path(r["scenario"]).name: r for r in rows}
    assert by_name["broken.json"]["status"] == "crash"
    assert by_name["broken.json"]["reason"].startswith("JSONDecodeError")
    assert by_name["1a.json"]["status"] == "ok"
    assert "Traceback" in next((out / "logs").glob("*broken*.log")).read_text()


def test_a_run_that_takes_too_long_is_killed_and_recorded():
    _, rows, _ = _run("--players", "e", "--scenarios", ONE_ROOM, "--wall-limit", "0.01")
    assert (rows[0]["status"], rows[0]["score"]) == ("wall_timeout", "0")


def test_resume_skips_what_is_done_and_refuses_to_overwrite_without_it():
    out = Path(tempfile.mkdtemp()) / "run"
    base = ["--players", "e", "--scenarios", str(ONE_ROOM), "--out", str(out)]
    assert t.main(base) == 0
    assert t.main([*base, "--resume"]) == 0
    with open(out / "results.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 1  # not run a second time
    try:
        t.main(base)
    except SystemExit:
        return
    raise AssertionError("an existing results.csv needs --resume")


# ----------------------------------------------------------------- summary


def _row(player, status, score, cpu="1", scenario_set="tournament"):
    return {
        "set": scenario_set,
        "player": player,
        "status": status,
        "score": score,
        "cpu_seconds": cpu,
    }


def test_the_summary_totals_each_player_and_ranks_by_score():
    rows = [
        _row("player1", "ok", "1100", "2"),
        _row("player1", "no_solution", "0", "1"),
        _row("player2", "ok", "1000", "1"),
        _row("player2", "ok", "500", "3"),
        _row("player2", "crash", "0", ""),
        _row("player2", "wall_timeout", "0", "", scenario_set="shared"),
    ]
    summary = {(s["set"], s["player"]): s for s in t.summarize(rows)}
    one = summary[("tournament", "player1")]
    assert (one["runs"], one["valid_runs"], one["total_score"]) == (2, 1, 1100)
    assert (one["failures"], one["max_cpu_seconds"]) == (1, 2)
    two = summary[("tournament", "player2")]
    assert (two["total_score"], two["crashes"], two["total_cpu_seconds"]) == (
        1500,
        1,
        4,
    )
    assert summary[("shared", "player2")]["timeouts"] == 1  # sets are kept apart
    ranking = [s["player"] for s in t.summarize(rows) if s["set"] == "tournament"]
    assert ranking == ["player2", "player1"]


ALL_TESTS = [v for k, v in list(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for test in ALL_TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {test.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failures}/{len(ALL_TESTS)} passed")
    if failures:
        raise SystemExit(1)
