"""Dynamic allocation, actual simulator limits, GUI reuse, and beam bounds."""

import sys
import unittest
from unittest.mock import patch

from shapely.geometry import box

from players.player6 import time_budget
from players.player6.player import Player6
from players.player6.test_inscribed import make_player
from players.player6.test_player import make_shape
from src.args import Args
from src.enclosure import validate_construction
from src.game import Game
from src.inventory import Inventory
from src.limits import CAN_INTERRUPT, CpuLimit
from src.pieces import ConnectorType as CT
from src.room import Room
from src.weights import Weights


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.inventory = Inventory(
            walls={5: 20}, gates={5: 2}, connectors={CT.RIGHT: 12, CT.DIAGONAL: 12}
        )

    def budget(self, **kwargs):
        with patch.object(time_budget, "active_framework_limit", return_value=None):
            return time_budget.make_budget(self.inventory, argv=[], **kwargs)

    def test_cli_forms_and_default_match_simulator(self):
        for argv in (["--cpu-limit", "60"], ["--cpu-limit=60"], ["--cpu-l", "60"]):
            with (
                self.subTest(argv=argv),
                patch.object(time_budget, "active_framework_limit", return_value=None),
            ):
                budget = time_budget.make_budget(self.inventory, argv=argv)
                self.assertEqual(budget.cpu_limit, 60)
                self.assertEqual(budget.source, "command line")
        self.assertEqual(self.budget().cpu_limit, 300)

    def test_proportions_scale_to_input_limit(self):
        for limit, usable in ((10, 9.7), (60, 58.2), (300, 295)):
            with self.subTest(limit=limit):
                budget = self.budget(cpu_limit=limit)
                self.assertAlmostEqual(budget.usable_seconds, usable)
                self.assertAlmostEqual(budget.grid_seconds, usable * 0.30)
                self.assertAlmostEqual(budget.beam_seconds, usable * 0.50)
                self.assertAlmostEqual(budget.local_seconds, usable * 0.05)
                orthogonal = usable - budget.beam_seconds - budget.local_seconds
                self.assertAlmostEqual(orthogonal, usable * 0.45)

    def test_active_simulator_limit_overrides_different_cli_limit(self):
        with patch.object(
            time_budget, "active_framework_limit", return_value=(10, 0.8)
        ):
            budget = time_budget.make_budget(self.inventory, argv=["--cpu-limit=300"])
        self.assertEqual(budget.cpu_limit, 10)
        self.assertAlmostEqual(budget.available_seconds, 0.8)
        self.assertAlmostEqual(budget.usable_seconds, 0.64)
        self.assertEqual(budget.source, "simulator timer")

    def test_explicit_api_budget_still_respects_shorter_external_timer(self):
        with patch.object(
            time_budget, "active_framework_limit", return_value=(10, 0.4)
        ):
            budget = time_budget.make_budget(self.inventory, cpu_limit=2)
        self.assertAlmostEqual(budget.available_seconds, 0.4)
        self.assertLess(budget.usable_seconds, 0.4)

    def test_stage_overrides_scale_down_under_small_total_budget(self):
        budget = self.budget(
            cpu_limit=0.1, stage_limits={"grid": 20, "beam": 20, "local": 2}
        )
        self.assertLessEqual(
            budget.grid_seconds + budget.beam_seconds + budget.local_seconds,
            budget.usable_seconds + 1e-10,
        )
        self.assertGreater(budget.safety_seconds, 0)

    def test_missing_connectors_redistribute_budget(self):
        self.inventory.connectors = {CT.RIGHT: 4}
        budget = self.budget(cpu_limit=10)
        self.assertEqual(budget.beam_seconds, 0)
        self.assertEqual(budget.local_seconds, 0)
        self.assertAlmostEqual(budget.grid_seconds, budget.usable_seconds * 0.6)

    def test_zero_local_cap_disables_local_pass(self):
        self.assertEqual(
            self.budget(cpu_limit=10, stage_limits={"local": 0}).local_seconds, 0
        )

    def test_bad_limits_are_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.budget(cpu_limit=value)
        with self.assertRaises(ValueError):
            self.budget(stage_limits={"beam": -1})
        with self.assertRaises(ValueError):
            self.budget(stage_limits={"typo": 1})

    @unittest.skipUnless(CAN_INTERRUPT, "Unix CPU timer required")
    def test_timer_is_observed_without_being_reset(self):
        with CpuLimit(1):
            active = time_budget.active_framework_limit()
            time_budget.make_budget(self.inventory)
            later = time_budget.active_framework_limit()
        self.assertEqual(active[0], 1)
        self.assertGreater(active[1], 0)
        self.assertLessEqual(later[1], active[1])


class DynamicRunTests(unittest.TestCase):
    @unittest.skipUnless(CAN_INTERRUPT, "Unix CPU timer required")
    def test_same_player_refreshes_limit_for_each_run(self):
        player = Player6(
            Room(box(0, 0, 5, 5)),
            Inventory(walls={5: 3}, gates={5: 1}, connectors={CT.RIGHT: 4}),
            Weights(A=2, C=-3, G=100),
        )
        for seconds in (0.2, 0.3):
            with CpuLimit(seconds):
                construction = player.build_enclosure()
            self.assertEqual(player.budget_info.cpu_limit, seconds)
            self.assertTrue(
                validate_construction(construction, player.room, player.inventory).valid
            )

    @unittest.skipUnless(CAN_INTERRUPT, "Unix CPU timer required")
    def test_programmatic_game_limit_is_used_even_when_argv_says_300(self):
        with patch.object(sys, "argv", ["main.py", "--cpu-limit=300"]):
            game = Game(
                Args(
                    gui=False,
                    player="6",
                    scenario_path="scenarios/simple_rectangle.json",
                    seed=1,
                    difficulty="generous",
                    export_scenario=None,
                    debug=False,
                    sandbox=False,
                    cpu_limit=0.5,
                )
            )
        self.assertIsNone(game.timeout_message)
        self.assertTrue(game.result.valid, game.result.reason)
        self.assertEqual(game.player.budget_info.cpu_limit, 0.5)
        self.assertLess(game.cpu_used, 0.5)

    def test_large_budget_reclaims_time_for_beam_instead_of_capping_at_20(self):
        player = make_player(
            walls={5: 3},
            gates={5: 2},
            room=box(0, 0, 5, 5),
            connectors={CT.RIGHT: 4, CT.DIAGONAL: 4},
        )
        player.cpu_limit = 300
        shape = make_shape([[(Player6.GATE, 5)]] + [[(Player6.WALL, 5)]] * 3, [90] * 4)
        with (
            patch.object(time_budget, "active_framework_limit", return_value=None),
            patch("players.player6.player.time.process_time", return_value=10),
            patch.object(player, "enumerate_faces"),
            patch.object(
                player,
                "full_checks",
                side_effect=lambda: player._consider_shape(shape, (None, (0, 0, 0))),
            ),
            patch.object(player, "generate_quads"),
            patch.object(player, "generate_quad_shapes"),
            patch.object(player, "_improve_local"),
            patch("players.player6.player.DiagonalSearch") as beam,
        ):
            beam.MAX_FACES = 12
            beam.return_value.build_enclosure.return_value = None
            player.build_enclosure()
        self.assertGreater(beam.call_args.kwargs["time_budget"], 250)
        self.assertEqual(beam.call_args.kwargs["incumbent_score"], player.best_score)
        self.assertAlmostEqual(player.budget_info.local_seconds, 14.75)

    def test_beam_upper_bound_safely_skips_uncompetitive_large_room_search(self):
        player = make_player(
            walls={5: 1000},
            gates={5: 2},
            room=box(0, 0, 400, 400),
            connectors={CT.RIGHT: 4000, CT.DIAGONAL: 4000},
        )
        player.cpu_limit = 60

        def grid():
            player.best_score = 101390

        with (
            patch.object(player, "enumerate_faces"),
            patch.object(player, "full_checks", side_effect=grid),
            patch.object(player, "generate_quads"),
            patch.object(player, "generate_quad_shapes"),
            patch.object(player, "_improve_local"),
            patch("players.player6.player.DiagonalSearch") as beam,
        ):
            beam.MAX_FACES = 12
            player.build_enclosure()
        beam.assert_not_called()
        self.assertTrue(player.beam_skipped)
        self.assertLess(player.beam_upper_bound, 101390)


if __name__ == "__main__":
    unittest.main()
