"""Inventory, scoring, and cooperative time-budget regressions for the grid solver."""

import copy
import time
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import box

from players.player6 import inscribed_polygon as ip
from players.player6 import inscribed_polygon_stock as stock_solver
from players.player6.player import Player6
from players.player6.test_player import make_shape
from src.enclosure import score_construction, validate_construction
from src.inventory import Inventory
from src.pieces import ConnectorType as CT
from src.room import Room
from src.weights import Weights


def make_player(walls=None, gates=None, connectors=None, room=None, weights=None):
    player = Player6(
        Room(room if room is not None else box(0, 0, 40, 40)),
        Inventory(
            walls=walls if walls is not None else {5: 40},
            gates=gates if gates is not None else {5: 2},
            connectors=connectors
            if connectors is not None
            else {CT.RIGHT: 12, CT.STRAIGHT: 40},
        ),
        weights if weights is not None else Weights(A=1, C=-3, G=100),
    )
    player.best_score = 0
    player.best_shape = None
    player.best_fit = None
    player.best_construction = None
    return player


def gate_faces(plan):
    return sum(
        count
        for items in plan.values()
        for pieces, count in items
        if any(kind == Player6.GATE for kind, _ in pieces)
    )


class InscribedInventoryTests(unittest.TestCase):
    def test_enum_connector_limits_are_enforced(self):
        for angles in ({CT.RIGHT: 0}, {"right": 0}):
            with self.subTest(angles=angles):
                self.assertIsNone(
                    stock_solver.buildable([5] * 4, {5: 4}, angle_stock=angles)
                )

    def test_typed_assignment_enforces_straight_limits(self):
        self.assertIsNone(
            stock_solver.assign_inventory(
                [10] * 4, {5: 7}, {5: 1}, {CT.RIGHT: 4, CT.STRAIGHT: 3}
            )
        )
        self.assertIsNotNone(
            stock_solver.assign_inventory(
                [10] * 4, {5: 7}, {5: 1}, {CT.RIGHT: 4, CT.STRAIGHT: 4}
            )
        )

    def test_equal_length_walls_do_not_count_as_gates(self):
        player = make_player(walls={5: 3}, gates={5: 1})
        original = player.inventory.copy()
        plan = stock_solver.assign_inventory([5] * 4, {5: 3}, {5: 1})
        saved_plan = copy.deepcopy(plan)
        self.assertEqual(gate_faces(plan), 1)
        for _ in range(2):
            player._consider_inscribed(box(0, 0, 5, 5), plan)
        result = validate_construction(
            player.best_construction, player.room, player.inventory
        )
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.gate_count, 1)
        self.assertFalse(result.second_gate_bonus)
        self.assertEqual(plan, saved_plan)
        self.assertEqual(player.inventory, original)

    def test_two_gates_are_spread_across_actual_faces(self):
        player = make_player(
            walls={11: 2, 5: 2, 12: 2},
            gates={6: 2},
            connectors={CT.RIGHT: 4, CT.STRAIGHT: 2},
        )
        plan = stock_solver.assign_inventory(
            [11, 12, 11, 12],
            player.inventory.walls,
            player.inventory.gates,
            player.inventory.connectors,
        )
        self.assertEqual(gate_faces(plan), 2)
        player._consider_inscribed(box(0, 0, 11, 12), plan)
        result = validate_construction(
            player.best_construction, player.room, player.inventory
        )
        self.assertTrue(result.valid, result.reason)
        self.assertTrue(result.second_gate_bonus)

    def test_two_gates_on_one_face_remain_a_legal_fallback(self):
        plan = stock_solver.assign_inventory(
            [12, 5, 12, 5], {12: 1, 5: 2}, {6: 2}, {CT.RIGHT: 4, CT.STRAIGHT: 1}
        )
        self.assertIsNotNone(plan)
        self.assertEqual(gate_faces(plan), 1)

    def test_zero_gate_stock_cannot_produce_a_plan(self):
        self.assertIsNone(stock_solver.assign_inventory([5] * 4, {5: 4}, {5: 0}))

    def test_bonus_timeout_retains_feasible_gate_assignment(self):
        feasible = {5: [(((0, 5),), 1), (((1, 5),), 3)]}
        with patch.object(
            stock_solver, "_dfs", side_effect=[feasible, ip.SearchStopped()]
        ):
            plan = stock_solver.assign_inventory([5] * 4, {5: 3}, {5: 2})
        self.assertEqual(plan, feasible)


class InscribedSearchTests(unittest.TestCase):
    def test_smaller_outline_can_win_on_actual_perimeter_score(self):
        player = make_player()
        large, small = box(0, 0, 30, 5), box(0, 0, 10, 10)

        def generate(poly, *, on_candidate, **kwargs):
            on_candidate(large)
            on_candidate(small)
            return large  # Geometry-only ranking would discard the smaller one.

        with (
            patch.object(ip, "solve", side_effect=generate),
            patch.object(stock_solver, "best_rectangle"),
        ):
            polygon, _, _ = stock_solver.solve_with_stock(
                player.room.polygon,
                player.inventory.walls,
                gate_stock=player.inventory.gates,
                cells=(5,),
                angle_stock=player.inventory.connectors,
                evaluate=player._consider_inscribed,
            )
        self.assertTrue(polygon.equals(small))
        result = validate_construction(
            player.best_construction, player.room, player.inventory
        )
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(player.best_score, 1080)
        self.assertAlmostEqual(
            player.best_score, score_construction(result, player.weights)
        )

    def test_timeout_returns_published_candidate_and_restores_grid_settings(self):
        player = make_player()
        original_settings = (ip.CELL, ip.MAX_RUN)

        def generate(poly, *, on_candidate, **kwargs):
            on_candidate(box(0, 0, 10, 10))
            raise ip.SearchStopped()

        with patch.object(ip, "solve", side_effect=generate):
            polygon, plan, _ = stock_solver.solve_with_stock(
                player.room.polygon,
                player.inventory.walls,
                gate_stock=player.inventory.gates,
                cells=(7,),
                evaluate=player._consider_inscribed,
            )
        self.assertIsNotNone(polygon)
        self.assertIsNotNone(plan)
        self.assertIsNotNone(player.best_construction)
        self.assertEqual((ip.CELL, ip.MAX_RUN), original_settings)

    def test_invalid_candidate_does_not_block_later_valid_candidate(self):
        player = make_player(connectors={CT.RIGHT: 4, CT.STRAIGHT: 40})
        bad = make_shape([[(Player6.GATE, 5)]] + [[(Player6.WALL, 5)]] * 3, [90] * 4)

        def generate(poly, *, on_candidate, **kwargs):
            on_candidate(box(100, 100, 110, 110))
            on_candidate(bad.to_polygon())
            return bad.to_polygon()

        with (
            patch.object(ip, "solve", side_effect=generate),
            patch.object(stock_solver, "best_rectangle"),
        ):
            stock_solver.solve_with_stock(
                player.room.polygon,
                player.inventory.walls,
                gate_stock=player.inventory.gates,
                cells=(5,),
                evaluate=player._consider_inscribed,
            )
        self.assertTrue(
            validate_construction(
                player.best_construction, player.room, player.inventory
            ).valid
        )

    def test_full_room_coverage_still_runs_beam_and_compares_gate_bonus(self):
        player = make_player(
            walls={5: 3},
            gates={5: 2},
            room=box(0, 0, 5, 5),
            connectors={CT.RIGHT: 4, CT.DIAGONAL: 2},
            weights=Weights(A=1, C=0, G=100),
        )
        one = make_shape([[(Player6.GATE, 5)]] + [[(Player6.WALL, 5)]] * 3, [90] * 4)
        two = make_shape(
            [[(Player6.GATE, 5)]] * 2 + [[(Player6.WALL, 5)]] * 2, [90] * 4
        )
        with (
            patch.object(
                player,
                "full_checks",
                side_effect=lambda: player._consider_shape(one, (None, (0, 0, 0))),
            ),
            patch.object(player, "generate_quads"),
            patch.object(player, "generate_quad_shapes"),
            patch("players.player6.player.DiagonalSearch") as beam,
        ):
            beam.MAX_FACES = 12
            beam.return_value.build_enclosure.return_value = two.to_construction(
                (0, 0), 0
            )
            construction = player.build_enclosure()
        beam.assert_called_once()
        result = validate_construction(construction, player.room, player.inventory)
        self.assertTrue(result.second_gate_bonus)
        self.assertAlmostEqual(score_construction(result, player.weights), 1125)

    def test_grid_deadline_respects_reserved_beam_budget_and_missing_connectors(self):
        player = make_player(connectors={CT.RIGHT: 4})
        player._deadline = 15
        clock = [0]

        def solve(*args, stop, angle_stock, **kwargs):
            self.assertEqual(angle_stock, {"right": 4, "straight": 0, "diagonal": 0})
            self.assertFalse(stop())
            clock[0] = 16
            self.assertTrue(stop())

        with (
            patch(
                "players.player6.player.time.process_time", side_effect=lambda: clock[0]
            ),
            patch.object(stock_solver, "solve_with_stock", side_effect=solve),
        ):
            player.full_checks()

    def test_real_grid_budget_preserves_existing_valid_answer(self):
        player = make_player()
        plan = stock_solver.assign_inventory(
            [5] * 4, player.inventory.walls, player.inventory.gates
        )
        player._consider_inscribed(box(0, 0, 5, 5), plan)
        player.stage_limits["grid"] = 0.02
        started = time.process_time()
        player.full_checks()
        self.assertLess(time.process_time() - started, 0.5)
        self.assertTrue(
            validate_construction(
                player.best_construction, player.room, player.inventory
            ).valid
        )

    def test_inner_loops_honor_stop(self):
        for action in (
            lambda: ip.enforce(np.ones((20, 20), dtype=bool), stop=lambda: True),
            lambda: stock_solver.compositions(30, range(5, 16), stop=lambda: True),
            lambda: stock_solver.shrink_search(
                box(0, 0, 40, 40), {5: 100}, stop=lambda: True
            ),
            lambda: stock_solver.assign_inventory(
                [5] * 4, {5: 3}, {5: 1}, stop=lambda: True
            ),
        ):
            with self.assertRaises(ip.SearchStopped):
                action()


if __name__ == "__main__":
    unittest.main()
