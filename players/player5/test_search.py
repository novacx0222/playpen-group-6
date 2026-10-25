"""End-to-end checks for the complete-shape and room-grid player."""

import math
import time
import unittest
from dataclasses import replace

from shapely.affinity import rotate
from shapely.geometry import Polygon

from players.player5.beam_fallback import BeamFallback
from players.player5.player import Player5
from src.enclosure import Construction, score_construction, validate_construction
from src.inventory import Inventory
from src.limits import CpuLimit
from src.pieces import Connector, ConnectorType, Piece, PieceType
from src.room import Room
from src.weights import Weights

RIGHT = ConnectorType.RIGHT


class StartingShapeTests(unittest.TestCase):
    def setUp(self):
        room = Room(Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]))
        inventory = Inventory({5: 3}, {5: 2}, {RIGHT: 4})
        self.player = Player5(room, inventory, Weights(A=2, C=-1, G=100))

    @staticmethod
    def square(start=(0, 0)):
        return Construction(
            start,
            0,
            [Piece(PieceType.GATE, 5)] + [Piece(PieceType.WALL, 5)] * 3,
            [Connector(RIGHT)] * 4,
        )

    def keep(self, construction):
        self.player._keep_candidate(construction, "test", time.process_time())

    def test_same_outline_keeps_its_better_gate_allocation(self):
        one_gate = self.square()
        two_gates = replace(
            one_gate,
            pieces=[Piece(PieceType.GATE, 5)] * 2 + [Piece(PieceType.WALL, 5)] * 2,
        )
        self.keep(one_gate)
        self.keep(two_gates)
        self.keep(one_gate)
        self.assertEqual(len(self.player._starting_shapes), 1)
        seed = self.player._starting_shapes[0]
        self.assertIs(seed.construction, two_gates)
        self.assertEqual(seed.score, 1130)
        self.assertEqual(self.player.best_score, seed.score)

    def test_pool_caps_twelve_outlines_and_prefers_different_categories(self):
        # Many squares in one cell must leave room for other locations.
        for x in range(15):
            self.keep(self.square((x, 0)))
        for x in (25, 50, 75):
            self.keep(self.square((x, 0)))
        seeds = self.player._starting_shapes
        self.assertEqual(len(seeds), 12)
        self.assertEqual(len({seed.outline for seed in seeds}), 12)
        self.assertEqual(len({seed.category for seed in seeds[:4]}), 4)

    def test_straight_piece_junctions_do_not_duplicate_an_outline(self):
        inventory = Inventory(
            {5: 2, 10: 3}, {5: 2, 10: 1}, {RIGHT: 4, ConnectorType.STRAIGHT: 2}
        )
        self.player = Player5(self.player.room, inventory, self.player.weights)
        one_gate = Construction(
            (0, 0),
            0,
            [Piece(PieceType.GATE, 10)] + [Piece(PieceType.WALL, 10)] * 3,
            [Connector(RIGHT)] * 4,
        )
        two_gates = Construction(
            (0, 0),
            0,
            [Piece(PieceType.GATE, 5), Piece(PieceType.WALL, 5)] * 2
            + [Piece(PieceType.WALL, 10)] * 2,
            [Connector(RIGHT), Connector(ConnectorType.STRAIGHT)] * 2
            + [Connector(RIGHT)] * 2,
        )
        self.keep(replace(one_gate, start=(25, 0)))
        self.keep(one_gate)
        self.keep(two_gates)
        self.keep(one_gate)
        seeds = self.player._starting_shapes
        self.assertEqual(len(seeds), 2)
        self.assertIs(seeds[0].construction, two_gates)
        self.assertEqual([seed.score for seed in seeds], [1260, 1160])


class PlayerTests(unittest.TestCase):
    def build(self, inventory, polygon=None, weights=None):
        room = Room(polygon or Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]))
        weights = weights or Weights(A=2, C=-1, G=100)
        before = inventory.to_dict()
        player = Player5(room, inventory, weights, cpu_budget=1.0)
        with CpuLimit(5):
            construction = player.build_enclosure()
        self.assertEqual(inventory.to_dict(), before)
        if construction is None:
            return player, None
        result = validate_construction(construction, room, inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(player.best_score, score_construction(result, weights))
        return player, result

    def test_square_without_straight_connectors(self):
        player, result = self.build(Inventory({5: 3}, {5: 1}, {RIGHT: 4}))
        self.assertIsNotNone(result)
        self.assertEqual(player.best_score, 1030)

    def test_both_gates_can_be_required_building_material(self):
        player, result = self.build(Inventory({5: 2}, {5: 2}, {RIGHT: 4}))
        self.assertIsNotNone(result)
        self.assertTrue(result.second_gate_bonus)
        self.assertEqual(player.best_score, 1130)

    def test_room_rotation_preserves_the_feasible_square(self):
        polygon = rotate(Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]), 30)
        player, result = self.build(Inventory({5: 3}, {5: 1}, {RIGHT: 4}), polygon)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(player.best_score, 1030)

    def test_no_gate_returns_none(self):
        player, result = self.build(Inventory({5: 4}, {}, {RIGHT: 4}))
        self.assertIsNone(result)
        self.assertEqual(player.best_score, 0)

    def test_negative_enclosures_do_not_replace_zero(self):
        player, result = self.build(
            Inventory({5: 3}, {5: 1}, {RIGHT: 4}),
            weights=Weights(A=1, C=-100, G=0),
        )
        self.assertIsNone(result)
        self.assertEqual(player.best_score, 0)

    def test_impossible_corner_budget_returns_none(self):
        player, result = self.build(Inventory({5: 4}, {5: 1}, {RIGHT: 3}))
        self.assertIsNone(result)
        self.assertEqual(player.stats.candidates, 0)

    def test_bad_budgets_are_rejected(self):
        room = Room(Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]))
        for budget in (0, -1, math.inf, math.nan):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                Player5(room, Inventory(), Weights(A=2, C=-1, G=0), cpu_budget=budget)

    def test_fallback_returns_a_valid_enclosure_within_the_shared_deadline(self):
        room = Room(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]))
        inventory = Inventory({5: 3}, {5: 1}, {RIGHT: 4})
        before = inventory.to_dict()
        weights = Weights(A=2, C=-1, G=100)
        player = BeamFallback(
            room, inventory, weights, deadline=time.process_time() + 0.5
        )
        construction = player.build_enclosure()
        self.assertIsNotNone(construction)
        result = validate_construction(construction, room, inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(player.best_score, 1030)
        self.assertEqual(inventory.to_dict(), before)
        player.deadline = time.process_time() - 1
        self.assertIsNone(player.build_enclosure())


if __name__ == "__main__":
    unittest.main()
