"""Local moves must improve real scores without breaking geometry or inventory."""

import copy
import unittest

from shapely import affinity
from shapely.geometry import box

from players.player6.local_search import Local45Search
from players.player6.player import Player6
from players.player6.test_player import make_shape
from src.enclosure import build_geometry, score_construction, validate_construction
from src.inventory import Inventory
from src.pieces import ConnectorType as CT
from src.room import Room
from src.weights import Weights


class Local45Tests(unittest.TestCase):
    def setUp(self):
        shape = make_shape(
            [[(Player6.GATE, 10)]] + [[(Player6.WALL, 10)]] * 3, [90] * 4
        )
        self.seed = shape.to_construction((0, 0), 0)
        self.room = Room(box(0, 0, 15, 15))
        self.inventory = Inventory(
            walls={10: 3, 5: 2}, gates={10: 1}, connectors={CT.RIGHT: 4, CT.DIAGONAL: 4}
        )
        self.weights = Weights(A=2, C=-3, G=100)

    def result(self, construction):
        result = validate_construction(construction, self.room, self.inventory)
        self.assertTrue(result.valid, result.reason)
        return result

    def test_opposite_diagonals_improve_and_close_without_fractional_pieces(self):
        original_seed, original_inventory = (
            copy.deepcopy(self.seed),
            self.inventory.copy(),
        )
        search = Local45Search(self.room, self.inventory, self.weights)
        improved = search.improve(self.seed)
        result = self.result(improved)
        self.assertAlmostEqual(result.area, 100 + 50 * 2**0.5)
        self.assertAlmostEqual(
            score_construction(result, self.weights), 1191.4213562373095
        )
        self.assertEqual(search.stats.accepted, 1)
        self.assertTrue(
            all(isinstance(p.length, int) and p.length >= 5 for p in improved.pieces)
        )
        self.assertTrue(all(face.length <= 30 for face in result.faces))
        _, position_gap, heading_gap = build_geometry(improved)
        self.assertLess(position_gap, 1e-7)
        self.assertLess(heading_gap, 1e-7)
        self.assertEqual(self.seed, original_seed)
        self.assertEqual(self.inventory, original_inventory)

    def test_rotated_translated_room_has_same_improvement(self):
        self.room = Room(
            affinity.translate(
                affinity.rotate(self.room.polygon, 37, origin=(0, 0)), xoff=-15, yoff=8
            )
        )
        self.seed.start = (-15, 8)
        self.seed.start_heading = 37
        result = self.result(
            Local45Search(self.room, self.inventory, self.weights).improve(self.seed)
        )
        self.assertAlmostEqual(result.area, 100 + 50 * 2**0.5)

    def test_clockwise_seed_has_same_improvement(self):
        shape = make_shape(
            [[(Player6.GATE, 10)]] + [[(Player6.WALL, 10)]] * 3, [-90] * 4
        )
        self.seed = shape.to_construction((0, 10), 0)
        result = self.result(
            Local45Search(self.room, self.inventory, self.weights).improve(self.seed)
        )
        self.assertAlmostEqual(result.area, 100 + 50 * 2**0.5)

    def test_start_in_middle_of_straight_face_preserves_gate_and_closure(self):
        shape = make_shape(
            [[(Player6.GATE, 5), (Player6.WALL, 5)]] + [[(Player6.WALL, 10)]] * 3,
            [90] * 4,
        )
        self.seed = shape.to_construction((0, 0), 0)
        self.seed.pieces = self.seed.pieces[1:] + self.seed.pieces[:1]
        self.seed.connectors = self.seed.connectors[1:] + self.seed.connectors[:1]
        self.seed.start = (5, 0)
        self.inventory.walls[5] = 3
        self.inventory.gates = {5: 1}
        self.inventory.connectors[CT.STRAIGHT] = 1
        result = self.result(
            Local45Search(self.room, self.inventory, self.weights).improve(self.seed)
        )
        self.assertAlmostEqual(result.area, 100 + 50 * 2**0.5)
        self.assertEqual(result.gate_count, 1)

    def test_bounded_translation_can_fit_move_when_seed_is_off_center(self):
        self.seed.start = (3, 2)
        result = self.result(
            Local45Search(self.room, self.inventory, self.weights).improve(self.seed)
        )
        self.assertAlmostEqual(result.area, 100 + 50 * 2**0.5)

    def test_missing_diagonal_connectors_keeps_seed(self):
        self.inventory.connectors[CT.DIAGONAL] = 3
        improved = Local45Search(self.room, self.inventory, self.weights).improve(
            self.seed
        )
        self.assertEqual(improved, self.seed)
        self.result(improved)

    def test_insufficient_spare_pieces_keeps_seed(self):
        self.inventory.walls[5] = 1
        improved = Local45Search(self.room, self.inventory, self.weights).improve(
            self.seed
        )
        self.assertEqual(improved, self.seed)
        self.result(improved)

    def test_room_boundary_rejects_outward_moves(self):
        self.room = Room(box(0, 0, 10, 10))
        improved = Local45Search(self.room, self.inventory, self.weights).improve(
            self.seed
        )
        self.assertEqual(improved, self.seed)

    def test_perimeter_cost_can_make_extra_area_unprofitable(self):
        weights = Weights(A=1, C=-100, G=0)
        improved = Local45Search(self.room, self.inventory, weights).improve(self.seed)
        self.assertEqual(improved, self.seed)

    def test_spare_gate_can_earn_real_second_face_bonus(self):
        self.inventory.gates[5] = 1
        self.inventory.walls[5] = 1
        improved = Local45Search(self.room, self.inventory, self.weights).improve(
            self.seed
        )
        result = self.result(improved)
        self.assertEqual(result.gate_count, 2)
        self.assertTrue(result.second_gate_bonus)
        self.assertAlmostEqual(
            score_construction(result, self.weights), 1291.4213562373095
        )

    def test_two_existing_gates_on_one_face_do_not_invent_bonus(self):
        shape = make_shape(
            [[(Player6.GATE, 5), (Player6.GATE, 5)]] + [[(Player6.WALL, 10)]] * 3,
            [90] * 4,
        )
        self.seed = shape.to_construction((0, 0), 0)
        self.inventory.gates = {5: 2}
        self.inventory.connectors[CT.STRAIGHT] = 1
        result = self.result(
            Local45Search(self.room, self.inventory, self.weights).improve(self.seed)
        )
        self.assertEqual(result.gate_count, 2)
        self.assertFalse(result.second_gate_bonus)

    def test_stop_after_improvement_retains_that_valid_result(self):
        stopped = [False]

        def accept(_):
            stopped[0] = True

        search = Local45Search(
            self.room,
            self.inventory,
            self.weights,
            stop=lambda: stopped[0],
            on_improvement=accept,
        )
        result = self.result(search.improve(self.seed))
        self.assertGreater(score_construction(result, self.weights), 1080)
        self.assertEqual(search.stats.accepted, 1)

    def test_zero_budget_returns_valid_seed(self):
        search = Local45Search(self.room, self.inventory, self.weights, time_budget=0)
        self.assertEqual(search.improve(self.seed), self.seed)
        self.assertEqual(search.stats.accepted, 0)

    def test_invalid_seed_is_not_returned(self):
        self.seed.start = (100, 100)
        self.assertIsNone(
            Local45Search(self.room, self.inventory, self.weights).improve(self.seed)
        )


if __name__ == "__main__":
    unittest.main()
