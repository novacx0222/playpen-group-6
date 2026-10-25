"""Focused stock-allocation checks: python -m unittest players.player5.test_packing."""

import math
import unittest
from unittest.mock import patch

from shapely import box, union_all
from shapely.geometry import Polygon

from players.player5.packing import PolygonPacker, pack_polygon
from src.enclosure import score_construction, validate_construction
from src.inventory import Inventory
from src.pieces import ConnectorType
from src.room import Room
from src.weights import Weights

RIGHT = ConnectorType.RIGHT
DIAGONAL = ConnectorType.DIAGONAL
STRAIGHT = ConnectorType.STRAIGHT


class PolygonPackingTests(unittest.TestCase):
    def setUp(self):
        self.room = Room(Polygon([(-20, -20), (80, -20), (80, 80), (-20, 80)]))
        self.weights = Weights(2, -1, 100)

    def packer(self, walls, gates, right=4, diagonal=0, straight=0):
        return PolygonPacker(
            Inventory(
                walls, gates, {RIGHT: right, DIAGONAL: diagonal, STRAIGHT: straight}
            ),
            self.room,
            self.weights,
        )

    def validate(self, packer, vertices, bonus=None):
        construction = packer.pack(vertices)
        self.assertIsNotNone(construction)
        result = validate_construction(construction, self.room, packer.inventory)
        self.assertTrue(result.valid, result.reason)
        if bonus is not None:
            self.assertEqual(result.second_gate_bonus, bonus)
        return construction, result

    def test_bonus_bound_counts_only_usable_gates(self):
        for gates, expected in [({5: 1, 40: 1}, 0), ({5: 2, 40: -2}, 100)]:
            with self.subTest(gates=gates):
                packer = self.packer({5: 3}, gates)
                self.assertEqual(packer.max_possible_gate_bonus, expected)

    def test_square_uses_required_gate(self):
        packer = self.packer({5: 3}, {5: 1})
        _, result = self.validate(packer, [(0, 0), (5, 0), (5, 5), (0, 5)], False)
        self.assertEqual(result.gate_count, 1)
        self.assertEqual(result.area, 25)
        self.assertEqual(packer.inventory.walls, {5: 3})

    def test_spare_gates_are_required_building_material(self):
        packer = self.packer({5: 2}, {5: 2})
        _, result = self.validate(packer, [(0, 0), (5, 0), (5, 5), (0, 5)], True)
        self.assertEqual(result.gate_count, 2)

    def test_gates_on_separate_faces_receive_bonus(self):
        packer = self.packer({5: 12}, {5: 2}, straight=4)
        _, result = self.validate(packer, [(0, 0), (10, 0), (10, 10), (0, 10)], True)
        self.assertEqual(score_construction(result, self.weights), 1260)

    def test_two_gates_on_only_one_usable_face_do_not_receive_bonus(self):
        packer = self.packer({6: 1, 10: 2}, {5: 2}, straight=1)
        self.assertIsNone(packer.pack([(0, 0), (10, 0), (10, 6), (0, 6)]))
        # A 10-by-6 rectangle needs two 6-unit sides. Add that missing wall;
        # both gates then must share a 10-unit face to complete the stock.
        packer = self.packer({6: 2, 10: 1}, {5: 2}, straight=1)
        _, result = self.validate(packer, [(0, 0), (10, 0), (10, 6), (0, 6)], False)
        self.assertEqual(result.gate_count, 2)

    def test_shared_stock_and_straight_connectors(self):
        vertices = [(0, 0), (20, 0), (20, 10), (10, 10), (10, 20), (0, 20)]
        packer = self.packer({10: 6}, {10: 2}, right=6, straight=2)
        construction, result = self.validate(packer, vertices, True)
        self.assertEqual(len(construction.pieces), 8)
        self.assertEqual(result.area, 300)
        short = self.packer({10: 6}, {10: 2}, right=6, straight=1)
        self.assertIsNone(short.pack(vertices))

    def test_exact_allocation_recovers_from_a_greedy_dead_end(self):
        # A 15-unit side can use the 10-unit gate plus a 5-unit wall, but
        # that strands a 10-unit side. Both 15-unit walls must be kept, and
        # both 5-unit walls must share the other 10-unit side.
        packer = self.packer({5: 2, 15: 2}, {10: 1}, straight=1)
        _, result = self.validate(packer, [(0, 0), (10, 0), (10, 15), (0, 15)], False)
        self.assertEqual(result.perimeter, 50)

    def test_diagonal_and_clockwise_polygons(self):
        d = 5 / math.sqrt(2)
        vertices = [
            (0, 0),
            (10, 0),
            (10 + d, d),
            (10 + d, 10 + d),
            (d, 10 + d),
            (0, 10),
        ]
        packer = self.packer({10: 3, 5: 2}, {10: 1}, right=2, diagonal=4)
        _, forward = self.validate(packer, vertices, False)
        _, reverse = self.validate(packer, list(reversed(vertices)), False)
        self.assertAlmostEqual(forward.area, reverse.area)

    def test_collinear_points_are_simplified_before_counting_corners(self):
        packer = self.packer({5: 7}, {5: 1}, straight=4)
        construction, _ = self.validate(
            packer, [(0, 0), (5, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        )
        self.assertEqual(len(construction.pieces), 8)

    def test_straight_subdivision_cannot_hide_an_overlong_face(self):
        packer = self.packer({5: 50}, {5: 1}, straight=50)
        self.assertIsNone(packer.pack([(0, 0), (20, 0), (40, 0), (40, 10), (0, 10)]))

    def test_rejects_crossings_unsupported_turns_and_noninteger_lengths(self):
        packer = self.packer(
            {5: 50, 10: 50}, {5: 2}, right=20, diagonal=20, straight=20
        )
        self.assertIsNone(packer.pack([(0, 0), (10, 10), (0, 10), (10, 0)]))
        self.assertIsNone(packer.pack([(0, 0), (10, 0), (8, 6), (0, 10)]))
        self.assertIsNone(packer.pack([(0, 0), (10.1, 0), (10.1, 10), (0, 10)]))

    def test_rejects_missing_corner_connector_and_outside_room(self):
        packer = self.packer({5: 3}, {5: 1}, right=3)
        self.assertIsNone(packer.pack([(0, 0), (5, 0), (5, 5), (0, 5)]))
        packer = self.packer({5: 3}, {5: 1})
        self.assertIsNone(packer.pack([(90, 90), (95, 90), (95, 95), (90, 95)]))

    def test_length_and_assignment_caches_support_different_shapes(self):
        packer = self.packer({5: 6, 10: 5}, {5: 2}, straight=4)
        self.assertEqual(packer.feasible_lengths, frozenset(range(5, 31, 5)))
        self.validate(packer, [(0, 0), (10, 0), (10, 5), (0, 5)], True)
        first_cache_size = len(packer._polygon_allocations)
        self.validate(packer, [(20, 20), (20, 30), (25, 30), (25, 20)], True)
        self.assertEqual(len(packer._polygon_allocations), first_cache_size)

    def test_expired_call_does_not_poison_later_cache(self):
        packer = self.packer({5: 3}, {5: 1})
        vertices = [(0, 0), (5, 0), (5, 5), (0, 5)]
        self.assertIsNone(packer.pack(vertices, deadline=0))
        self.validate(packer, vertices)

    def test_geometry_free_stock_check_shares_the_placement_cache(self):
        packer = self.packer({5: 2, 15: 2}, {10: 1}, straight=1)
        self.assertTrue(packer.can_allocate([10, 15, 10, 15]))
        cached = len(packer._polygon_allocations)
        self.validate(packer, [(0, 0), (10, 0), (10, 15), (0, 15)])
        self.assertEqual(len(packer._polygon_allocations), cached)

    def test_geometry_free_stock_check_distinguishes_impossible_and_unknown(self):
        packer = self.packer({5: 3}, {6: 1})
        self.assertFalse(packer.can_allocate([5, 5, 5, 5]))
        self.assertFalse(packer.can_allocate([5, 5, 5, 31]))
        self.assertFalse(packer.can_allocate([5, 5, 5.5, 5]))
        packer = self.packer({5: 3}, {5: 1})
        self.assertIsNone(packer.can_allocate([5, 5, 5, 5], deadline=0))
        self.assertFalse(packer._polygon_allocations)
        self.assertTrue(packer.can_allocate([5, 5, 5, 5]))

    def test_deadline_expiry_preserves_a_validated_result(self):
        packer = self.packer({6: 2, 10: 1}, {5: 2}, straight=1)
        clock = [0.0]
        original = validate_construction

        def validate_then_expire(*args, **kwargs):
            result = original(*args, **kwargs)
            clock[0] = 10.0
            return result

        with (
            patch(
                "players.player5.packing.time.process_time",
                side_effect=lambda: clock[0],
            ),
            patch(
                "players.player5.packing.validate_construction",
                side_effect=validate_then_expire,
            ),
        ):
            candidate = packer.pack([(0, 0), (10, 0), (10, 6), (0, 6)], deadline=1.0)
        self.assertIsNotNone(candidate)
        self.assertTrue(original(candidate, self.room, packer.inventory).valid)

    def test_function_wrapper(self):
        inventory = Inventory({5: 3}, {5: 1}, {RIGHT: 4})
        self.assertIsNotNone(
            pack_polygon(
                [(0, 0), (5, 0), (5, 5), (0, 5)], inventory, self.room, self.weights
            )
        )

    def test_more_than_a_thousand_faces_do_not_exhaust_the_python_stack(self):
        polygon = union_all(
            [
                box(5 * x, 5 * y, 5 * x + 5, 5 * y + 5)
                for step in range(270)
                for x, y in ((step, step), (step + 1, step))
            ]
        )
        room = Room(polygon)
        inventory = Inventory({5: 3000}, {5: 2}, {RIGHT: 3000, STRAIGHT: 3000})
        packer = PolygonPacker(inventory, room, self.weights)
        construction = packer.pack(polygon.exterior.coords[:-1])
        self.assertIsNotNone(construction)
        result = validate_construction(construction, room, inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertGreater(len(result.faces), 1000)
        self.assertTrue(result.second_gate_bonus)


if __name__ == "__main__":
    unittest.main()
