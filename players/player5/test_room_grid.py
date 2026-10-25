"""Geometry and stock regressions for the room-following grid search."""

import math
import time
import unittest
from collections import Counter
from itertools import pairwise

from shapely.affinity import rotate
from shapely.geometry import Polygon, box

from players.player5.room_grid import (
    _break_long_faces,
    _corners,
    _edges,
    _mixed_axis,
    _smooth_short_faces,
    search_grid,
)
from src.enclosure import validate_construction
from src.inventory import Inventory
from src.pieces import ConnectorType
from src.room import Room
from src.weights import Weights


def generous_inventory():
    return Inventory(
        walls={length: 100 for length in range(5, 31)},
        gates={10: 2},
        connectors={ConnectorType.RIGHT: 100, ConnectorType.STRAIGHT: 100},
    )


class RoomGridTests(unittest.TestCase):
    def test_notches_are_real_turns_and_preserve_containment(self):
        original = box(0, 0, 100, 40)
        changed = _break_long_faces(original, time.process_time() + 1)
        self.assertIsNotNone(changed)
        self.assertTrue(original.covers(changed))
        self.assertLess(changed.area, original.area)
        lengths = [math.dist(a, b) for a, b in _edges(_corners(changed))]
        self.assertTrue(all(5 <= length <= 30 for length in lengths))

    def test_thin_passage_is_not_illegally_subdivided(self):
        self.assertIsNone(_break_long_faces(box(0, 0, 100, 5), time.process_time() + 1))

    def test_smoothing_removes_unavailable_short_staircase_faces(self):
        original = Polygon(
            [(0, 0), (10, 0), (10, 5), (15, 5), (15, 10), (20, 10), (20, 20), (0, 20)]
        )
        inventory = generous_inventory()
        inventory.walls[5] = 0
        changed = _smooth_short_faces(original, inventory, time.process_time() + 1)
        old_lengths = Counter(
            round(math.dist(a, b)) for a, b in _edges(_corners(original))
        )
        new_lengths = Counter(
            round(math.dist(a, b)) for a, b in _edges(_corners(changed))
        )
        self.assertLess(new_lengths[5], old_lengths[5])
        self.assertTrue(original.covers(changed))
        self.assertTrue(all(5 <= length <= 30 for length in new_lengths))

    def test_mixed_coordinates_have_integer_inventory_lengths(self):
        coordinates = _mixed_axis(-3.2, 40.7, (5, 6, 7, 8), 0.5)
        differences = [round(b - a, 8) for a, b in pairwise(coordinates)]
        self.assertEqual(differences[:4], [5, 6, 7, 8])

    def test_search_validates_rotated_irregular_room_and_preserves_stock(self):
        polygon = Polygon([(0, 0), (40, 0), (40, 20), (20, 20), (20, 40), (0, 40)])
        room = Room(rotate(polygon, 37, origin=(0, 0)))
        inventory = generous_inventory()
        before = inventory.to_dict()
        constructions = []
        stats = search_grid(
            room,
            inventory,
            Weights(2, -1, 100),
            time.process_time() + 1,
            constructions.append,
        )
        self.assertGreater(stats.constructions, 0)
        for construction in constructions:
            self.assertTrue(validate_construction(construction, room, inventory).valid)
        self.assertEqual(inventory.to_dict(), before)

    def test_expired_budget_does_no_search(self):
        constructions = []
        stats = search_grid(
            Room(box(0, 0, 40, 30)),
            generous_inventory(),
            Weights(2, -1, 100),
            time.process_time() - 1,
            constructions.append,
        )
        self.assertEqual(stats.grids, 0)
        self.assertEqual(constructions, [])


if __name__ == "__main__":
    unittest.main()
