"""Exact closure and a diagonal improvement unavailable to rectangular bumps."""

import math
import time
import unittest
from itertools import pairwise

from shapely.geometry import Polygon

from players.player5.packing import PolygonPacker
from players.player5.path_refinement import (
    _closing_lengths,
    _direction_totals,
    _turn_patterns,
    refine_paths,
)
from src.enclosure import validate_construction
from src.inventory import Inventory
from src.pieces import ConnectorType
from src.room import Room
from src.weights import Weights


class PathRefinementTests(unittest.TestCase):
    def test_integer_closure_keeps_cardinal_and_diagonal_terms_separate(self):
        # A diagonal V cannot replace a horizontal integer-length face: the
        # required equal diagonal lengths would be 10*sqrt(2), not integers.
        self.assertEqual(
            list(_closing_lengths((1, 7), (20, 0, 0, 0), tuple(range(5, 31)))),
            [],
        )

    def test_opposite_diagonals_cancel_exactly(self):
        headings = (1, 2, 4, 5)
        target = (-20, 0, 20, 0)
        answers = list(_closing_lengths(headings, target, (10, 20, 30)))
        self.assertIn((10, 20, 20, 10), answers)
        self.assertTrue(
            all(_direction_totals(headings, lengths) == target for lengths in answers)
        )

    def test_patterns_preserve_turn_total(self):
        patterns = _turn_patterns(6, 4)
        self.assertIn((1, 2, 4, 5), patterns)
        for headings in patterns:
            walk = (0,) + headings + (6,)
            turns = [(b - a + 4) % 8 - 4 for a, b in pairwise(walk)]
            self.assertTrue(all(turn in (-2, -1, 1, 2) for turn in turns))
            self.assertEqual(sum(turns), 6)

    def test_rectangle_expands_into_exact_diagonal_room(self):
        offset = 10 / math.sqrt(2)
        room = Room(
            Polygon(
                [
                    (0, 0),
                    (20, 0),
                    (20 + offset, offset),
                    (20 + offset, 20 + offset),
                    (offset, 20 + offset),
                    (0, 20),
                ]
            )
        )
        inventory = Inventory(
            walls={10: 30},
            gates={10: 2},
            connectors={
                ConnectorType.RIGHT: 20,
                ConnectorType.DIAGONAL: 20,
                ConnectorType.STRAIGHT: 30,
            },
        )
        weights = Weights(A=2, C=-1, G=100)
        seed = PolygonPacker(inventory, room, weights).pack(
            [(0, 0), (20, 0), (20, 20), (0, 20)]
        )
        self.assertIsNotNone(seed)
        results = []
        refine_paths(
            room, inventory, weights, [seed], time.process_time() + 2, results.append
        )
        validations = [
            validate_construction(result, room, inventory) for result in results
        ]
        self.assertTrue(validations)
        self.assertTrue(all(result.valid for result in validations))
        self.assertAlmostEqual(
            max(result.area for result in validations), room.polygon.area
        )


if __name__ == "__main__":
    unittest.main()
