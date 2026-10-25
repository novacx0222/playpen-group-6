"""Checks for closed-outline moves and immediate reporting of improvements."""

import time
import unittest

from shapely.geometry import Polygon

from players.player5.packing import PolygonPacker
from players.player5.refinement import _measure_faces, _simplify_outline, refine
from src.enclosure import score_construction, validate_construction
from src.inventory import Inventory
from src.pieces import ConnectorType
from src.room import Room
from src.weights import Weights


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.inventory = Inventory(
            walls={10: 20},
            gates={10: 2},
            connectors={ConnectorType.RIGHT: 20, ConnectorType.STRAIGHT: 30},
        )
        self.weights = Weights(A=2, C=-1, G=100)

    def _seed(self, room, vertices):
        construction = PolygonPacker(self.inventory, room, self.weights).pack(vertices)
        self.assertIsNotNone(construction)
        return construction

    def test_straight_piece_junctions_are_removed(self):
        points = [(0, 0), (10, 0), (20, 0), (20, 10), (0, 10), (0, 0)]
        self.assertEqual(len(_simplify_outline(points)), 4)
        self.assertEqual(Polygon(_simplify_outline(points)).area, 200)

    def test_faces_must_have_integer_lengths_and_allowed_turns(self):
        self.assertIsNotNone(_measure_faces(((0, 0), (10, 0), (10, 10), (0, 10))))
        self.assertIsNone(_measure_faces(((0, 0), (10.5, 0), (10.5, 10), (0, 10))))
        self.assertIsNone(_measure_faces(((0, 0), (31, 0), (31, 10), (0, 10))))

    def test_complete_square_grows_to_fill_rectangle(self):
        room = Room(Polygon([(0, 0), (30, 0), (30, 20), (0, 20)]))
        seed = self._seed(room, [(0, 0), (10, 0), (10, 10), (0, 10)])
        results = []
        refine(
            room,
            self.inventory,
            self.weights,
            [seed],
            time.process_time() + 1.5,
            results.append,
        )
        self.assertTrue(results)
        validations = [
            validate_construction(result, room, self.inventory) for result in results
        ]
        self.assertTrue(all(result.valid for result in validations))
        self.assertEqual(max(result.area for result in validations), 600)

    def test_partial_face_extension_enters_side_room(self):
        room = Room(
            Polygon(
                [
                    (0, 0),
                    (20, 0),
                    (20, 10),
                    (30, 10),
                    (30, 20),
                    (20, 20),
                    (20, 30),
                    (0, 30),
                ]
            )
        )
        seed = self._seed(room, [(0, 0), (20, 0), (20, 30), (0, 30)])
        original_score = score_construction(
            validate_construction(seed, room, self.inventory), self.weights
        )
        results = []
        refine(
            room,
            self.inventory,
            self.weights,
            [seed],
            time.process_time() + 1.5,
            results.append,
        )
        validations = [
            validate_construction(result, room, self.inventory) for result in results
        ]
        self.assertTrue(any(result.area == 700 for result in validations))
        self.assertTrue(
            all(
                result.valid
                and score_construction(result, self.weights) > original_score
                for result in validations
            )
        )

    def test_expired_deadline_does_no_work(self):
        room = Room(Polygon([(0, 0), (30, 0), (30, 20), (0, 20)]))
        results = []
        refine(
            room,
            self.inventory,
            self.weights,
            [],
            time.process_time() - 1,
            results.append,
        )
        self.assertEqual(results, [])

    def test_impossible_large_outlines_do_not_crowd_out_feasible_move(self):
        self.inventory = Inventory(
            walls={8: 1, 9: 4, 7: 1, 6: 3},
            gates={8: 2},
            connectors={ConnectorType.RIGHT: 10, ConnectorType.STRAIGHT: 13},
        )
        room = Room(Polygon([(-30, -30), (60, -30), (60, 60), (-30, 60)]))
        seed = self._seed(room, [(0, 0), (27, 0), (27, 8), (0, 8)])
        original_score = score_construction(
            validate_construction(seed, room, self.inventory), self.weights
        )
        results = []
        refine(
            room,
            self.inventory,
            self.weights,
            [seed],
            time.process_time() + 1.5,
            results.append,
        )
        self.assertTrue(
            any(
                score_construction(
                    validate_construction(result, room, self.inventory), self.weights
                )
                > original_score
                for result in results
            )
        )


if __name__ == "__main__":
    unittest.main()
