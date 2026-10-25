import math
import unittest

from shapely.geometry import Polygon

from players.player10.assign import Assigner
from players.player10.beam import Walk, Walker
from players.player10.placement import RoomSearch
from players.player10.player import Player10, segmented_rectangle
from players.player10.shapes import FREE, Shape
from src.scenario import read_scenario


class SegmentedRectangleTests(unittest.TestCase):
    def test_segmented_rectangle_closes_and_matches_analytic_area(self):
        shape = segmented_rectangle(a=30, middle=20, b=25, d=5, h=8)
        polygon = Polygon(shape.vertices())

        self.assertTrue(polygon.is_valid)
        self.assertAlmostEqual(polygon.area, shape.area)
        self.assertAlmostEqual(polygon.length, shape.perimeter)
        self.assertAlmostEqual(
            shape.area,
            (30 + 20 + 25 + math.sqrt(2.0) * 5) * 8 - math.sqrt(2.0) * 20 * 5 - 25,
        )

    def test_segmented_rectangle_obeys_face_limit(self):
        shape = segmented_rectangle(a=30, middle=30, b=30, d=5, h=8)

        self.assertEqual(len(shape.faces), 12)
        self.assertLessEqual(max(shape.faces), 30)
        self.assertEqual(sum(abs(turn) == 45 for turn in shape.turns), 8)
        self.assertEqual(sum(abs(turn) == 90 for turn in shape.turns), 4)


class DenseRasterTests(unittest.TestCase):
    @staticmethod
    def _player(path):
        scenario = read_scenario(path)
        player = Player10(scenario.room, scenario.inventory, scenario.weights)
        player._assigner = Assigner(scenario.inventory, want_second_gate=True)
        return player

    def test_dense_raster_is_reserved_for_complex_well_supplied_rooms(self):
        large_spiral = self._player("scenarios/players/player6/spiral.json")
        curved = self._player("scenarios/players/player8/curved.json")

        self.assertTrue(large_spiral._dense_raster_worthwhile())
        self.assertFalse(curved._dense_raster_worthwhile())

    def test_polish_description_stays_compact(self):
        shape = Shape("polish", tuple(range(200)), (5,) * 200, (90.0,) * 200, 1, 2)

        self.assertEqual(shape.describe(), "polish, 200 faces")

    def test_small_complex_inventory_gets_stock_aware_search(self):
        curved = self._player("scenarios/players/player8/curved.json")
        curved._search_room = RoomSearch(curved.room)
        curved._best_shape = Shape(FREE, (), (), (), 532.13, 94.0)

        self.assertTrue(curved._small_stock_search_worthwhile())


class PartialSupplyTests(unittest.TestCase):
    def test_partial_face_filter_does_not_require_a_gate_yet(self):
        scenario = read_scenario("scenarios/players/player8/curved.json")
        assigner = Assigner(scenario.inventory, want_second_gate=True)

        self.assertTrue(assigner.can_supply((5,)))
        self.assertIsNone(assigner.assign((5,)))
        self.assertFalse(assigner.can_supply((30, 30, 30, 30)))

    def test_diverse_beam_preserves_different_topology_resources(self):
        scenario = read_scenario("scenarios/players/player8/curved.json")
        assigner = Assigner(scenario.inventory, want_second_gate=True)
        walker = Walker(
            RoomSearch(scenario.room),
            assigner.lengths,
            assigner.total_length,
            8,
            8,
            scenario.weights,
            assigner.can_supply,
            diverse=True,
        )
        base = Walk(((0.0, 0.0),), 0, 0, (0, 0, 0, 0), (), (), 0, 0)
        candidates = [
            (100.0, base.__class__(base.points, 0, 0, base.lattice, (24,), (), 0, 0)),
            (
                90.0,
                base.__class__(base.points, 0, 0, base.lattice, (8, 8, 8), (), 0, 2),
            ),
        ]

        selected = walker._select(candidates, 2)

        self.assertEqual({walk.diagonals for walk in selected}, {0, 2})


if __name__ == "__main__":
    unittest.main()
