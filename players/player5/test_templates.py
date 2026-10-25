"""Geometry and stock regressions for complete-shape search."""

import math
import time
import unittest
from pathlib import Path

from shapely import union_all
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box

from src.enclosure import score_construction, validate_construction
from src.inventory import Inventory
from src.pieces import ConnectorType
from src.room import Room
from src.scenario import read_scenario
from src.weights import Weights

from .packing import PolygonPacker
from .templates import Placement, Shape, _shapes, search_templates

ROOT = Path(__file__).resolve().parents[2]


class TemplateTests(unittest.TestCase):
    def test_stock_filter_face_totals_match_the_generated_geometry(self):
        shapes = [
            Shape("rectangle", (20, 15), 0, 0),
            Shape("hexagon", (20, 15, 8), 0, 0),
            Shape("octagon", (20, 15, 6, 8), 0, 0),
            Shape("opposite corner cuts", (40, 30, 12, 6), 0, 0),
            Shape("four corner cuts", (48, 36, 8, 10, 5, 5), 0, 0),
            Shape("staggered rectangle", (30, 12, 28, 14, 8, 30), 0, 0),
        ]
        for shape in shapes:
            points = shape.vertices()
            actual = sorted(
                round(math.dist(start, end))
                for start, end in zip(points, points[1:] + points[:1])
            )
            self.assertEqual(sorted(shape.face_lengths()), actual, shape.family)

    def test_different_diagonal_lengths_preserve_exact_integer_faces(self):
        shape = Shape("octagon", (30, 20, 6, 8), 0, 128)
        points = shape.vertices()
        lengths = [
            math.dist(start, end) for start, end in zip(points, points[1:] + points[:1])
        ]
        self.assertEqual([round(length) for length in lengths], [30, 6, 20, 8] * 2)
        self.assertAlmostEqual(Polygon(points).area, 1142.974746830583)
        self.assertLess(max(x for x, _ in points), 40)
        self.assertLess(max(y for _, y in points), 30)

    def test_four_corner_cuts_preserve_box_perimeter(self):
        shape = Shape("four corner cuts", (48, 36, 8, 10, 5, 5), 1548, 168)
        polygon = Polygon(shape.vertices())
        self.assertTrue(polygon.is_valid)
        self.assertAlmostEqual(polygon.area, shape.area)
        self.assertAlmostEqual(polygon.length, shape.perimeter)

    def test_opposite_corner_cuts_preserve_box_perimeter(self):
        shape = Shape("opposite corner cuts", (48, 36, 18, 6), 1512, 168)
        polygon = Polygon(shape.vertices())
        self.assertTrue(polygon.is_valid)
        self.assertAlmostEqual(polygon.area, shape.area)
        self.assertAlmostEqual(polygon.length, shape.perimeter)

    def test_staggered_rectangle_preserves_area_with_fewer_corner_connectors(self):
        scenario = read_scenario(
            str(ROOT / "scenarios/players/player4/easy_rectangle_generous.json")
        )
        shape = Shape("staggered rectangle", (30, 12, 30, 12, 8, 30), 0, 160)
        polygon = Polygon(shape.vertices())
        self.assertTrue(polygon.is_valid)
        self.assertAlmostEqual(polygon.area, 1531.5290039756344)
        self.assertAlmostEqual(polygon.length, 160)
        candidate = PolygonPacker(
            scenario.inventory, scenario.room, scenario.weights
        ).pack(shape.vertices(), deadline=time.process_time() + 1)
        self.assertIsNotNone(candidate)
        result = validate_construction(candidate, scenario.room, scenario.inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(
            score_construction(result, scenario.weights), 3683.0580079512692
        )
        self.assertEqual(
            sum(
                connector.connector_type == ConnectorType.RIGHT
                for connector in candidate.connectors
            ),
            4,
        )
        self.assertEqual(
            sum(
                connector.connector_type == ConnectorType.DIAGONAL
                for connector in candidate.connectors
            ),
            4,
        )

    def test_convex_translation_finds_boundary_fit_after_rotation(self):
        polygon = translate(
            rotate(Polygon([(0, 0), (40, 0), (40, 30), (0, 30)]), 23), 71, -42
        )
        shape = Shape("octagon", (30, 20, 6, 8), 0, 128)
        placements = list(Placement(polygon, 23).fit(shape.vertices()))
        self.assertTrue(placements)
        self.assertTrue(
            all(polygon.buffer(1e-4).covers(Polygon(points)) for points in placements)
        )

    def test_simple_rectangle_uses_two_diagonal_sizes(self):
        scenario = read_scenario(str(ROOT / "scenarios/simple_rectangle.json"))
        candidate = search_templates(
            scenario.room,
            scenario.inventory,
            scenario.weights,
            time.process_time() + 2,
            lambda construction: None,
        )
        self.assertIsNotNone(candidate)
        result = validate_construction(candidate, scenario.room, scenario.inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertGreaterEqual(score_construction(result, scenario.weights), 3001.94)

    def test_generous_room_uses_asymmetric_stock_feasible_cut_widths(self):
        scenario = read_scenario(
            str(ROOT / "scenarios/players/player4/easy_rectangle_generous.json")
        )
        candidate = search_templates(
            scenario.room,
            scenario.inventory,
            scenario.weights,
            time.process_time() + 2,
            lambda construction: None,
        )
        self.assertIsNotNone(candidate)
        result = validate_construction(candidate, scenario.room, scenario.inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertGreaterEqual(
            score_construction(result, scenario.weights), 3692 - 1e-6
        )

    def test_expired_deadline_does_not_search(self):
        scenario = read_scenario(str(ROOT / "scenarios/simple_rectangle.json"))
        candidate = search_templates(
            scenario.room,
            scenario.inventory,
            scenario.weights,
            time.process_time() - 1,
            lambda construction: self.fail("expired search reported a candidate"),
        )
        self.assertIsNone(candidate)

    def test_dimensions_use_the_official_containment_tolerance(self):
        side = 19.99986
        room = Room(Polygon([(0, 0), (side, 0), (side, side), (0, side)]))
        inventory = Inventory(
            walls={20: 3},
            gates={20: 1},
            connectors={ConnectorType.RIGHT: 4},
        )
        weights = Weights(1, -1, 0)
        candidate = search_templates(
            room, inventory, weights, time.process_time() + 1, lambda construction: None
        )
        self.assertIsNotNone(candidate)
        result = validate_construction(candidate, room, inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.area, 400)

    def test_nonconvex_placement_finds_a_fit_between_corner_alignments(self):
        scenario = read_scenario(str(ROOT / "scenarios/players/player8/curved.json"))
        shape = Shape("octagon", (6, 8, 25, 8), 0, 0)
        placements = list(Placement(scenario.room.polygon, 0).fit(shape.vertices()))
        self.assertTrue(placements)
        self.assertTrue(
            all(
                scenario.room.polygon.buffer(1e-4).covers(Polygon(points))
                for points in placements
            )
        )
        packer = PolygonPacker(scenario.inventory, scenario.room, scenario.weights)
        candidate = packer.pack(placements[0], deadline=time.process_time() + 1)
        self.assertIsNotNone(candidate)
        result = validate_construction(candidate, scenario.room, scenario.inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertGreaterEqual(score_construction(result, scenario.weights), 1987.3666)

    def test_nonconvex_placement_preserves_a_tight_boundary_contact(self):
        half_width = (10 + 10 / math.sqrt(2)) / 2
        polygon = union_all(
            [
                box(-70, -half_width, 120, half_width),
                box(-half_width, -80, half_width, 100),
            ]
        )
        room = Room(polygon)
        inventory = Inventory(
            walls={10: 6},
            gates={10: 2},
            connectors={ConnectorType.DIAGONAL: 8},
        )
        weights = Weights(2, -3, 100)
        shape = Shape("octagon", (10, 10, 10, 10), 0, 0)
        placements = list(Placement(polygon, 0).fit(shape.vertices()))
        self.assertTrue(placements)
        self.assertTrue(
            all(polygon.buffer(1e-4).covers(Polygon(points)) for points in placements)
        )
        candidate = PolygonPacker(inventory, room, weights).pack(placements[0])
        self.assertIsNotNone(candidate)
        result = validate_construction(candidate, room, inventory)
        self.assertTrue(result.valid, result.reason)

    def test_opposite_corner_cuts_do_not_overlap(self):
        shapes = _shapes(
            "opposite corner cuts", [7, 8], 15, 15, time.process_time() + 1
        )
        dimensions = {shape.dimensions for shape in shapes}
        self.assertNotIn((15, 15, 8, 8), dimensions)
        for shape in _shapes(
            "opposite corner cuts", [7, 8], 15, 15, time.process_time() + 1
        ):
            self.assertTrue(Polygon(shape.vertices()).is_valid)

    def test_curved_room_keeps_stock_feasible_chamfered_shapes(self):
        scenario = read_scenario(str(ROOT / "scenarios/players/player8/curved.json"))
        candidate = search_templates(
            scenario.room,
            scenario.inventory,
            scenario.weights,
            time.process_time() + 10,
            lambda construction: None,
        )
        self.assertIsNotNone(candidate)
        result = validate_construction(candidate, scenario.room, scenario.inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertGreaterEqual(score_construction(result, scenario.weights), 1987.3666)


if __name__ == "__main__":
    unittest.main()
