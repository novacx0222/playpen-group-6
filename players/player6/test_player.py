"""Regression checks: python -m unittest players.player6.test_player."""

import unittest
from collections import Counter
from unittest.mock import patch

from shapely import affinity
from shapely.geometry import Polygon

from players.player6.player import Player6
from players.player6.polygon import find_fit, room_search_angles
from src.enclosure import score_construction, validate_construction
from src.inventory import Inventory
from src.pieces import ConnectorType
from src.room import Room
from src.weights import Weights


def make_shape(face_pieces, turns):
    edges = [
        Player6.Edge(pieces, sum(length for _, length in pieces), len(pieces) - 1)
        for pieces in face_pieces
    ]
    position = (0, 0)
    heading = 0
    vertices = []
    for edge, turn in zip(edges, turns):
        vertices.append(position)
        dx, dy = Player6.STEPS[heading]
        position = (position[0] + dx * edge.length, position[1] + dy * edge.length)
        heading = (heading + turn // 90) % 4
    assert position == (0, 0) and heading == 0
    return Player6.Shape(edges, turns, vertices)


def inventory_for(shape):
    walls = Counter()
    gates = Counter()
    for edge in shape.edges:
        for kind, length in edge.pieces:
            (gates if kind == Player6.GATE else walls)[length] += 1
    return Inventory(
        walls=dict(walls),
        gates=dict(gates),
        connectors={
            ConnectorType.RIGHT: len(shape.edges),
            ConnectorType.STRAIGHT: sum(e.straight_count for e in shape.edges),
        },
    )


class Player6RegressionTests(unittest.TestCase):
    def setUp(self):
        self.room = Room(Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)]))
        self.weights = Weights(A=2, C=-3, G=100)

    def test_concave_turns_and_straight_joins_preserve_geometry(self):
        face_pieces = [
            [(Player6.WALL, 5), (Player6.WALL, 10)],
            [(Player6.GATE, 5)],
            [(Player6.WALL, 5)],
            [(Player6.WALL, 5)],
            [(Player6.WALL, 10)],
            [(Player6.WALL, 5), (Player6.WALL, 5)],
        ]
        for sign in (1, -1):
            shape = make_shape(
                face_pieces, [sign * t for t in [90, 90, -90, 90, 90, 90]]
            )
            for theta in (0, 37, 90):
                with self.subTest(sign=sign, theta=theta):
                    construction = shape.to_construction((10, 20), theta)
                    result = validate_construction(
                        construction, self.room, inventory_for(shape)
                    )
                    self.assertTrue(result.valid, result.reason)
                    expected = affinity.translate(
                        affinity.rotate(shape.to_polygon(), theta, origin=(0, 0)),
                        xoff=10,
                        yoff=20,
                    )
                    self.assertLess(
                        result.polygon.symmetric_difference(expected).area, 1e-7
                    )
                    self.assertEqual(
                        [f.length for f in result.faces], [15, 5, 5, 5, 10, 10]
                    )

    def test_closing_connector_belongs_to_first_vertex(self):
        shape = make_shape(
            [
                [(Player6.GATE, 5), (Player6.WALL, 5)],
                [(Player6.WALL, 5)],
                [(Player6.WALL, 10)],
                [(Player6.WALL, 5)],
            ],
            [90] * 4,
        )
        construction = shape.to_construction((0, 0), 0)
        self.assertEqual(
            [c.turn_angle() for c in construction.connectors], [90, 0, 90, 90, 90]
        )
        result = validate_construction(construction, self.room, inventory_for(shape))
        self.assertTrue(result.valid, result.reason)

    def test_shape_limit_is_strict_and_inventory_is_unchanged(self):
        for limit in (0, 1, 2, 5, 10):
            with self.subTest(limit=limit):
                materials = Player6.Materials(
                    [[Player6.WALL, 5, 3], [Player6.WALL, 10, 2], [Player6.GATE, 5, 1]],
                    6,
                    0,
                )
                original = [row.copy() for row in materials.materials]
                shapes = Player6.generate_shapes(
                    materials, Player6.generate_edges(materials), 6, limit
                )
                self.assertEqual(len(shapes), limit)
                self.assertEqual(materials.materials, original)
                for shape in shapes:
                    result = validate_construction(
                        shape.to_construction((0, 0), 0),
                        self.room,
                        inventory_for(shape),
                    )
                    self.assertTrue(result.valid, result.reason)

    def test_valid_small_rectangle_is_not_rejected_by_missing_baseline(self):
        shape = make_shape([[(Player6.GATE, 5)]] + [[(Player6.WALL, 5)]] * 3, [90] * 4)
        player = Player6(self.room, inventory_for(shape), self.weights)
        theta, tx, ty = 37, 10, 20
        placed = affinity.translate(
            affinity.rotate(shape.to_polygon(), theta, origin=(0, 0)), xoff=tx, yoff=ty
        )
        # Ring order is not used to recover the construction's first piece.
        reordered = Polygon(list(placed.exterior.coords)[:-1][::-1])
        player.best_score = 0
        player.best_construction = None
        player._consider_shape(shape, (reordered, (theta, tx, ty)))
        construction = player.best_construction
        self.assertIsNotNone(construction)
        self.assertAlmostEqual(construction.start_heading, theta)
        self.assertEqual(construction.start, (tx, ty))
        result = validate_construction(construction, self.room, player.inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(score_construction(result, self.weights), 990)

    def test_second_gate_bonus_is_used_to_rank_candidates(self):
        one_gate = make_shape(
            [[(Player6.GATE, 5)]] + [[(Player6.WALL, 5)]] * 3, [90] * 4
        )
        two_gates = make_shape(
            [
                [(Player6.GATE, 5)],
                [(Player6.GATE, 5)],
                [(Player6.WALL, 5)],
                [(Player6.WALL, 5)],
            ],
            [90] * 4,
        )
        inventory = Inventory(
            walls={5: 3}, gates={5: 2}, connectors={ConnectorType.RIGHT: 4}
        )
        player = Player6(self.room, inventory, self.weights)
        player.best_score = 0
        player.best_construction = None
        for shape in (one_gate, two_gates):
            player._consider_shape(shape, (shape.to_polygon(), (0, 0, 0)))
        construction = player.best_construction
        result = validate_construction(construction, self.room, inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertTrue(result.second_gate_bonus)

    def test_invalid_placement_is_not_returned(self):
        shape = make_shape([[(Player6.GATE, 5)]] + [[(Player6.WALL, 5)]] * 3, [90] * 4)
        player = Player6(self.room, inventory_for(shape), self.weights)
        player.best_score = 0
        player.best_construction = None
        player._consider_shape(shape, (shape.to_polygon(), (0, 1000, 1000)))
        self.assertIsNone(player.best_construction)


class IterativeFitRegressionTests(unittest.TestCase):
    def make_player(self, inventory, room=None):
        room = room or Room(Polygon([(0, 0), (40, 0), (40, 40), (0, 40)]))
        player = Player6(room, inventory, Weights(A=2, C=-3, G=100))
        player.possible_faces = {}
        player._fit_cache = {}
        player.best_score = 0
        player.best_shape = None
        player.best_fit = None
        player.best_construction = None
        return player

    def test_one_straight_allows_exactly_two_pieces(self):
        for count in (0, 1):
            with self.subTest(straights=count):
                player = self.make_player(
                    Inventory(
                        walls={5: 4},
                        gates={5: 1},
                        connectors={
                            ConnectorType.RIGHT: 4,
                            ConnectorType.STRAIGHT: count,
                        },
                    )
                )
                player.enumerate_faces(max_length=11)
                self.assertEqual(10 in player.possible_faces, count == 1)

    def test_gate_bonus_counts_distinct_actual_faces(self):
        shape = make_shape(
            [
                [(Player6.GATE, 5), (Player6.GATE, 5)],
                [(Player6.WALL, 5)],
                [(Player6.WALL, 10)],
                [(Player6.WALL, 5)],
            ],
            [90] * 4,
        )
        player = self.make_player(inventory_for(shape))
        score = player._consider_shape(shape, (shape.to_polygon(), (0, 0, 0)))
        self.assertEqual(score, 1010)
        result = validate_construction(
            player.best_construction, player.room, player.inventory
        )
        self.assertFalse(result.second_gate_bonus)

    def test_failed_l_does_not_prune_smaller_feasible_l(self):
        target = Polygon([(0, 0), (5, 0), (5, 8), (15, 8), (15, 15), (0, 15)])
        player = self.make_player(
            Inventory(
                walls={8: 2, 7: 2, 10: 2},
                gates={5: 2},
                connectors={ConnectorType.RIGHT: 6, ConnectorType.STRAIGHT: 2},
            ),
            Room(target),
        )
        player.possible_faces = {
            15: [
                [(Player6.WALL, 8), (Player6.WALL, 7)],
                [(Player6.GATE, 5), (Player6.WALL, 10)],
            ]
        }
        checked = []

        def fit(room, polygon):
            checked.append(polygon)
            return (polygon, (0, 0, 0)) if polygon.equals(target) else None

        with patch("players.player6.player.find_fit", side_effect=fit):
            player.generate_quad_shapes()
        self.assertTrue(any(polygon.equals(target) for polygon in checked))
        self.assertIsNotNone(player.best_construction)
        result = validate_construction(
            player.best_construction, player.room, player.inventory
        )
        self.assertTrue(result.valid, result.reason)

    def test_l_shape_cannot_overspend_straight_connectors(self):
        shape = make_shape(
            [
                [(Player6.GATE, 5)],
                [(Player6.WALL, 5)],
                [(Player6.WALL, 5)],
                [(Player6.WALL, 5)],
                [(Player6.WALL, 5), (Player6.WALL, 5)],
                [(Player6.WALL, 5), (Player6.WALL, 5)],
            ],
            [90, -90, 90, 90, 90, 90],
        )
        inventory = inventory_for(shape)
        inventory.connectors[ConnectorType.STRAIGHT] = 1
        player = self.make_player(inventory)
        self.assertIsNone(
            player._consider_shape(shape, (shape.to_polygon(), (0, 0, 0)))
        )
        self.assertIsNone(player.best_construction)

    def test_gate_only_inventory_and_input_preservation(self):
        inventory = Inventory(
            walls={64: 1}, gates={5: 4, 8: 0}, connectors={ConnectorType.RIGHT: 4}
        )
        player = self.make_player(inventory)
        original = inventory.copy()
        with patch(
            "players.player6.player.find_fit", side_effect=lambda q, p: (p, (0, 0, 0))
        ):
            construction = player.build_enclosure()
        self.assertIsNotNone(construction)
        self.assertEqual(inventory, original)
        self.assertTrue(
            validate_construction(construction, player.room, inventory).valid
        )

    def test_missing_gate_returns_none(self):
        player = self.make_player(
            Inventory(walls={5: 4}, connectors={ConnectorType.RIGHT: 4})
        )
        self.assertIsNone(player.build_enclosure())


class RoomOrientationTests(unittest.TestCase):
    def test_rotated_rectangle_selects_its_actual_axis(self):
        room = affinity.rotate(
            Polygon([(0, 0), (40, 0), (40, 5), (0, 5)]), 20, origin=(0, 0)
        )
        angles = room_search_angles(room)
        expected = [20, 200, 110, 290, 65, 245, 155, 335]
        for actual, wanted in zip(angles, expected):
            self.assertAlmostEqual(actual, wanted)

    def test_axis_clustering_wraps_at_ninety_degrees(self):
        # A nearly vertical side must support a nearly horizontal side's axis.
        room = Polygon([(0, 0), (10, 0), (10.01, 10), (0, 10)])
        self.assertAlmostEqual(room_search_angles(room)[0], 0)

    def test_tight_rotated_corridor_has_valid_fit(self):
        room = affinity.rotate(
            Polygon([(0, 0), (40, 0), (40, 5), (0, 5)]), 20, origin=(0, 0)
        )
        candidate = Polygon([(0, 0), (14, 0), (14, 5), (0, 5)])
        fit = find_fit(room, candidate)
        self.assertIsNotNone(fit)
        placed, (theta, _, _) = fit
        self.assertAlmostEqual(theta, 20)
        self.assertTrue(room.buffer(1e-7).covers(placed))

    def test_inner_fit_never_falls_back_to_per_degree_search(self):
        room = Polygon([(0, 0), (40, 0), (40, 30), (0, 30)])
        candidate = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        with (
            patch("players.player6.polygon.prep") as prepare,
            patch("players.player6.polygon.place_at_angle", return_value=None) as place,
        ):
            prepare.return_value.covers.return_value = False
            self.assertIsNone(
                find_fit(room, candidate, n_angles=360, max_cpu_seconds=5)
            )
        self.assertEqual(place.call_count, 8)
        self.assertEqual(
            [call.args[2] for call in place.call_args_list], room_search_angles(room)
        )


if __name__ == "__main__":
    unittest.main()
