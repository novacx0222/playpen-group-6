"""Resource-aware, bounded diversity for partial beam outlines."""

import unittest
from itertools import permutations

from shapely.geometry import box

from players.player6.diagonal_search import DIRS, SQ, Chain, DiagonalSearch
from src.inventory import Inventory
from src.pieces import ConnectorType as CT
from src.room import Room
from src.weights import Weights


def make_chain(faces):
    faces = tuple(faces)
    position = (0, 0, 0, 0)
    turn = 0
    vertices = [(0.0, 0.0)]
    for length, next_turn in faces:
        position = tuple(a + length * b for a, b in zip(position, DIRS[turn % 8]))
        xm, xn, ym, yn = position
        vertices.append((xm + xn * SQ, ym + yn * SQ))
        turn += next_turn
    return Chain(
        position,
        turn,
        faces,
        vertices,
        sum(length for length, _ in faces),
        sum(abs(t) == 2 for _, t in faces),
        sum(abs(t) == 1 for _, t in faces),
        sum(length // 5 - 1 for length, _ in faces),
    )


class ChainDedupTests(unittest.TestCase):
    def setUp(self):
        self.search = DiagonalSearch(
            Room(box(-100, -100, 100, 100)),
            Inventory(
                walls={5: 100},
                gates={5: 2},
                connectors={CT.RIGHT: 12, CT.DIAGONAL: 12, CT.STRAIGHT: 100},
            ),
            Weights(A=2, C=-3, G=100),
        )
        # Same endpoint, direction, edge lengths and connector consumption,
        # but distinct actual paths. The first two tie above the third.
        self.faces = (
            ((5, 2), (5, -2), (5, 1), (5, 1)),
            ((5, 1), (5, 1), (5, -2), (5, 2)),
            ((5, 2), (5, -1), (5, -1), (5, 2)),
        )

    def test_different_connector_budgets_are_separate_states(self):
        first = make_chain(((5, 1), (5, 1), (5, 1)))
        second = make_chain(((5, 2), (5, -1), (5, 2)))
        self.assertEqual(first.pos, second.pos)
        self.assertEqual(first.turn, second.turn)
        self.assertEqual(
            sorted(length for length, _ in first.faces),
            sorted(length for length, _ in second.faces),
        )
        self.assertEqual((first.right, first.diag), (0, 3))
        self.assertEqual((second.right, second.diag), (2, 1))
        groups = {}
        self.search._keep_chain(groups, first)
        self.search._keep_chain(groups, second)
        self.assertEqual(len(groups), 2)
        self.assertEqual(sum(len(group) for group in groups.values()), 2)

    def test_equal_score_distinct_outlines_both_survive(self):
        first, second = map(make_chain, self.faces[:2])
        self.assertNotEqual(first.verts, second.verts)
        groups = {}
        self.search._keep_chain(groups, first)
        self.search._keep_chain(groups, second)
        self.assertEqual(len(groups), 1)
        kept = next(iter(groups.values()))
        self.assertAlmostEqual(kept[0][0], kept[1][0])
        self.assertEqual({chain.faces for _, chain in kept}, set(self.faces[:2]))

    def test_only_two_best_distinct_outlines_survive_in_any_insertion_order(self):
        for order in permutations(self.faces):
            with self.subTest(order=order):
                groups = {}
                for faces in order:
                    self.search._keep_chain(groups, make_chain(faces))
                self.assertEqual(len(groups), 1)
                kept = next(iter(groups.values()))
                self.assertEqual(len(kept), 2)
                self.assertEqual(
                    {chain.faces for _, chain in kept}, set(self.faces[:2])
                )

    def test_duplicate_outline_does_not_use_second_slot(self):
        groups = {}
        for _ in range(3):
            self.search._keep_chain(groups, make_chain(self.faces[0]))
        self.assertEqual(len(next(iter(groups.values()))), 1)
        self.search._keep_chain(groups, make_chain(self.faces[1]))
        self.assertEqual(len(next(iter(groups.values()))), 2)

    def test_global_beam_width_still_bounds_survivors(self):
        groups = {}
        for scale in (1, 2, 3):
            for faces in self.faces[:2]:
                self.search._keep_chain(
                    groups,
                    make_chain([(length * scale, turn) for length, turn in faces]),
                )
        candidates = [candidate for group in groups.values() for candidate in group]
        self.assertEqual(len(candidates), 6)
        self.search._width = 3
        self.search._truncated = False
        beam = self.search.select_beam(candidates)
        self.assertEqual(len(beam), 3)
        self.assertTrue(self.search._truncated)

    def test_fixed_outline_cap_does_not_trigger_unproductive_widening(self):
        groups = {}
        self.search._truncated = False
        self.search._width = 2
        for faces in self.faces:
            self.search._keep_chain(groups, make_chain(faces))
        candidates = [candidate for group in groups.values() for candidate in group]
        self.assertEqual(len(self.search.select_beam(candidates)), 2)
        self.assertFalse(self.search._truncated)


if __name__ == "__main__":
    unittest.main()
