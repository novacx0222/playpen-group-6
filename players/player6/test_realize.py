"""Material assignment correctness, repeated dead states, and cutoff recovery."""

import random
import unittest
from collections import Counter
from itertools import product
from unittest.mock import patch

from shapely.geometry import box

from players.player6.diagonal_search import GATE, WALL, DiagonalSearch
from src.inventory import Inventory
from src.pieces import ConnectorType as CT
from src.room import Room
from src.weights import Weights


def make_search(walls, gates=None, straight=10):
    search = DiagonalSearch(
        Room(box(0, 0, 60, 60)),
        Inventory(walls=walls, gates=gates or {}, connectors={CT.STRAIGHT: straight}),
        Weights(A=2, C=-3, G=100),
        time_budget=10,
    )
    search.prepare_inventory()
    return search


def exhaustive_assignment_exists(search, lengths):
    """Independent small-case oracle: try the full Cartesian product."""
    for candidate in product(*(search.comps.get(length, ()) for length in lengths)):
        if sum(comp[2] for comp in candidate) > search.straight_count:
            continue
        needed = Counter(piece for comp in candidate for piece in comp[0])
        if all(count <= search.stock.get(piece, 0) for piece, count in needed.items()):
            return True
    return False


class RealizeTests(unittest.TestCase):
    def assert_assignment_valid(self, search, lengths, assignment):
        self.assertIsNotNone(assignment)
        self.assertEqual(len(assignment), len(lengths))
        needed = Counter()
        for length, (pieces, gates, straights) in zip(lengths, assignment):
            self.assertEqual(sum(n for _, n in pieces), length)
            self.assertEqual(gates, sum(kind == GATE for kind, _ in pieces))
            self.assertEqual(straights, len(pieces) - 1)
            needed.update(pieces)
        self.assertTrue(
            all(n <= search.stock.get(piece, 0) for piece, n in needed.items())
        )
        self.assertLessEqual(sum(comp[2] for comp in assignment), search.straight_count)

    def test_reordering_preserves_face_alignment_and_inventory(self):
        search = make_search({5: 2, 10: 2}, {5: 1}, straight=3)
        original = search.inventory.copy()
        original_stock = dict(search.stock)
        lengths = (15, 10, 5)
        self.assert_assignment_valid(search, lengths, search.realize(lengths))
        self.assertEqual(search.stock, original_stock)
        self.assertEqual(search.inventory, original)

    def test_equal_length_walls_and_gates_remain_distinct(self):
        search = make_search({5: 1}, {5: 1}, straight=1)
        assignment = search.realize((10,))
        self.assert_assignment_valid(search, (10,), assignment)
        self.assertEqual(
            Counter(assignment[0][0]), Counter({(WALL, 5): 1, (GATE, 5): 1})
        )
        search = make_search({5: 1}, {5: 1}, straight=0)
        self.assertIsNone(search.realize((10,)))

    def test_repeated_pieces_cannot_exceed_stock(self):
        search = make_search({5: 3}, straight=2)
        self.assertIsNone(search.realize((10, 10)))
        search = make_search({5: 4}, straight=2)
        self.assert_assignment_valid(search, (10, 10), search.realize((10, 10)))

    def test_straight_connector_capacity_rejects_long_outline_without_dfs(self):
        search = make_search({5: 10, 10: 1}, straight=2)
        lengths = (15, 15)  # Total stock length suffices, but four pieces cannot.
        self.assertGreaterEqual(search.total_length, sum(lengths))
        with patch.object(
                search, "out_of_time", side_effect=AssertionError("unnecessary DFS")
        ):
            self.assertIsNone(search.realize(lengths))
        self.assertIn(lengths, search._realize_cache)

    def test_straight_connector_capacity_keeps_exact_boundary_feasible(self):
        search = make_search({5: 2, 10: 1}, straight=1)
        self.assert_assignment_valid(search, (10, 10), search.realize((10, 10)))

    def test_memoized_dead_states_allow_exhaustion_with_four_nodes(self):
        # Three 10-unit faces need more usable stock than this inventory has.
        # Swapping a 10 wall and a pair of 5 walls reaches the same dead state.
        search = make_search({5: 3, 10: 1, 6: 10})
        search.REALIZE_NODE_LIMIT = 4
        lengths = (10, 10, 10)
        self.assertIsNone(search.realize(lengths))
        self.assertIn(lengths, search._realize_cache)  # Exhaustive, not a cutoff.
        with patch.object(
                search, "out_of_time", side_effect=AssertionError("cache miss")
        ):
            self.assertIsNone(search.realize(lengths))

    def test_node_cutoff_is_retryable(self):
        search = make_search({5: 4}, straight=2)
        search.REALIZE_NODE_LIMIT = 1
        lengths = (10, 10)
        self.assertIsNone(search.realize(lengths))
        self.assertNotIn(lengths, search._realize_cache)
        search.REALIZE_NODE_LIMIT = 4
        self.assert_assignment_valid(search, lengths, search.realize(lengths))

    def test_cutoff_is_deferred_until_next_beam_pass(self):
        search = make_search({5: 4}, straight=2)
        search.REALIZE_NODE_LIMIT = 1
        search._deferred_realize = set()
        lengths = (10, 10)
        self.assertIsNone(search.realize(lengths))
        self.assertNotIn(lengths, search._realize_cache)
        search.REALIZE_NODE_LIMIT = 4
        with patch.object(
                search, "out_of_time", side_effect=AssertionError("restarted in same pass")
        ):
            self.assertIsNone(search.realize(lengths))
        search._deferred_realize = set()  # The next widening pass can retry.
        self.assert_assignment_valid(search, lengths, search.realize(lengths))

    def test_deferred_gate_query_does_not_block_unconstrained_assignment(self):
        search = make_search({5: 2}, {5: 2}, straight=2)
        search._deferred_realize = set()
        lengths = (10, 10)
        self.assertIsNone(search.realize(lengths, min_gate_faces=2, stop=lambda: True))
        self.assertNotIn((lengths, 2), search._gated_realize_cache)
        self.assert_assignment_valid(search, lengths, search.realize(lengths))
        search._deferred_realize = set()
        assignment = search.realize(lengths, min_gate_faces=2)
        self.assert_assignment_valid(search, lengths, assignment)
        self.assertEqual(sum(comp[1] > 0 for comp in assignment), 2)

    def test_timeout_does_not_poison_cache_or_mutate_stock(self):
        search = make_search({5: 4}, straight=2)
        original = dict(search.stock)
        lengths = (10, 10)
        # Interrupt after a branch has deducted material, rather than at entry.
        with patch.object(search, "out_of_time", side_effect=[False, True]):
            self.assertIsNone(search.realize(lengths))
        self.assertNotIn(lengths, search._realize_cache)
        self.assertEqual(search.stock, original)
        self.assert_assignment_valid(search, lengths, search.realize(lengths))

    def test_small_inventories_match_exhaustive_oracle(self):
        rng = random.Random(617)
        for _ in range(80):
            search = make_search(
                {5: rng.randrange(4), 6: rng.randrange(4)},
                {5: rng.randrange(3)},
                straight=rng.randrange(5),
            )
            lengths = tuple(rng.choices((5, 6, 10, 11, 12), k=rng.randrange(5)))
            with self.subTest(
                    stock=search.stock, lengths=lengths, straight=search.straight_count
            ):
                expected = exhaustive_assignment_exists(search, lengths)
                actual = search.realize(lengths)
                self.assertEqual(actual is not None, expected)
                if expected:
                    self.assert_assignment_valid(search, lengths, actual)


if __name__ == "__main__":
    unittest.main()
