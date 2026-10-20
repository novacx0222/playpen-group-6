"""Closed-candidate gate bonuses, bounded fallbacks, and incumbent pruning."""

import random
import unittest
from collections import Counter
from itertools import product
from unittest.mock import patch

from shapely.geometry import box

from players.player6.diagonal_search import DiagonalSearch
from players.player6.test_realize import make_search
from src.enclosure import score_construction, validate_construction
from src.inventory import Inventory
from src.pieces import ConnectorType as CT
from src.room import Room
from src.weights import Weights

LENGTHS = (13, 13, 15, 15)
FACES = ((13, 2), (15, 2), (13, 2), (15, 2))


def bonus_search(**kwargs):
    search = DiagonalSearch(
        Room(box(0, 0, 60, 60)),
        Inventory(
            walls={5: 3, 7: 3, 8: 3},
            gates={6: 2},
            connectors={CT.RIGHT: 4, CT.STRAIGHT: 20},
        ),
        Weights(A=2, C=-3, G=100),
        time_budget=1,
        **kwargs,
    )
    search.prepare_inventory()
    return search


def result_for(search, assignment):
    pool = {}
    for length, comp in zip(LENGTHS, assignment):
        pool.setdefault(length, []).append(comp)
    construction = search.to_construction(
        FACES, [pool[length].pop(0) for length, _ in FACES], (None, (0, 0, 0))
    )
    return validate_construction(construction, search.room, search.inventory)


def one_closed_candidate(chain, closed):
    closed.append((FACES, 56))
    return []


class GateAssignmentTests(unittest.TestCase):
    def test_same_outline_can_earn_second_gate_bonus(self):
        search = bonus_search()
        original_inventory = search.inventory.copy()
        baseline = search.realize(LENGTHS)
        before = result_for(search, baseline)
        improved = search._improve_gate_assignment(LENGTHS, baseline)
        after = result_for(search, improved)
        self.assertTrue(before.valid and after.valid)
        self.assertFalse(before.second_gate_bonus)
        self.assertTrue(after.second_gate_bonus)
        self.assertAlmostEqual(after.area, before.area)
        self.assertAlmostEqual(after.perimeter, before.perimeter)
        self.assertAlmostEqual(score_construction(before, search.weights), 1222)
        self.assertAlmostEqual(score_construction(after, search.weights), 1322)
        self.assertEqual(search.inventory, original_inventory)
        self.assertIs(search.realize(LENGTHS), baseline)

    def test_two_gate_pieces_on_one_face_do_not_qualify(self):
        search = make_search({12: 1, 5: 2}, {6: 2}, straight=1)
        lengths = (5, 5, 12, 12)
        baseline = search.realize(lengths, min_gate_faces=1)
        self.assertIsNotNone(baseline)
        self.assertEqual(sum(comp[1] for comp in baseline), 2)
        self.assertEqual(sum(comp[1] > 0 for comp in baseline), 1)
        self.assertIsNone(search.realize(lengths, min_gate_faces=2))
        self.assertIs(search._improve_gate_assignment(lengths, baseline), baseline)
        self.assertIs(search.realize(lengths, min_gate_faces=1), baseline)

    def test_cutoff_does_not_poison_gate_cache(self):
        search = bonus_search()
        baseline = search.realize(LENGTHS)
        self.assertIsNone(search.realize(LENGTHS, min_gate_faces=2, stop=lambda: True))
        self.assertNotIn((LENGTHS, 2), search._gated_realize_cache)
        search.REALIZE_NODE_LIMIT = 1
        self.assertIs(search._improve_gate_assignment(LENGTHS, baseline), baseline)
        self.assertNotIn((LENGTHS, 2), search._gated_realize_cache)
        search.REALIZE_NODE_LIMIT = 4000
        improved = search._improve_gate_assignment(LENGTHS, baseline)
        self.assertTrue(result_for(search, improved).second_gate_bonus)

    def test_exhausted_bonus_quota_keeps_baseline_but_reuses_known_plan(self):
        search = bonus_search()
        baseline = search.realize(LENGTHS)
        search._gate_bonus_cpu = search.TIME_BUDGET * search.GATE_BONUS_FRACTION
        with patch.object(
                search, "realize", side_effect=AssertionError("quota exceeded")
        ):
            self.assertIs(search._improve_gate_assignment(LENGTHS, baseline), baseline)
        known = search.realize(LENGTHS, min_gate_faces=2)
        self.assertIs(search._improve_gate_assignment(LENGTHS, baseline), known)

    def test_gate_constraints_match_exhaustive_small_inventories(self):
        rng = random.Random(318)
        for _ in range(60):
            search = make_search(
                {5: rng.randrange(4), 6: rng.randrange(4)},
                {5: rng.randrange(4)},
                straight=rng.randrange(5),
            )
            lengths = tuple(rng.choices((5, 6, 10, 11, 12), k=rng.randrange(5)))
            for minimum in (1, 2):
                expected = False
                for candidate in product(*(search.comps.get(n, ()) for n in lengths)):
                    if sum(comp[1] > 0 for comp in candidate) < minimum:
                        continue
                    if sum(comp[2] for comp in candidate) > search.straight_count:
                        continue
                    need = Counter(piece for comp in candidate for piece in comp[0])
                    if all(n <= search.stock.get(p, 0) for p, n in need.items()):
                        expected = True
                        break
                with self.subTest(lengths=lengths, stock=search.stock, minimum=minimum):
                    actual = search.realize(lengths, min_gate_faces=minimum)
                    self.assertEqual(actual is not None, expected)
                    if actual is not None:
                        self.assertGreaterEqual(
                            sum(comp[1] > 0 for comp in actual), minimum
                        )
                        need = Counter(piece for comp in actual for piece in comp[0])
                        self.assertTrue(
                            all(n <= search.stock[p] for p, n in need.items())
                        )


class ClosedCandidateTests(unittest.TestCase):
    def test_deferred_closed_candidate_is_revisited_after_widening(self):
        search = bonus_search()
        passes = 0

        def expand(chain, closed):
            nonlocal passes
            passes += 1
            search.REALIZE_NODE_LIMIT = 1 if passes == 1 else 4000
            return one_closed_candidate(chain, closed)

        def select(scored):
            search._truncated = passes == 1
            return []

        with (
            patch.object(search, "expand", side_effect=expand),
            patch.object(search, "select_beam", side_effect=select),
        ):
            construction = search.build_enclosure()
        result = validate_construction(construction, search.room, search.inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(score_construction(result, search.weights), 1322)
        self.assertEqual(passes, 2)
        self.assertIsNone(search._deferred_realize)

    def test_closed_candidate_is_published_then_improved(self):
        search = bonus_search()
        with patch.object(search, "expand", side_effect=one_closed_candidate):
            construction = search.build_enclosure()
        result = validate_construction(construction, search.room, search.inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertEqual([score for _, score in search.history], [1222, 1322])
        self.assertTrue(result.second_gate_bonus)

    def test_bonus_timeout_retains_valid_baseline(self):
        search = bonus_search()
        with (
            patch.object(search, "expand", side_effect=one_closed_candidate),
            patch.object(
                search, "_improve_gate_assignment", side_effect=DiagonalSearch.OutOfTime
            ),
        ):
            construction = search.build_enclosure()
        result = validate_construction(construction, search.room, search.inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(score_construction(result, search.weights), 1222)
        self.assertIsNone(search._deferred_realize)

    def test_uncompetitive_incumbent_bound_skips_placement(self):
        search = bonus_search(incumbent_score=1400)
        with (
            patch.object(search, "expand", side_effect=one_closed_candidate),
            patch(
                "players.player6.diagonal_search.find_fit",
                side_effect=AssertionError("uncompetitive candidate fitted"),
            ),
        ):
            self.assertIsNone(search.build_enclosure())
        self.assertEqual(search.history, [])

    def test_candidate_below_incumbent_can_still_improve_with_bonus(self):
        search = bonus_search(incumbent_score=1300)
        with patch.object(search, "expand", side_effect=one_closed_candidate):
            construction = search.build_enclosure()
        result = validate_construction(construction, search.room, search.inventory)
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(score_construction(result, search.weights), 1322)
        self.assertEqual([score for _, score in search.history], [1322])


if __name__ == "__main__":
    unittest.main()
