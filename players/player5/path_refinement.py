"""Improve complete outlines by replacing short paths of whole faces.

Each replacement uses two to five faces, with 45- or 90-degree turns. Integer
lengths make closure much cheaper than a general geometric search: in a frame
aligned with one existing face, horizontal, vertical and the two diagonal
directions each have a separate signed length sum. Matching those four sums
closes the path exactly. Patterns that cannot reconnect are discarded.
"""

import heapq
import math
import time
from collections import deque
from functools import lru_cache
from itertools import count, pairwise, product

from shapely.geometry import Polygon

from players.player5.refinement import (
    _measure_faces,
    _outline_key,
    _simplify_outline,
    search_improvements,
)
from src.constants import BASELINE_SCORE, TOL

_TURNS = (-2, -1, 1, 2)  # Units of 45 degrees.
_SHORTLIST = 64
_MAX_PARENTS = 6
_EPSILON = 1e-6


def _turn_steps(previous, current):
    cross = previous[0] * current[1] - previous[1] * current[0]
    dot = previous[0] * current[0] + previous[1] * current[1]
    return round(math.degrees(math.atan2(cross, dot)) / 45)


@lru_cache(maxsize=128)
def _turn_patterns(total_turn, face_count):
    """Headings relative to the incoming face, preserving the span's total turn."""
    patterns = []

    def extend(headings):
        previous = headings[-1] if headings else 0
        if len(headings) == face_count:
            if total_turn - previous in _TURNS:
                # Limit longer routes to at least three axes to bound the search.
                axes = {heading % 4 for heading in headings}
                if face_count <= 3 or len(axes) >= 3:
                    patterns.append(tuple(heading % 8 for heading in headings))
            return
        remaining = face_count - len(headings)
        for turn in _TURNS:
            heading = previous + turn
            if abs(total_turn - heading) <= 2 * remaining:
                extend(headings + [heading])

    extend([])
    return tuple(patterns)


def _direction_totals(headings, lengths):
    """Net lengths along 0, 45, 90 and 135 degrees; reverse travel subtracts."""
    totals = [0, 0, 0, 0]
    for heading, length in zip(headings, lengths):
        heading %= 8
        totals[heading % 4] += length if heading < 4 else -length
    return tuple(totals)


@lru_cache(maxsize=4096)
def _axis_length_options(signs, target, feasible_lengths):
    """Choose early lengths; the final length on this axis is forced by closure."""
    answers = []
    for early in product(feasible_lengths, repeat=len(signs) - 1):
        remaining = target - sum(sign * length for sign, length in zip(signs, early))
        final = remaining * signs[-1]
        if final in feasible_lengths:
            answers.append(early + (final,))
    return tuple(answers)


def _closing_lengths(headings, target, feasible_lengths):
    groups = [[] for _ in range(4)]
    for index, heading in enumerate(headings):
        groups[heading % 4].append((index, 1 if heading < 4 else -1))
    choices = []
    for axis, group in enumerate(groups):
        if not group:
            if target[axis]:
                return
            continue
        options = _axis_length_options(
            tuple(sign for _, sign in group), target[axis], feasible_lengths
        )
        if not options:
            return
        choices.append((group, options))
    for allocations in product(*(options for _, options in choices)):
        lengths = [0] * len(headings)
        for (group, _), allocation in zip(choices, allocations):
            for (index, _), length in zip(group, allocation):
                lengths[index] = length
        yield tuple(lengths)


def _heading_vectors(incoming):
    return tuple(
        (
            incoming[0] * math.cos(step * math.pi / 4)
            - incoming[1] * math.sin(step * math.pi / 4),
            incoming[0] * math.sin(step * math.pi / 4)
            + incoming[1] * math.cos(step * math.pi / 4),
        )
        for step in range(8)
    )


def _path_area(points):
    return sum(a[0] * b[1] - a[1] * b[0] for a, b in pairwise(points)) / 2


def _replacement_paths(
    state, start, span, face_count, feasible_lengths, face_data, deadline
):
    """Replace part of an outline while keeping its two endpoints fixed."""
    vertices = state.vertices[start:] + state.vertices[:start]
    original_lengths, original_directions = face_data
    lengths = original_lengths[start:] + original_lengths[:start]
    directions = original_directions[start:] + original_directions[:start]
    incoming = directions[-1]
    old_headings = tuple(
        _turn_steps(incoming, direction) % 8 for direction in directions[:span]
    )
    target = _direction_totals(old_headings, lengths[:span])
    around = [incoming] + directions[: span + 1]
    total_turn = sum(_turn_steps(a, b) for a, b in pairwise(around))
    units = _heading_vectors(incoming)
    old_area = _path_area(vertices[: span + 1])
    old_perimeter = sum(lengths[:span])
    for headings in _turn_patterns(total_turn, face_count):
        if time.process_time() >= deadline:
            return
        for new_lengths in _closing_lengths(headings, target, feasible_lengths):
            if time.process_time() >= deadline:
                return
            if (
                face_count == span
                and headings == old_headings
                and new_lengths == tuple(lengths[:span])
            ):
                continue
            path = [vertices[0]]
            for heading, length in zip(headings, new_lengths):
                x, y = path[-1]
                unit_x, unit_y = units[heading]
                path.append((x + length * unit_x, y + length * unit_y))
            if math.dist(path[-1], vertices[span]) > TOL:
                continue
            path[-1] = vertices[span]
            area = state.area - old_area + _path_area(path)
            perimeter = state.perimeter - old_perimeter + sum(new_lengths)
            yield path[:-1] + list(vertices[span:]), area, perimeter


def _candidate_paths(
    state, usable_room, prepared_room, weights, feasible_lengths, stock_limits, deadline
):
    """Rank fitting geometry before performing global inventory allocation."""
    now = time.process_time()
    deadline = min(deadline, now + min(1.5, max(0.3, (deadline - now) / 3)))
    feasible_lengths = tuple(sorted(feasible_lengths))
    total_length, right_available, diagonal_available = stock_limits
    face_data = _measure_faces(state.vertices)
    if face_data is None:
        return []
    bounds = usable_room.bounds
    generators = deque(
        _replacement_paths(
            state, start, span, face_count, feasible_lengths, face_data, deadline
        )
        for face_count in range(2, 6)
        for span in range(1, min(3, len(state.vertices) - 1) + 1)
        for start in range(len(state.vertices))
        # Two integer-length faces with an allowed turn cannot replace one face.
        if not (span == 1 and face_count == 2)
    )
    candidates, unique, sequence = [], set(), count()
    while generators and time.process_time() < deadline:
        generator = generators.popleft()
        try:
            raw, area, perimeter = next(generator)
        except StopIteration:
            continue
        generators.append(generator)
        optimistic = (
            BASELINE_SCORE + weights.A * area + weights.C * perimeter + weights.G
        )
        if optimistic <= state.score + _EPSILON or perimeter > total_length + TOL:
            continue
        if len(candidates) == _SHORTLIST and optimistic <= candidates[0][0]:
            continue
        if any(
            x < bounds[0] or y < bounds[1] or x > bounds[2] or y > bounds[3]
            for x, y in raw
        ):
            continue
        vertices = _simplify_outline(raw)
        identity = _outline_key(vertices)
        if identity in unique:
            continue
        unique.add(identity)
        data = _measure_faces(vertices)
        if data is None:
            continue
        _, directions = data
        right_needed = sum(
            abs(a[0] * b[0] + a[1] * b[1]) < _EPSILON
            for a, b in zip(directions[-1:] + directions[:-1], directions)
        )
        if (
            right_needed > right_available
            or len(vertices) - right_needed > diagonal_available
        ):
            continue
        polygon = Polygon(vertices)
        if not polygon.is_valid or not prepared_room.contains(polygon):
            continue
        entry = optimistic, next(sequence), vertices
        if len(candidates) < _SHORTLIST:
            heapq.heappush(candidates, entry)
        else:
            heapq.heapreplace(candidates, entry)
    return sorted(candidates, reverse=True)


def refine_paths(room, inventory, weights, seeds, deadline, consider):
    """Report valid short-path improvements until an absolute process-CPU deadline."""
    search_improvements(
        room,
        inventory,
        weights,
        seeds,
        deadline,
        consider,
        candidates=_candidate_paths,
        select_states=lambda states: sorted(
            states, key=lambda state: state.score, reverse=True
        )[:_MAX_PARENTS],
    )
