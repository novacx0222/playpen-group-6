"""Improve complete enclosures without searching through unfinished chains.

The search changes the lengths of existing faces or adds a rectangular extension
to part of a face. Each proposed outline is checked against the room before the
stock allocator tries to build it. Several different outlines remain available
as parents, so one locally good enclosure does not end the search.
"""

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass
from itertools import count

from shapely.geometry import Polygon
from shapely.prepared import prep

from players.player5.packing import PolygonPacker
from src.constants import BASELINE_SCORE, MAX_FACE_LENGTH, MIN_WALL_LENGTH, TOL
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import ConnectorType

BEAM_WIDTH = 10
CANDIDATES_PER_PARENT = 80
PACKING_SLICE = 0.08
GEOMETRY_SLICE = 0.45
_EPSILON = 1e-6


@dataclass
class _SearchShape:
    """A complete outline, its measured score, and whether its moves were tried."""

    vertices: tuple
    score: float
    area: float
    perimeter: float
    expanded: bool = False


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _subtract(a, b):
    return a[0] - b[0], a[1] - b[1]


def _advance(point, direction, distance):
    return point[0] + direction[0] * distance, point[1] + direction[1] * distance


def _simplify_outline(vertices):
    """Remove piece junctions on straight faces and use counterclockwise order."""
    points = []
    for point in vertices:
        point = tuple(point)
        if not points or math.dist(points[-1], point) > _EPSILON:
            points.append(point)
    if len(points) > 1 and math.dist(points[0], points[-1]) < _EPSILON:
        points.pop()
    changed = True
    while changed and len(points) >= 3:
        changed = False
        for index, point in enumerate(points):
            incoming = _subtract(point, points[index - 1])
            outgoing = _subtract(points[(index + 1) % len(points)], point)
            scale = math.hypot(*incoming) * math.hypot(*outgoing)
            # Opposite directions would make the boundary double back.
            if (
                scale
                and abs(_cross(incoming, outgoing)) <= _EPSILON * scale
                and incoming[0] * outgoing[0] + incoming[1] * outgoing[1] > 0
            ):
                points.pop(index)
                changed = True
                break
    signed_area = sum(_cross(a, b) for a, b in zip(points, points[1:] + points[:1]))
    if signed_area < 0:
        points.reverse()
    return tuple(points)


def _outline_key(vertices):
    """The same outline has one key regardless of its starting vertex."""
    rounded = tuple((round(x, 6), round(y, 6)) for x, y in vertices)
    first = min(range(len(rounded)), key=rounded.__getitem__)
    return rounded[first:] + rounded[:first]


def _measure_faces(vertices):
    edges = [_subtract(b, a) for a, b in zip(vertices, vertices[1:] + vertices[:1])]
    lengths = [math.hypot(*edge) for edge in edges]
    if any(
        length < MIN_WALL_LENGTH - TOL or length > MAX_FACE_LENGTH + TOL
        for length in lengths
    ):
        return None
    if any(abs(length - round(length)) > TOL for length in lengths):
        return None
    directions = [
        (edge[0] / length, edge[1] / length) for edge, length in zip(edges, lengths)
    ]
    for previous, current in zip(directions[-1:] + directions[:-1], directions):
        angle = abs(
            math.degrees(
                math.atan2(
                    _cross(previous, current),
                    previous[0] * current[0] + previous[1] * current[1],
                )
            )
        )
        if min(abs(angle - 45), abs(angle - 90)) > 0.001:
            return None
    return [round(length) for length in lengths], directions


def _slide_face(vertices, lengths, directions, index):
    """Change three consecutive face lengths while their two endpoints stay fixed.

    Choose a new integer length for the first face, then solve the two remaining
    lengths. For rectangular corners this slides the middle face outward or
    inward. The same calculation also handles allowed diagonal directions.
    """
    size = len(vertices)
    first, second, third = (directions[(index + offset) % size] for offset in range(3))
    determinant = _cross(second, third)
    if abs(determinant) < _EPSILON:
        return
    start, end = vertices[index], vertices[(index + 3) % size]
    alternatives = sorted(
        range(MIN_WALL_LENGTH, MAX_FACE_LENGTH + 1),
        key=lambda length: -abs(length - lengths[index]),
    )
    for first_length in alternatives:
        if first_length == lengths[index]:
            continue
        corner = _advance(start, first, first_length)
        remaining = _subtract(end, corner)
        second_length = _cross(remaining, third) / determinant
        third_length = _cross(second, remaining) / determinant
        if any(
            length < MIN_WALL_LENGTH - TOL
            or length > MAX_FACE_LENGTH + TOL
            or abs(length - round(length)) > TOL
            for length in (second_length, third_length)
        ):
            continue
        changed = list(vertices)
        changed[(index + 1) % size] = corner
        changed[(index + 2) % size] = _advance(corner, second, round(second_length))
        yield changed


def _extensions(vertices, lengths, directions, index):
    """Attach a rectangle to an entire face or a buildable portion of it."""
    start, length, direction = vertices[index], lengths[index], directions[index]
    outward = direction[1], -direction[0]
    widths = sorted(
        {length, length - 5, length - 10, 5, 10, 15, 20, 25, 30}, reverse=True
    )
    for width in widths:
        if not MIN_WALL_LENGTH <= width <= length:
            continue
        spare = length - width
        offsets = sorted({0, spare, spare // 2})
        for offset in offsets:
            if 0 < offset < MIN_WALL_LENGTH or 0 < spare - offset < MIN_WALL_LENGTH:
                continue
            near = _advance(start, direction, offset)
            far = _advance(near, direction, width)
            for depth in range(MAX_FACE_LENGTH, MIN_WALL_LENGTH - 1, -1):
                replacement = [
                    near,
                    _advance(near, outward, depth),
                    _advance(far, outward, depth),
                    far,
                ]
                yield (
                    list(vertices[: index + 1])
                    + replacement
                    + list(vertices[index + 1 :])
                )


def _candidate_outlines(
    state, usable_room, prepared_room, weights, feasible_lengths, stock_limits, deadline
):
    """Keep the best room-fitting geometric moves before allocating any stock."""
    deadline = min(deadline, time.process_time() + GEOMETRY_SLICE)
    total_length, right_available, diagonal_available = stock_limits
    data = _measure_faces(state.vertices)
    if data is None:
        return []
    lengths, directions = data
    bounds = usable_room.bounds
    candidates = []
    unique = set()
    sequence = count()
    # Visit every face and both move families before taking their second move.
    # A complex outline must not spend its entire slice on its first edge.
    generators = deque(
        generator
        for index in range(len(state.vertices))
        for generator in (
            _slide_face(state.vertices, lengths, directions, index),
            _extensions(state.vertices, lengths, directions, index),
        )
    )
    while generators and time.process_time() < deadline:
        generator = generators.popleft()
        try:
            raw = next(generator)
        except StopIteration:
            continue
        generators.append(generator)
        vertices = _simplify_outline(raw)
        face_data = _measure_faces(vertices)
        if (
            len(vertices) < 3
            or face_data is None
            or any(length not in feasible_lengths for length in face_data[0])
        ):
            continue
        face_lengths, face_directions = face_data
        right_corners = sum(
            abs(previous[0] * current[0] + previous[1] * current[1]) < _EPSILON
            for previous, current in zip(
                face_directions[-1:] + face_directions[:-1], face_directions
            )
        )
        if (
            sum(face_lengths) > total_length + TOL
            or right_corners > right_available
            or len(vertices) - right_corners > diagonal_available
        ):
            continue
        identity = _outline_key(vertices)
        if identity in unique:
            continue
        unique.add(identity)
        if any(
            x < bounds[0] or y < bounds[1] or x > bounds[2] or y > bounds[3]
            for x, y in vertices
        ):
            continue
        polygon = Polygon(vertices)
        optimistic = (
            BASELINE_SCORE
            + weights.A * polygon.area
            + weights.C * polygon.length
            + weights.G
        )
        if optimistic <= state.score + _EPSILON:
            continue
        if len(candidates) >= CANDIDATES_PER_PARENT and optimistic <= candidates[0][0]:
            continue
        if not polygon.is_valid or not prepared_room.contains(polygon):
            continue
        candidate = optimistic, next(sequence), vertices
        if len(candidates) < CANDIDATES_PER_PARENT:
            heapq.heappush(candidates, candidate)
        else:
            heapq.heapreplace(candidates, candidate)
    return sorted(candidates, reverse=True)


def _different_states(states, limit=BEAM_WIDTH):
    """Keep good alternatives without filling the frontier with one tiny change."""
    ranked = sorted(states, key=lambda state: state.score, reverse=True)
    selected, cells = [], set()
    for state in ranked:
        polygon = Polygon(state.vertices)
        center = polygon.centroid
        cell = (
            round(center.x / 5),
            round(center.y / 5),
            len(state.vertices),
            round(polygon.area / 50),
        )
        if cell not in cells:
            selected.append(state)
            cells.add(cell)
        if len(selected) == limit:
            return selected
    selected_ids = {id(state) for state in selected}
    selected.extend(state for state in ranked if id(state) not in selected_ids)
    return selected[:limit]


def refine(room, inventory, weights, seeds, deadline, consider):
    """Improve outlines with face moves and rectangular extensions."""
    search_improvements(
        room,
        inventory,
        weights,
        seeds,
        deadline,
        consider,
        candidates=_candidate_outlines,
        select_states=_different_states,
    )


def search_improvements(
    room, inventory, weights, seeds, deadline, consider, *, candidates, select_states
):
    """Run the shared search for either kind of complete-outline improvement.

    ``seeds`` contains complete constructions. ``consider`` receives every newly
    built improvement immediately. ``candidates`` proposes ranked geometry for
    one parent and limits its own generation time; ``select_states`` retains the
    next frontier. Allocation, validation, scoring and deduplication are shared.
    """
    if time.process_time() >= deadline:
        return
    if isinstance(seeds, Construction):
        seeds = [seeds]
    states, seen = [], set()
    for construction in seeds:
        if time.process_time() >= deadline:
            return
        result = validate_construction(construction, room, inventory)
        if not result.valid:
            continue
        vertices = _simplify_outline(result.vertices)
        identity = _outline_key(vertices)
        if identity not in seen:
            seen.add(identity)
            states.append(
                _SearchShape(
                    vertices,
                    score_construction(result, weights),
                    result.area,
                    result.perimeter,
                )
            )
    states = select_states(states)
    usable_room = room.polygon.buffer(TOL)
    prepared_room = prep(usable_room)
    packer = PolygonPacker(inventory, room, weights, deadline)
    stock_limits = (
        packer.total_length,
        inventory.connectors.get(ConnectorType.RIGHT, 0),
        inventory.connectors.get(ConnectorType.DIAGONAL, 0),
    )
    while time.process_time() < deadline:
        unexpanded = [state for state in states if not state.expanded]
        if not unexpanded:
            break
        parent = max(unexpanded, key=lambda state: state.score)
        parent.expanded = True
        proposals = candidates(
            parent,
            usable_room,
            prepared_room,
            weights,
            packer.feasible_lengths,
            stock_limits,
            deadline,
        )
        children = []
        for _, _, vertices in proposals:
            if time.process_time() >= deadline:
                break
            identity = _outline_key(vertices)
            if identity in seen:
                continue
            seen.add(identity)
            construction = packer.pack(
                vertices, deadline=min(deadline, time.process_time() + PACKING_SLICE)
            )
            if construction is None:
                continue
            result = validate_construction(construction, room, inventory)
            if not result.valid:
                continue
            score = score_construction(result, weights)
            if score > parent.score + _EPSILON:
                consider(construction)
                children.append(
                    _SearchShape(vertices, score, result.area, result.perimeter)
                )
        states = select_states(states + children)
