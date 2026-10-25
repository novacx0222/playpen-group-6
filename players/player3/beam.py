"""Bounded beam search whose edges are complete, at-most-30-unit faces.

The score of the polygon made by joining the current endpoint to the start is
only a ranking heuristic.  A temporarily unhelpful addition is NOT proof that
all of its descendants are bad.  Hard pruning below uses geometry, connector
counts, available perimeter, or a conservative area upper bound instead.
"""

import math
import time
from dataclasses import dataclass

from shapely.geometry import LineString, Point
from shapely.prepared import prep

from src.pieces import ConnectorType

from .placement import search_starts


@dataclass(frozen=True, slots=True)
class State:
    points: tuple
    lengths: tuple
    heading: int
    turn: int
    right: int
    diagonal: int
    perimeter: float
    cross_sum: float
    merit: float


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _intersects(a, b, c, d):
    """Inclusive segment intersection, for rejecting non-adjacent contacts."""
    eps = 1e-7
    if (
        max(a[0], b[0]) < min(c[0], d[0]) - eps
        or max(c[0], d[0]) < min(a[0], b[0]) - eps
        or max(a[1], b[1]) < min(c[1], d[1]) - eps
        or max(c[1], d[1]) < min(a[1], b[1]) - eps
    ):
        return False

    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    x, y = side(a, b, c), side(a, b, d)
    u, v = side(c, d, a), side(c, d, b)
    return x * y <= eps and u * v <= eps


def _turn(a, b):
    return (b - a + 4) % 8 - 4


def tapered_width(width, depth, max_faces, compactness):
    """Preserve early alternatives; trade breadth for depth gradually."""
    fraction = min(1.0, max(0.0, (depth - 4) / max(1, max_faces - 5)))
    target = max(4, round(width * (0.25 + 0.75 * compactness)))
    return max(1, round(width + fraction * (target - width)))


def face_beam(
    room,
    inventory,
    catalog,
    weights,
    deadline,
    consider,
    *,
    beam_width=24,
    max_faces=14,
    starts=None,
    boundary_guided=False,
    compactness=1.0,
    expansions_per_start=128,
):
    """Search inside the room, checking exact one/two-face closures throughout.

    ``consider`` validates and scores a complete list of face vertices and
    returns that candidate's score (or None). It retains the overall incumbent.
    """
    stats = {
        "states": 0,
        "expanded": 0,
        "closures": 0,
        "starts": 0,
        "geometry_prunes": 0,
        "resource_prunes": 0,
        "bound_prunes": 0,
        "return_prunes": 0,
        "initial_states": 0,
        "initial_lengths": 0,
        "deepest": 0,
    }
    if not catalog.lengths or time.process_time() >= deadline:
        return stats
    initial_lengths = set()
    lengths = tuple(sorted(catalog.lengths, reverse=True))
    length_set = set(lengths)
    right_max = inventory.connectors.get(ConnectorType.RIGHT, 0)
    diagonal_max = inventory.connectors.get(ConnectorType.DIAGONAL, 0)
    max_faces = min(max_faces, right_max + diagonal_max)
    if max_faces < 4:
        return stats
    legal_turns = tuple(
        t for t in (2, 1, -1, -2) if (right_max if abs(t) == 2 else diagonal_max)
    )
    polygon = room.polygon
    prepared = prep(polygon.buffer(1e-7))
    prepared_boundary = prep(polygon.boundary.buffer(1e-7)) if boundary_guided else None
    minx, miny, maxx, maxy = polygon.bounds
    available_perimeter = catalog.max_perimeter
    incumbent = 0.0
    if starts is None:
        starts = search_starts(room, min(lengths), max_starts=36, deadline=deadline)
    # Fixed expansion counts make branch selection independent of CPU speed.
    # Only the overall deadline can interrupt this deterministic search prefix.
    for start, angle in starts:
        if time.process_time() >= deadline:
            break
        stats["starts"] += 1
        start_expanded = 0
        directions = tuple(
            (
                math.cos(math.radians(angle + 45 * i)),
                math.sin(math.radians(angle + 45 * i)),
            )
            for i in range(8)
        )

        def inside_edge(a, b):
            if b[0] < minx - 1e-7 or b[0] > maxx + 1e-7:
                return False
            if b[1] < miny - 1e-7 or b[1] > maxy + 1e-7:
                return False
            return prepared.covers(LineString((a, b)))

        def merit(points, perimeter, cross_sum, start=start):
            distance = math.dist(points[-1], start)
            area = max(0.0, (cross_sum + _cross(points[-1], start)) / 2)
            return (
                1000
                + weights.A * area
                + weights.C * (perimeter + distance)
                + max(0.0, weights.G)
            )

        closure_best = None

        def record(vertices):
            nonlocal incumbent, closure_best
            stats["closures"] += 1
            score = consider(vertices)
            if score is not None:
                incumbent = max(incumbent, score)
                closure_best = (
                    score if closure_best is None else max(closure_best, score)
                )

        def close(state, start=start, directions=directions, deadline=deadline):
            """Solve for remaining lengths rather than enumerate every piece."""
            count = len(state.lengths)
            end = state.points[-1]
            dx, dy = start[0] - end[0], start[1] - end[1]
            for t in legal_turns:
                heading = (state.heading + t) % 8
                u = directions[heading]
                first = dx * u[0] + dy * u[1]
                final_turn = _turn(heading, 0)
                if (
                    count + 1 <= max_faces
                    and final_turn in legal_turns
                    and state.turn + t + final_turn == 8
                    and abs(dx * u[1] - dy * u[0]) < 1e-6
                    and abs(first - round(first)) < 1e-6
                    and round(first) in length_set
                    and state.perimeter + first <= available_perimeter
                ):
                    r = state.right + (abs(t) == 2) + (abs(final_turn) == 2)
                    d = state.diagonal + (abs(t) == 1) + (abs(final_turn) == 1)
                    if r <= right_max and d <= diagonal_max:
                        record(state.points)
                if count + 2 > max_faces:
                    continue
                for next_turn in legal_turns:
                    last_heading = (heading + next_turn) % 8
                    final_turn = _turn(last_heading, 0)
                    if final_turn not in legal_turns:
                        continue
                    if state.turn + t + next_turn + final_turn != 8:
                        continue
                    r = state.right + sum(
                        abs(v) == 2 for v in (t, next_turn, final_turn)
                    )
                    d = state.diagonal + sum(
                        abs(v) == 1 for v in (t, next_turn, final_turn)
                    )
                    if r > right_max or d > diagonal_max:
                        continue
                    v = directions[last_heading]
                    det = u[0] * v[1] - u[1] * v[0]
                    if abs(det) < 1e-8:
                        continue
                    a = (dx * v[1] - dy * v[0]) / det
                    b = (u[0] * dy - u[1] * dx) / det
                    if abs(a - round(a)) > 1e-6 or abs(b - round(b)) > 1e-6:
                        continue
                    if round(a) not in length_set or round(b) not in length_set:
                        continue
                    if state.perimeter + a + b > available_perimeter:
                        continue
                    middle = (end[0] + a * u[0], end[1] + a * u[1])
                    record(state.points + (middle,))
                    if time.process_time() >= deadline:
                        return

        beam = []
        for length in lengths:
            dx, dy = directions[0]
            end = (start[0] + length * dx, start[1] + length * dy)
            if inside_edge(start, end) and (
                not boundary_guided
                or prepared_boundary.covers(LineString((start, end)))
            ):
                stats["initial_states"] += 1
                initial_lengths.add(length)
                stats["initial_lengths"] = len(initial_lengths)
                pts = (start, end)
                cross_sum = _cross(start, end)
                beam.append(
                    State(
                        pts,
                        (length,),
                        0,
                        0,
                        0,
                        0,
                        length,
                        cross_sum,
                        merit(pts, length, cross_sum),
                    )
                )
        for _depth in range(1, max_faces):
            if (
                not beam
                or start_expanded >= expansions_per_start
                or time.process_time() >= deadline
            ):
                break
            children = {}
            for state in beam:
                if start_expanded >= expansions_per_start:
                    break
                start_expanded += 1
                stats["expanded"] += 1
                stats["deepest"] = max(stats["deepest"], len(state.lengths))
                if not catalog.feasible_prefix(state.lengths):
                    stats["resource_prunes"] += 1
                    continue
                closure_best = None
                if len(state.lengths) >= 2:
                    close(state)
                if time.process_time() >= deadline:
                    break
                # For any completion, area <= room area and <= P^2/(4*pi).
                # With C<0, even an optimistic completion must pay for the
                # current perimeter plus the Euclidean gap back to the start.
                upper_perimeter = min(
                    available_perimeter,
                    state.perimeter + 30 * (max_faces - len(state.lengths)),
                )
                lower_perimeter = state.perimeter + math.dist(state.points[-1], start)
                upper_area = min(
                    polygon.area, upper_perimeter * upper_perimeter / (4 * math.pi)
                )
                bound = 1000 + max(0.0, weights.A) * upper_area
                bound += weights.C * (
                    upper_perimeter if weights.C > 0 else lower_perimeter
                )
                bound += max(0.0, weights.G)
                if bound <= incumbent + 1e-8:
                    stats["bound_prunes"] += 1
                    continue
                for turn in legal_turns:
                    r = state.right + (abs(turn) == 2)
                    d = state.diagonal + (abs(turn) == 1)
                    if r > right_max or d > diagonal_max:
                        stats["resource_prunes"] += 1
                        continue
                    total_turn = state.turn + turn
                    remaining = max_faces - len(state.lengths)
                    if abs(8 - total_turn) > 2 * remaining:
                        continue
                    heading = (state.heading + turn) % 8
                    ux, uy = directions[heading]
                    a = state.points[-1]
                    for length in lengths:
                        stats["states"] += 1
                        perimeter = state.perimeter + length
                        b = (a[0] + length * ux, a[1] + length * uy)
                        distance = math.dist(b, start)
                        if perimeter + distance > available_perimeter + 1e-7:
                            stats["resource_prunes"] += 1
                            continue
                        if distance < 1e-6:
                            continue  # exact closure was handled above
                        if not inside_edge(a, b) or any(
                            _intersects(a, b, p, q)
                            for p, q in zip(state.points[:-2], state.points[1:-1])
                        ):
                            stats["geometry_prunes"] += 1
                            continue
                        prefix = state.lengths + (length,)
                        if boundary_guided:
                            return_faces = max(
                                1,
                                math.ceil((distance - 1e-7) / 30),
                                math.ceil(abs(8 - total_turn) / 2) - 1,
                            )
                            if (
                                len(prefix) + return_faces > max_faces
                                or abs(8 - total_turn)
                                > 2 * (right_max - r) + diagonal_max - d
                                or not catalog.can_reserve_return(
                                    prefix, distance, return_faces, deadline
                                )
                            ):
                                stats["return_prunes"] += 1
                                continue
                        pts = state.points + (b,)
                        cross_sum = state.cross_sum + _cross(a, b)
                        score = merit(pts, perimeter, cross_sum)
                        # Evaluating each child measures its one-step change
                        # in projected area/perimeter utility. It is ranking,
                        # never an invalid claim that a partial loop is valid.
                        rank = score + 0.15 * (score - state.merit)
                        if boundary_guided:
                            midpoint = Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                            contact = 1 / (1 + polygon.boundary.distance(midpoint))
                            # Score drives the beam; contact is a small local
                            # bonus, never a reward for endlessly tracing walls.
                            rank += (
                                0.02
                                * contact
                                * (
                                    abs(weights.A) * length * length
                                    + abs(weights.C) * length
                                )
                            )
                            if closure_best is not None:
                                rank += 0.1 * max(0.0, closure_best - score)
                        child = State(
                            pts,
                            prefix,
                            heading,
                            total_turn,
                            r,
                            d,
                            perimeter,
                            cross_sum,
                            score,
                        )
                        key = (
                            round(b[0], 5),
                            round(b[1], 5),
                            heading,
                            total_turn,
                            child.lengths,
                        )
                        old = children.get(key)
                        if old is None or rank > old[0]:
                            children[key] = (rank, child)
                        if (
                            stats["states"] % 128 == 0
                            and time.process_time() >= deadline
                        ):
                            break
                    if time.process_time() >= deadline:
                        break
                if time.process_time() >= deadline:
                    break
            # Keep heading/position diversity so a single long-side family
            # does not consume every beam slot and suppress useful turns.
            ranked = sorted(children.values(), key=lambda item: item[0], reverse=True)
            width = (
                tapered_width(beam_width, _depth, max_faces, compactness)
                if boundary_guided
                else beam_width
            )
            beam, buckets = [], {}
            for _rank, state in ranked:
                end = state.points[-1]
                bucket = (state.heading, round(end[0] / 8), round(end[1] / 8))
                if boundary_guided:
                    bucket += (state.turn,)
                if buckets.get(bucket, 0) >= max(2, width // 8):
                    continue
                buckets[bucket] = buckets.get(bucket, 0) + 1
                beam.append(state)
                if len(beam) >= width:
                    break
    return stats
