"""Geometric seeds for Group 3's inventory-aware face search.

Rectangles give the search an early valid answer.  Boundary and interior seeds
then let less regular shapes explore space away from the room's corners.
"""

import math
import time
from itertools import pairwise

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.prepared import prep

from src.constants import MAX_FACE_LENGTH, TOL


def _expired(deadline):
    return deadline is not None and time.process_time() >= deadline


def _edges(polygon, deadline=None):
    if _expired(deadline):
        return
    vertices = orient(polygon, sign=1.0).exterior.coords
    for index, (a, b) in enumerate(pairwise(vertices)):
        if index % 64 == 0 and _expired(deadline):
            return
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length > TOL:
            yield a, b, dx, dy, length


def search_starts(room, minimum_length, max_starts=40, deadline=None):
    """Return diverse ``(point, heading_degrees)`` seeds, with edge seeds first.

    Each seed must allow at least the shortest available piece.  Seed families
    are interleaved so a roomy corner cannot consume the entire start budget.
    """
    if max_starts <= 0 or _expired(deadline):
        return []
    polygon = room.polygon
    tolerant = prep(polygon.buffer(TOL))
    if _expired(deadline):
        return []
    edges = _edges(polygon, deadline)
    minimum_length = max(float(minimum_length), TOL)
    radius = 2 * minimum_length
    families = [[], [], [], []]
    seen = set()

    def add(family, point, heading):
        if _expired(deadline):
            return
        heading %= 360.0
        key = (round(point[0], 7), round(point[1], 7), round(heading, 7))
        if key in seen:
            return
        angle = math.radians(heading)
        endpoint = (
            point[0] + minimum_length * math.cos(angle),
            point[1] + minimum_length * math.sin(angle),
        )
        if not tolerant.covers(LineString([point, endpoint])):
            return
        if _expired(deadline):
            return
        seen.add(key)
        space = polygon.intersection(Point(point).buffer(radius, quad_segs=8)).area
        families[family].append((space, point, heading))

    for a, _b, dx, dy, length in edges:
        if _expired(deadline):
            break
        heading = math.degrees(math.atan2(dy, dx))
        for fraction in (0.5, 0.25, 0.75, 0.0):
            for inset in (0.0, minimum_length / 2):
                point = (
                    a[0] + fraction * dx - inset * dy / length,
                    a[1] + fraction * dy + inset * dx / length,
                )
                family = 3 if fraction == 0 else (1 if inset else 0)
                add(family, point, heading)
                if inset:
                    for axis_heading in (0.0, 90.0, 180.0, 270.0):
                        add(1, point, axis_heading)

    if not _expired(deadline):
        xmin, ymin, xmax, ymax = polygon.bounds
        interior = [polygon.representative_point(), polygon.centroid]
        for fx, fy in ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
            interior.append(Point(xmin + fx * (xmax - xmin), ymin + fy * (ymax - ymin)))
        for point in interior:
            if _expired(deadline):
                break
            for heading in (0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0):
                add(2, (point.x, point.y), heading)

    for family in families:
        family.sort(key=lambda candidate: -candidate[0])
    starts = []
    index = 0
    while len(starts) < max_starts:
        added = False
        for family in families:
            if index < len(family):
                _space, point, heading = family[index]
                starts.append((point, heading))
                added = True
                if len(starts) == max_starts:
                    break
        if not added:
            break
        index += 1
    return starts


def rectangle_seeds(room, catalog, weights, deadline, consider, *, max_candidates=2000):
    """Try a fixed number of rectangles, subject to the global CPU deadline.

    ``catalog.lengths`` gives individually attainable face lengths.  ``consider``
    must assign pieces jointly to all four faces and validate the construction;
    independent face feasibility alone is not enough.  Its return value is the
    valid score, or ``None`` for an inventory-infeasible rectangle.
    """
    started = time.process_time()
    stats = {"rectangles_tested": 0, "rectangles_contained": 0, "rectangles_valid": 0}

    def finish():
        stats["rectangle_cpu"] = time.process_time() - started
        return stats

    if _expired(deadline):
        return finish()
    polygon = room.polygon
    tolerant = prep(polygon.buffer(TOL))
    if _expired(deadline):
        return finish()
    lengths = sorted(
        {float(length) for length in catalog.lengths if 0 < length <= MAX_FACE_LENGTH}
    )
    if not lengths or _expired(deadline):
        return finish()

    # The bonus is an upper bound until the joint inventory assignment confirms
    # that gates can actually be placed on distinct faces.
    def optimistic_score(width, height):
        return (
            1000
            + weights.A * width * height
            + weights.C * 2 * (width + height)
            + max(0, weights.G)
        )

    sizes = [(optimistic_score(w, h), w, h) for w in lengths for h in lengths if w >= h]
    sizes.sort(reverse=True)
    angles = []
    angle_keys = set()

    def add_angle(angle):
        angle %= 180.0
        key = round(angle, 7)
        if key not in angle_keys:
            angles.append(angle)
            angle_keys.add(key)

    for angle in (0, 90, 45, 135):
        add_angle(angle)
    for _a, _b, dx, dy, _length in _edges(polygon, deadline):
        if _expired(deadline):
            return finish()
        angle = math.degrees(math.atan2(dy, dx))
        add_angle(angle)
        add_angle(angle + 90)
    for angle in range(15, 180, 15):
        add_angle(angle)

    room_vertices = []
    for index, point in enumerate(polygon.exterior.coords):
        if index % 64 == 0 and _expired(deadline):
            return finish()
        room_vertices.append(point)
    room_vertices.pop()
    if _expired(deadline):
        return finish()
    centers = [polygon.centroid, polygon.representative_point()]
    projections = []
    for angle in angles:
        if _expired(deadline):
            return finish()
        radians = math.radians(angle)
        c, s = math.cos(radians), math.sin(radians)
        vertices = []
        for index, (x, y) in enumerate(room_vertices):
            if index % 64 == 0 and _expired(deadline):
                return finish()
            vertices.append((x * c + y * s, -x * s + y * c))
        us, vs = zip(*vertices)
        projected_centers = [(p.x * c + p.y * s, -p.x * s + p.y * c) for p in centers]
        projections.append(
            (c, s, min(us), max(us), min(vs), max(vs), vertices, projected_centers)
        )

    best_score = 0.0
    tried = set()
    stop = False
    for upper, width, height in sizes:
        if upper <= best_score + 1e-8 or time.process_time() >= deadline:
            break
        size_done = False
        assignments_tried = 0
        for c, s, umin, umax, vmin, vmax, vertices, fixed_centers in projections:
            if time.process_time() >= deadline:
                stop = True
                break
            if width > umax - umin + TOL or height > vmax - vmin + TOL:
                continue
            ulo, uhi = umin + width / 2, umax - width / 2
            vlo, vhi = vmin + height / 2, vmax - height / 2
            candidate_centers = list(fixed_centers)
            for uf, vf in (
                (0.5, 0.5),
                (0.25, 0.25),
                (0.75, 0.75),
                (0.25, 0.75),
                (0.75, 0.25),
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 1),
            ):
                candidate_centers.append(
                    (ulo + uf * (uhi - ulo), vlo + vf * (vhi - vlo))
                )
            # Room edge samples become candidate rectangle centers after moving
            # half a width or height inward.  This includes edge-aligned starts
            # while still checking the whole polygon in a concave room.
            for index, a in enumerate(vertices):
                if index % 64 == 0 and _expired(deadline):
                    return finish()
                b = vertices[(index + 1) % len(vertices)]
                for fraction in (0.5, 0.25, 0.75):
                    u = a[0] + fraction * (b[0] - a[0])
                    v = a[1] + fraction * (b[1] - a[1])
                    candidate_centers.extend(
                        (
                            (u, v + height / 2),
                            (u, v - height / 2),
                            (u + width / 2, v),
                            (u - width / 2, v),
                        )
                    )
            for u, v in candidate_centers:
                if time.process_time() >= deadline:
                    stop = True
                    break
                if u < ulo - TOL or u > uhi + TOL or v < vlo - TOL or v > vhi + TOL:
                    continue
                corners = [
                    (u - width / 2, v - height / 2),
                    (u + width / 2, v - height / 2),
                    (u + width / 2, v + height / 2),
                    (u - width / 2, v + height / 2),
                ]
                points = [(x * c - y * s, x * s + y * c) for x, y in corners]
                key = tuple(sorted((round(x, 7), round(y, 7)) for x, y in points))
                if key in tried:
                    continue
                if stats["rectangles_tested"] >= max_candidates:
                    return finish()
                tried.add(key)
                stats["rectangles_tested"] += 1
                if not tolerant.covers(Polygon(points)):
                    continue
                stats["rectangles_contained"] += 1
                score = consider(points)
                assignments_tried += 1
                if score is not None:
                    stats["rectangles_valid"] += 1
                    best_score = max(best_score, score)
                    size_done = True
                elif assignments_tried >= 2:
                    size_done = True
                # Packing and score are translation/rotation independent.  One
                # backup placement guards against a numerical boundary failure;
                # repeated attempts cannot repair an inventory shortfall.
                if size_done:
                    break
            if stop or size_done:
                break
        if stop:
            break
    return finish()
