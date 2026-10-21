import math
from itertools import pairwise
from time import process_time

from shapely import affinity, constrained_delaunay_triangles
from shapely.geometry import MultiPoint, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep


def room_search_angles(room, cluster_tolerance=3.0):
    """Choose a length-weighted boundary axis and eight relative orientations."""
    ring = list(room.exterior.coords)
    directions = []
    for (x0, y0), (x1, y1) in pairwise(ring):
        length = math.hypot(x1 - x0, y1 - y0)
        if length > 0:
            # Opposite and perpendicular walls describe the same orthogonal axis.
            angle = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 90
            directions.append((angle, length))

    def support(direction):
        angle, length = direction
        weight = sum(
            other_length
            for other_angle, other_length in directions
            if min(abs(angle - other_angle), 90 - abs(angle - other_angle))
            <= cluster_tolerance
        )
        return weight, length

    # Use an actual boundary direction rather than rounding or averaging:
    # a small angular error can rule out a tight corridor placement.
    main_angle = max(directions, key=support)[0] if directions else 0.0
    return [
        (main_angle + offset) % 360 for offset in (0, 180, 90, 270, 45, 225, 135, 315)
    ]


def triangles(poly):
    return list(constrained_delaunay_triangles(poly).geoms)


def minkowski_convex(a, b):
    pa = list(a.exterior.coords)[:-1]
    pb = list(b.exterior.coords)[:-1]
    pts = [(x1 + x2, y1 + y2) for x1, y1 in pa for x2, y2 in pb]
    return MultiPoint(pts).convex_hull


def inner_fit_region(Q, P, eps=1e-9, obstacle_triangles=None):
    """Set of translations t with P + t inside Q (P already rotated)."""
    qx0, qy0, qx1, qy1 = Q.bounds
    px0, py0, px1, py1 = P.bounds
    wx0, wy0, wx1, wy1 = qx0 - px0, qy0 - py0, qx1 - px1, qy1 - py1
    if wx0 > wx1 or wy0 > wy1:
        return None  # P's bbox can't fit in Q's bbox
    W = box(wx0, wy0, wx1, wy1)

    negP = affinity.scale(P, xfact=-1, yfact=-1, origin=(0, 0))

    if obstacle_triangles is None:
        C = box(qx0, qy0, qx1, qy1).difference(Q)
        obstacle_triangles = [t for g in getattr(C, "geoms", [C]) for t in triangles(g)]
    P_tris = triangles(negP)

    forbidden = unary_union(
        [minkowski_convex(c, p) for c in obstacle_triangles for p in P_tris]
    )
    T = W.difference(forbidden)
    return None if T.is_empty else T


def pick_translation(T):
    """Choose a robust witness point from the feasible region T."""
    polys = [
        g for g in getattr(T, "geoms", [T]) if g.geom_type == "Polygon" and g.area > 0
    ]
    if polys:
        best = max(polys, key=lambda g: g.area)
        # Any interior witness is enough: finding the deepest point to 1e-6
        # precision is unnecessary and can be costly for long, thin regions.
        return best.representative_point()
    # degenerate (tight fit): T is a segment or point
    return T.representative_point()


def place_at_angle(Q, P, theta_deg, eps=1e-9, obstacle_triangles=None):
    """Return (placed_polygon, (theta_deg, tx, ty)) or None. P is rotated about the origin."""
    P0 = Polygon(P.exterior)  # holes in P are irrelevant
    Pr = affinity.rotate(P0, theta_deg, origin=(0, 0))
    T = inner_fit_region(
        Q, Pr.buffer(-eps) if eps else Pr, obstacle_triangles=obstacle_triangles
    )
    if T is None:
        return None
    t = pick_translation(T)
    placed = affinity.translate(Pr, xoff=t.x, yoff=t.y)
    return placed, (theta_deg, t.x, t.y)


def find_fit(
    Q, P, n_angles=8, tol=1e-7, max_cpu_seconds=2.0, *, angles=None, stop=None
):
    """Try at most eight room-oriented angles, with a bounded inner-fit search.

    n_angles caps the number of orientations; it no longer defines a uniform
    angular grid. This heuristic may miss fits outside the selected angles.

    The time budget is checked between geometry operations; a single native
    operation may exceed it. Returning None means no fit was found within
    the budget, not that no placement exists.
    """
    deadline = process_time() + max_cpu_seconds
    padded_room = Q.buffer(tol)
    prepared_room = prep(padded_room)
    angles = (room_search_angles(Q) if angles is None else list(angles))[
        : max(0, min(n_angles, 8))
    ]
    if not angles:
        return None

    # Align each corner of the candidate's bounding box with room vertices.
    # This catches tight corridor placements without Minkowski sums.
    for theta in angles:
        rotated = affinity.rotate(P, theta, origin=(0, 0))
        px0, py0, px1, py1 = rotated.bounds
        qx0, qy0, qx1, qy1 = Q.bounds
        if px1 - px0 > qx1 - qx0 + tol or py1 - py0 > qy1 - qy0 + tol:
            continue
        # Actual vertices also support aligning tilted shapes to tilted walls;
        # axis-aligned bounding-box corners alone do not preserve that alignment.
        anchors = list(
            dict.fromkeys(
                list(rotated.exterior.coords)[:-1]
                + [(px0, py0), (px0, py1), (px1, py0), (px1, py1)]
            )
        )
        for qx, qy in Q.exterior.coords:
            for px, py in anchors:
                if process_time() >= deadline or (stop is not None and stop()):
                    return None
                tx, ty = qx - px, qy - py
                if (
                    tx + px0 < qx0 - tol
                    or tx + px1 > qx1 + tol
                    or ty + py0 < qy0 - tol
                    or ty + py1 > qy1 + tol
                ):
                    continue
                placed = affinity.translate(rotated, xoff=tx, yoff=ty)
                if prepared_room.covers(placed):
                    return placed, (theta, tx, ty)

    if process_time() >= deadline or (stop is not None and stop()):
        return None
    qx0, qy0, qx1, qy1 = Q.bounds
    obstacles = box(qx0, qy0, qx1, qy1).difference(Q)
    obstacle_triangles = [
        t for g in getattr(obstacles, "geoms", [obstacles]) for t in triangles(g)
    ]
    for theta in angles:
        if process_time() >= deadline or (stop is not None and stop()):
            return None
        res = place_at_angle(Q, P, theta, obstacle_triangles=obstacle_triangles)
        if res is None:
            continue
        placed, params = res
        if prepared_room.covers(placed):  # verify the result directly
            return placed, params
    return None
