from shapely.geometry import Polygon, MultiPoint, box
from shapely.ops import unary_union, polylabel
from shapely import affinity, constrained_delaunay_triangles


def triangles(poly):
    return list(constrained_delaunay_triangles(poly).geoms)


def minkowski_convex(a, b):
    pa = list(a.exterior.coords)[:-1]
    pb = list(b.exterior.coords)[:-1]
    pts = [(x1 + x2, y1 + y2) for x1, y1 in pa for x2, y2 in pb]
    return MultiPoint(pts).convex_hull


def inner_fit_region(Q, P, eps=1e-9):
    """Set of translations t with P + t inside Q (P already rotated)."""
    qx0, qy0, qx1, qy1 = Q.bounds
    px0, py0, px1, py1 = P.bounds
    wx0, wy0, wx1, wy1 = qx0 - px0, qy0 - py0, qx1 - px1, qy1 - py1
    if wx0 > wx1 or wy0 > wy1:
        return None  # P's bbox can't fit in Q's bbox
    W = box(wx0, wy0, wx1, wy1)

    C = box(qx0, qy0, qx1, qy1).difference(Q)  # bounded "obstacle"
    negP = affinity.scale(P, xfact=-1, yfact=-1, origin=(0, 0))

    C_tris = [t for g in getattr(C, "geoms", [C]) for t in triangles(g)]
    P_tris = triangles(negP)

    forbidden = unary_union([minkowski_convex(c, p) for c in C_tris for p in P_tris])
    T = W.difference(forbidden)
    return None if T.is_empty else T


def pick_translation(T):
    """Choose a robust witness point from the feasible region T."""
    polys = [
        g for g in getattr(T, "geoms", [T]) if g.geom_type == "Polygon" and g.area > 0
    ]
    if polys:
        best = max(polys, key=lambda g: g.area)
        return polylabel(
            best, tolerance=1e-6
        )  # deepest interior point (pole of inaccessibility)
    # degenerate (tight fit): T is a segment or point
    return T.representative_point()


def place_at_angle(Q, P, theta_deg, eps=1e-9):
    """Return (placed_polygon, (theta_deg, tx, ty)) or None. P is rotated about the origin."""
    P0 = Polygon(P.exterior)  # holes in P are irrelevant
    Pr = affinity.rotate(P0, theta_deg, origin=(0, 0))
    T = inner_fit_region(Q, Pr.buffer(-eps) if eps else Pr)
    if T is None:
        return None
    t = pick_translation(T)
    placed = affinity.translate(Pr, xoff=t.x, yoff=t.y)
    return placed, (theta_deg, t.x, t.y)


def find_fit(Q, P, angles=None, tol=1e-7, stop=None):
    """Try each rotation in `angles` (degrees; default a 1-degree sweep) and
    return (placed_polygon, params) for the first that fits inside Q, else
    None. `stop` is an optional callable; when it returns True we give up."""
    if angles is None:
        angles = [float(k) for k in range(360)]
    for theta in angles:
        if stop is not None and stop():
            return None
        res = place_at_angle(Q, P, theta)
        if res is None:
            continue
        placed, params = res
        if Q.buffer(tol).covers(placed):
            return placed, params
    return None
