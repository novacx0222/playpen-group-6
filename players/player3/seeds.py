import math
import time

from shapely.geometry import Polygon
from shapely.prepared import prep

from src.constants import MAX_FACE_LENGTH, TOL
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import ConnectorType


def symmetric_hexagon_seeds(room, inventory, weights, catalog, deadline):
    begun = time.process_time()
    stats = {
        "hexagon_sizes": 0,
        "hexagon_packings": 0,
        "hexagon_placements": 0,
        "hexagons_valid": 0,
        "hexagon_best_score": None,
    }

    def finish(best=None):
        stats["hexagon_cpu"] = time.process_time() - begun
        return best, stats

    if (
        begun >= deadline
        or not inventory.has_gate()
        or inventory.connectors.get(ConnectorType.RIGHT, 0) < 2
        or inventory.connectors.get(ConnectorType.DIAGONAL, 0) < 4
    ):
        return finish()

    lengths = [length for length in catalog.lengths if 0 < length <= MAX_FACE_LENGTH]
    if not lengths:
        return finish()

    bonus_upper = max(0.0, weights.G)
    sizes = []
    for a in lengths:
        for b in lengths:
            if time.process_time() >= deadline:
                return finish()
            diagonal = b / math.sqrt(2)
            for c in lengths:
                perimeter = 2 * (a + b + c)
                if perimeter > catalog.max_perimeter + TOL:
                    continue
                area = a * c + diagonal * (a + c)
                upper = 1000 + weights.A * area + weights.C * perimeter + bonus_upper
                sizes.append((upper, a, b, c))
    sizes.sort(reverse=True)

    polygon = room.polygon
    prepared = prep(polygon.buffer(TOL))
    boundary = list(polygon.exterior.coords)[:-1]
    centers = [polygon.centroid, polygon.representative_point()]
    angles = {0.0, 45.0, 90.0, 135.0}
    for index, (x, y) in enumerate(boundary):
        if time.process_time() >= deadline:
            return finish()
        following = boundary[(index + 1) % len(boundary)]
        angle = math.degrees(math.atan2(following[1] - y, following[0] - x))
        for offset in (0.0, 45.0, 90.0, 135.0):
            angles.add(round((angle + offset) % 180, 7))
    projections = []
    for angle in sorted(angles):
        if time.process_time() >= deadline:
            return finish()
        cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        projected = [
            (x * cosine + y * sine, -x * sine + y * cosine) for x, y in boundary
        ]
        us, vs = zip(*projected)
        fixed = [
            (p.x * cosine + p.y * sine, -p.x * sine + p.y * cosine) for p in centers
        ]
        projections.append(
            (angle, cosine, sine, min(us), max(us), min(vs), max(vs), fixed)
        )

    best, best_score = None, -math.inf
    for upper, a, b, c in sizes:
        if time.process_time() >= deadline or upper <= best_score + TOL:
            break
        stats["hexagon_sizes"] += 1
        diagonal = b / math.sqrt(2)
        width, height = a + diagonal, c + diagonal
        local = [
            (0.0, 0.0),
            (a, 0.0),
            (width, diagonal),
            (width, height),
            (diagonal, height),
            (0.0, c),
        ]

        packed = catalog.build(local, prefer_bonus=weights.G > 0, deadline=deadline)
        if packed is None:
            continue
        stats["hexagon_packings"] += 1
        placed = False
        for angle, cosine, sine, umin, umax, vmin, vmax, fixed in projections:
            if time.process_time() >= deadline:
                break
            if width > umax - umin + TOL or height > vmax - vmin + TOL:
                continue
            ulo, uhi = umin + width / 2, umax - width / 2
            vlo, vhi = vmin + height / 2, vmax - height / 2
            candidates = list(fixed)
            candidates.extend(
                (ulo + fu * (uhi - ulo), vlo + fv * (vhi - vlo))
                for fu, fv in (
                    (0.5, 0.5),
                    (0, 0),
                    (1, 0),
                    (1, 1),
                    (0, 1),
                    (0.25, 0.25),
                    (0.75, 0.75),
                    (0.25, 0.75),
                    (0.75, 0.25),
                )
            )
            tried = set()
            for u, v in candidates:
                if time.process_time() >= deadline:
                    break
                key = (round(u, 7), round(v, 7))
                if key in tried:
                    continue
                tried.add(key)
                if not (ulo - TOL <= u <= uhi + TOL and vlo - TOL <= v <= vhi + TOL):
                    continue
                shifted = [(x + u - width / 2, y + v - height / 2) for x, y in local]
                vertices = [
                    (x * cosine - y * sine, x * sine + y * cosine) for x, y in shifted
                ]
                stats["hexagon_placements"] += 1
                if not prepared.covers(Polygon(vertices)):
                    continue
                construction = Construction(
                    vertices[0], angle, list(packed.pieces), list(packed.connectors)
                )
                result = validate_construction(construction, room, inventory)
                if not result.valid:
                    continue
                stats["hexagons_valid"] += 1
                score = score_construction(result, weights)
                if score > best_score + TOL:
                    best, best_score = construction, score
                    stats["hexagon_best_score"] = score

                placed = True
                break
            if placed:
                break
    return finish(best)
