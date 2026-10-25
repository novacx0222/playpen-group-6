import math
import time
from dataclasses import dataclass, field
from itertools import pairwise

from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.prepared import prep

from src.constants import TOL
from src.enclosure import score_construction, validate_construction

from .passages import passage_patches
from .room_map import RoomMap


def face_vertices(polygon):
    """Remove union-generated straight joints before packing complete faces."""
    if polygon.geom_type != "Polygon" or polygon.interiors or not polygon.is_valid:
        return None
    points = list(orient(polygon, sign=1.0).exterior.coords[:-1])
    changed = True
    while changed and len(points) >= 3:
        changed = False
        result = []
        for i, b in enumerate(points):
            a, c = points[i - 1], points[(i + 1) % len(points)]
            u, v = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
            if abs(u[0] * v[1] - u[1] * v[0]) <= TOL and u[0] * v[0] + u[1] * v[1] >= 0:
                changed = True
            else:
                result.append(b)
        points = result
    return points if len(points) >= 3 else None


def face_key(face):
    return tuple(tuple(round(value, 6) for value in point) for point in face)


def outward_normal(face):
    a, b = face
    length = math.dist(a, b)
    return (b[1] - a[1]) / length, (a[0] - b[0]) / length


def ray_clearance(room, origin, direction):
    """Distance to the first room exit, never to a later re-entry wall."""
    minx, miny, maxx, maxy = room.bounds
    reach = 2 * math.hypot(maxx - minx, maxy - miny) + 1
    dx, dy = direction
    ray = LineString((origin, (origin[0] + reach * dx, origin[1] + reach * dy)))
    intersection = room.intersection(ray)

    def contiguous(geometry):
        if geometry.is_empty:
            return 0.0
        if geometry.geom_type == "LineString":
            projections = [
                (x - origin[0]) * dx + (y - origin[1]) * dy for x, y in geometry.coords
            ]
            return max(projections) if min(projections) <= TOL else 0.0
        if hasattr(geometry, "geoms"):
            return max((contiguous(part) for part in geometry.geoms), default=0.0)
        return 0.0

    return max(0.0, contiguous(intersection))


def face_clearance(room, face):
    a, b = face
    direction = outward_normal(face)
    distances = [
        ray_clearance(
            room, (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])), direction
        )
        for f in (0.05, 0.25, 0.5, 0.75, 0.95)
    ]
    return max(distances), sum(distances) / len(distances)


def ranked_faces(polygon, room, blocked=(), deadline=math.inf):
    """Greatest perpendicular clearance first, with potential area as a tie-break."""
    points = face_vertices(polygon) or []
    ranked = []
    for face in pairwise(points + points[:1]):
        if time.process_time() >= deadline:
            break
        if face_key(face) in blocked:
            continue
        clearance, mean = face_clearance(room, face)
        if clearance > TOL:
            ranked.append((clearance, mean * math.dist(*face), face))
    ranked.sort(key=lambda item: item[:2], reverse=True)
    return [face for _, _, face in ranked]


def starting_faces(polygon, room, deadline=math.inf):
    """Up to four different faces, preferring different outward headings."""
    ranked = ranked_faces(polygon, room, deadline=deadline)
    points = face_vertices(polygon) or []
    # Keep blocked faces as initial fronts too: they will explicitly redirect.
    ranked += [face for face in pairwise(points + points[:1]) if face not in ranked]
    chosen, headings = [], set()
    for face in ranked:
        direction = tuple(round(n, 6) for n in outward_normal(face))
        if direction not in headings:
            chosen.append(face)
            headings.add(direction)
        if len(chosen) == 4:
            return chosen
    return (chosen + [face for face in ranked if face not in chosen])[:4]


def outer_face(polygon, patch, previous):
    """Track the new outer boundary after extending the active face."""
    nx, ny = outward_normal(previous)
    target = max(x * nx + y * ny for x, y in patch.exterior.coords)
    points = face_vertices(polygon) or []
    for face in pairwise(points + points[:1]):
        normal = outward_normal(face)
        if normal[0] * nx + normal[1] * ny < 1 - 1e-6:
            continue
        if (
            all(abs(x * nx + y * ny - target) <= TOL for x, y in face)
            and LineString(face).distance(patch) <= TOL
        ):
            return face
    return None


def growth_patches(polygon, faces, lengths):
    """Try full and partial extensions perpendicular to the selected face only."""
    choices = sorted(lengths, reverse=True)
    for a, b in faces:
        size = math.dist(a, b)
        ux, uy = (b[0] - a[0]) / size, (b[1] - a[1]) / size
        nx, ny = uy, -ux  # exterior is on the right of a CCW face
        widths = [size] + [n for n in choices if n < size - TOL]
        # Spread partial-face sizes instead of spending all time on similar ones.
        if len(widths) > 7:
            widths = [widths[round(i * (len(widths) - 1) / 6)] for i in range(7)]
        for width in widths:
            offsets = (
                (0.0,)
                if abs(width - size) < TOL
                else (0.0, size - width, (size - width) / 2)
            )
            for offset in offsets:
                p = (a[0] + offset * ux, a[1] + offset * uy)
                q = (p[0] + width * ux, p[1] + width * uy)
                for depth in choices:
                    yield Polygon(
                        (
                            p,
                            q,
                            (q[0] + depth * nx, q[1] + depth * ny),
                            (p[0] + depth * nx, p[1] + depth * ny),
                        )
                    )


@dataclass(eq=False)
class Front:
    polygon: Polygon
    construction: object
    active_face: object
    members: set
    pending: object = None
    seen: set = field(default_factory=set)
    exhausted: bool = False
    blocked: set = field(default_factory=set)
    route: object = None


def grow_enclosure(
    room,
    inventory,
    weights,
    catalog,
    initial,
    deadline,
    *,
    batch=48,
    use_map=True,
    beam_width=3,
):
    seed_result = validate_construction(initial, room, inventory)
    seed = seed_result.polygon
    faces = starting_faces(seed, room.polygon, deadline)
    room_map = None
    route_choices = []
    if use_map and time.process_time() < deadline:
        room_map = RoomMap(
            room.polygon,
            min(catalog.lengths),
            deadline,
        )
        points = face_vertices(seed)
        candidates = faces + [
            face for face in pairwise(points + points[:1]) if face not in faces
        ]
        routes = room_map.routes(seed, candidates, deadline=deadline)
        claimed, selected = set(), set()
        while routes and len(route_choices) < 4:
            route = max(
                routes,
                key=lambda r: (
                    r.value
                    * (1 - 0.85 * len(r.territory & claimed) / max(1, len(r.territory)))
                ),
            )
            routes = [r for r in routes if face_key(r.face) != face_key(route.face)]
            route_choices.append(route)
            claimed.update(route.territory)
            selected.add(face_key(route.face))
        faces = [r.face for r in route_choices] + [
            f for f in faces if face_key(f) not in selected
        ]
        faces = faces[:4]
    fronts = [Front(seed, initial, face, {i}) for i, face in enumerate(faces)]
    for front, route in zip(fronts, route_choices):
        front.route = route
    stats = {
        "initial_fronts": len(fronts),
        "growth_candidates": 0,
        "growth_steps": 0,
        "merges": 0,
        "resource_rejections": 0,
        "geometry_rejections": 0,
        "initial_area": seed.area,
        "rounds": 0,
        "face_switches": 0,
        "face_switch_history": [],
        "initial_faces": [face_key(face) for face in faces],
        "growth_history": [],
        "map_nodes": len(room_map.points) if room_map else 0,
        "map_complete": room_map.complete if room_map else False,
        "map_routes_selected": len(route_choices),
        "lookahead_expanded": 0,
        "lookahead_turns": 0,
        "route_guided_proposals": 0,
        "route_guided_extensions": 0,
        "lookahead_depth": 1,
        "route_targets": [route.target for route in route_choices],
    }
    cache = {}
    prepared_room = prep(room.polygon.buffer(TOL))
    guided_keys = set()

    def growth_proposals(front, polygon, face, until):
        seen = set()
        families = [
            (
                True,
                iter(
                    passage_patches(
                        polygon, face, catalog.lengths, front.route, room.polygon, until
                    )
                ),
            ),
            (False, iter(growth_patches(polygon, [face], catalog.lengths))),
        ]
        # Do not let detailed passage alignment crowd out broad ordinary moves.
        while families:
            remaining = []
            for guided, iterator in families:
                patch = next(iterator, None)
                if patch is None:
                    continue
                remaining.append((guided, iterator))
                key = patch.normalize().wkb
                if key in seen:
                    continue
                seen.add(key)
                if guided:
                    guided_keys.add(key)
                    stats["route_guided_proposals"] += 1
                yield patch
            families = remaining

    best_finished = [score_construction(seed_result, weights), seed.area, initial]

    def pack(polygon):
        if time.process_time() >= deadline:
            return None
        vertices = face_vertices(polygon)
        if not vertices or not prepared_room.covers(polygon):
            return None
        key = polygon.normalize().wkb
        if key in cache:
            return cache[key]
        construction = catalog.build(
            vertices, prefer_bonus=weights.G > 0, deadline=deadline
        )
        if construction is not None:
            result = validate_construction(construction, room, inventory)
            if not result.valid:
                construction = None
            else:
                # Keep a valid answer even if later growth or merging scores worse.
                score = score_construction(result, weights)
                if score > best_finished[0] + TOL:
                    best_finished[:] = [score, result.area, construction]
        if time.process_time() < deadline:
            cache[key] = construction
        return construction

    def feasible(front, candidate):
        construction = pack(candidate)
        if construction is None:
            return None
        shared = unary_union(
            [candidate] + [f.polygon for f in fronts if f is not front]
        )
        pack(shared)
        return construction

    def rank(front, candidate):
        future = (
            front.route.value * front.route.progress(candidate) if front.route else 0.0
        )
        return candidate.area + 0.5 * future

    def lookahead_layer(front, beam, deadline):
        """A second beam layer may continue forward or turn onto adjacent faces."""
        expanded = list(beam)
        for parent in beam:
            if time.process_time() >= deadline:
                break
            _, polygon, _, patch, previous = parent
            active = outer_face(polygon, patch, previous)
            if active is None:
                continue
            points = face_vertices(polygon)
            adjacent = [
                face
                for face in pairwise(points + points[:1])
                if face == active
                or any(math.dist(p, q) <= TOL for p in face for q in active)
            ]
            if front.route:
                adjacent.sort(
                    key=lambda face: math.dist(
                        ((face[0][0] + face[1][0]) / 2, (face[0][1] + face[1][1]) / 2),
                        front.route.target,
                    )
                )
            iterators = [
                (face, iter(growth_proposals(front, polygon, face, deadline)))
                for face in adjacent[:3]
            ]
            for _ in range(32):
                for face, proposals in iterators:
                    if time.process_time() >= deadline:
                        return sorted(expanded, key=lambda item: item[0], reverse=True)[
                            :beam_width
                        ]
                    next_patch = next(proposals, None)
                    if next_patch is None:
                        continue
                    candidate = polygon.union(next_patch)
                    stats["lookahead_expanded"] += 1
                    if candidate.area <= polygon.area + TOL or not prepared_room.covers(
                        candidate
                    ):
                        continue
                    construction = feasible(front, candidate)
                    if construction is None:
                        continue
                    expanded.append(
                        (
                            rank(front, candidate),
                            candidate,
                            construction,
                            next_patch,
                            face,
                        )
                    )
                    if outward_normal(face) != outward_normal(previous):
                        stats["lookahead_turns"] += 1
        expanded.sort(key=lambda item: item[0], reverse=True)
        return expanded[:beam_width]

    def lookahead(front, beam, deadline):
        # A narrow entrance may need an approach, a neck, and a turn before
        # opening out into the next region. Keep valid alternatives throughout.
        layers = (
            3 if any(item[3].normalize().wkb in guided_keys for item in beam) else 1
        )
        for layer in range(layers):
            now = time.process_time()
            if now >= deadline:
                break
            beam = lookahead_layer(front, beam, deadline)
            stats["lookahead_depth"] = max(stats["lookahead_depth"], layer + 2)
        return beam

    while fronts and time.process_time() < deadline:
        stats["rounds"] += 1
        progressed = False
        for front in list(fronts):
            if (
                front not in fronts
                or front.exhausted
                or time.process_time() >= deadline
            ):
                continue
            if front.pending is None:
                if front.active_face is None:
                    ranked = ranked_faces(
                        front.polygon, room.polygon, front.blocked, deadline
                    )
                    claimed = {
                        face_key(f.active_face)
                        for f in fronts
                        if f is not front and f.active_face is not None
                    }
                    available = [
                        face for face in ranked if face_key(face) not in claimed
                    ]
                    if not available:
                        available = ranked
                    if not available:
                        front.exhausted = True
                        continue
                    front.active_face = available[0]
                    if room_map is not None and room_map.complete:
                        claimed = frozenset().union(
                            *(
                                f.route.territory
                                for f in fronts
                                if f is not front and f.route
                            )
                        )
                        routes = room_map.routes(
                            front.polygon,
                            available[:12],
                            claimed,
                            deadline=deadline,
                        )
                        if routes:
                            front.route = routes[0]
                            front.active_face = front.route.face
                            stats["map_routes_selected"] += 1
                            stats["route_targets"].append(front.route.target)
                        else:
                            front.route = None
                    stats["face_switches"] += 1
                    stats["face_switch_history"].append(
                        {
                            "fronts": sorted(front.members),
                            "face": face_key(front.active_face),
                            "clearance": face_clearance(
                                room.polygon, front.active_face
                            )[0],
                        }
                    )
                front.pending = iter(
                    growth_proposals(front, front.polygon, front.active_face, deadline)
                )
            best = None
            beam = []
            selected_face = front.active_face
            for _ in range(batch):
                if time.process_time() >= deadline:
                    break
                patch = next(front.pending, None)
                if patch is None:
                    front.blocked.add(face_key(front.active_face))
                    front.active_face, front.pending = None, None
                    break
                stats["growth_candidates"] += 1
                candidate = front.polygon.union(patch)
                key = candidate.normalize().wkb
                if candidate.area <= front.polygon.area + TOL or key in front.seen:
                    continue
                front.seen.add(key)
                if not prepared_room.covers(candidate):
                    stats["geometry_rejections"] += 1
                    continue
                construction = feasible(front, candidate)
                if construction is None:
                    stats["resource_rejections"] += 1
                    continue
                proposal = (
                    rank(front, candidate),
                    candidate,
                    construction,
                    patch,
                    selected_face,
                )
                beam.append(proposal)
                beam.sort(key=lambda item: item[0], reverse=True)
                beam = beam[:beam_width]
            if beam:
                if use_map:
                    beam = lookahead(
                        front,
                        beam,
                        deadline,
                    )
                best = beam[0]
            if best is None:
                continue
            _, front.polygon, front.construction, patch, previous_face = best
            if patch.normalize().wkb in guided_keys:
                stats["route_guided_extensions"] += 1
            front.active_face = outer_face(front.polygon, patch, previous_face)
            front.pending, front.exhausted = None, False
            stats["growth_steps"] += 1
            progressed = True
            stats["growth_history"].append(
                {
                    "fronts": sorted(front.members),
                    "face": face_key(previous_face),
                    "direction": outward_normal(previous_face),
                    "area": front.polygon.area,
                }
            )
            # The seed belongs to every front. Only new territory counts as
            # meeting; otherwise all four would falsely merge at initialization.
            merging = True
            while merging and time.process_time() < deadline:
                merging = False
                territory = front.polygon.difference(seed)
                for other in list(fronts):
                    if other is front:
                        continue
                    contact = territory.intersection(other.polygon.difference(seed))
                    if contact.area <= TOL and contact.length <= TOL:
                        continue
                    merged = front.polygon.union(other.polygon)
                    construction = pack(merged)
                    if construction is None:
                        continue
                    front.polygon, front.construction = merged, construction
                    front.members.update(other.members)
                    fronts.remove(other)
                    front.pending, front.exhausted = None, False
                    front.blocked.clear()
                    front.route = None
                    if front.active_face is not None:
                        current = face_vertices(front.polygon)
                        if face_key(front.active_face) not in {
                            face_key(face) for face in pairwise(current + current[:1])
                        }:
                            front.active_face = None
                    territory = front.polygon.difference(seed)
                    stats["merges"] += 1
                    merging = True
                    break
        if not progressed and all(f.exhausted for f in fronts):
            break
    winner = next(
        (front for front in fronts if front.construction is best_finished[2]), None
    )
    stats.update(
        active_fronts=len(fronts),
        final_area=best_finished[1],
        winning_fronts=sorted(winner.members) if winner else [],
        front_areas=[front.polygon.area for front in fronts],
    )
    if winner is None:
        stats["returned_beam_alternative"] = True
    return best_finished[2], stats
