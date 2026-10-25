"""Build room-following enclosures from connected grids of interior cells.

The grid only proposes geometry. ``PolygonPacker`` assigns real walls, gates,
and connectors and validates the resulting construction. Long boundary runs
are broken by genuine inward notches, never by pretending straight joints
reset the simulator's 30-unit face limit.
"""

import math
import time
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
from shapely import box, covers, union_all
from shapely.affinity import rotate
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from players.player5.packing import PolygonPacker
from src.constants import MAX_FACE_LENGTH, MIN_WALL_LENGTH, TOL
from src.pieces import ConnectorType


@dataclass
class GridStats:
    """Counts for inspecting which geometric candidates were useful."""

    grids: int = 0
    candidates: int = 0
    constructions: int = 0


def _corners(polygon):
    """Remove collinear grid vertices and return a counterclockwise ring."""
    points = list(orient(polygon, sign=1).exterior.coords[:-1])
    result = []
    for index, point in enumerate(points):
        before = points[index - 1]
        after = points[(index + 1) % len(points)]
        incoming = point[0] - before[0], point[1] - before[1]
        outgoing = after[0] - point[0], after[1] - point[1]
        if abs(incoming[0] * outgoing[1] - incoming[1] * outgoing[0]) > 1e-7:
            result.append(point)
    return result


def _edges(points):
    return zip(points, points[1:] + points[:1])


def _legal_corner_lengths(polygon):
    points = _corners(polygon)
    return all(math.dist(a, b) >= MIN_WALL_LENGTH - 1e-7 for a, b in _edges(points))


def _smooth_short_faces(polygon, inventory, deadline):
    """Remove tiny staircase ears when they require unavailable short walls.

    A face shorter than ten cannot be assembled from multiple legal pieces.
    Replacing the legs around a convex corner by the other two sides of the
    same rectangle can merge neighboring faces. We accept only
    inward changes that reduce a real stock deficit and keep all face limits.
    """
    available = {
        length: inventory.walls.get(length, 0) + inventory.gates.get(length, 0)
        for length in range(5, 10)
    }
    for _ in range(500):
        if time.process_time() >= deadline:
            return polygon
        points = _corners(polygon)
        lengths = [round(math.dist(a, b)) for a, b in _edges(points)]
        counts = Counter(lengths)
        deficit = sum(
            max(0, counts[length] - count) for length, count in available.items()
        )
        if not deficit:
            return polygon
        accepted = False
        indices = sorted(
            range(len(points)), key=lambda index: lengths[index - 1] * lengths[index]
        )
        for index in indices:
            if time.process_time() >= deadline:
                return polygon
            corner = points[index]
            adjacent = lengths[index - 1], lengths[index]
            if not any(
                5 <= size < 10 and counts[size] > available[size] for size in adjacent
            ):
                continue
            a, c = points[index - 1], points[(index + 1) % len(points)]
            ux, uy = corner[0] - a[0], corner[1] - a[1]
            vx, vy = c[0] - corner[0], c[1] - corner[1]
            if ux * vy - uy * vx <= 0:
                continue
            inward = a[0] + c[0] - corner[0], a[1] + c[1] - corner[1]
            changed = Polygon(points[:index] + [inward] + points[index + 1 :])
            if not changed.is_valid or changed.is_empty or not polygon.covers(changed):
                continue
            changed_lengths = [
                round(math.dist(a, b)) for a, b in _edges(_corners(changed))
            ]
            if min(changed_lengths) < 5 or max(changed_lengths) > MAX_FACE_LENGTH:
                continue
            changed_counts = Counter(changed_lengths)
            new_deficit = sum(
                max(0, changed_counts[length] - count)
                for length, count in available.items()
            )
            if new_deficit >= deficit:
                continue
            polygon = changed
            accepted = True
            break
        if not accepted:
            return polygon
    return polygon


def _break_long_faces(polygon, deadline, notch_size=5):
    """Cut square inward notches until every straight face is at most 30.

    A notch is accepted only if it leaves one simple polygon, creates no hole,
    and leaves every face at least five units long. In a passage too thin to
    support a notch, return None rather than submit an illegal long face.
    """
    for _ in range(2000):
        if time.process_time() >= deadline:
            return None
        points = _corners(polygon)
        overlong = [
            (math.dist(a, b), a, b)
            for a, b in _edges(points)
            if math.dist(a, b) > MAX_FACE_LENGTH + 1e-7
        ]
        if not overlong:
            return polygon
        length, start, end = max(overlong)
        dx, dy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        # Counterclockwise polygons have their interior on the left.
        nx, ny = -dy, dx
        middle = round((length - notch_size) / (2 * notch_size)) * notch_size
        near_end = math.floor((MAX_FACE_LENGTH - notch_size) / notch_size) * notch_size
        locations = [middle, near_end, round(length) - near_end - notch_size]
        locations += [middle + shift * notch_size for shift in (-2, -1, 1, 2)]
        accepted = False
        for along in dict.fromkeys(locations):
            if not MIN_WALL_LENGTH <= along <= length - notch_size - MIN_WALL_LENGTH:
                continue
            a = start[0] + along * dx, start[1] + along * dy
            b = a[0] + notch_size * dx, a[1] + notch_size * dy
            notch = Polygon(
                [
                    a,
                    b,
                    (b[0] + notch_size * nx, b[1] + notch_size * ny),
                    (a[0] + notch_size * nx, a[1] + notch_size * ny),
                ]
            )
            if not polygon.covers(notch):
                continue
            reduced = polygon.difference(notch)
            if (
                reduced.geom_type != "Polygon"
                or reduced.interiors
                or not reduced.is_valid
                or not _legal_corner_lengths(reduced)
            ):
                continue
            polygon = reduced
            accepted = True
            break
        if not accepted:
            return None
    return None


def _orientations(room):
    """Prefer the room's longest wall directions, modulo a quarter turn."""
    points = list(room.polygon.exterior.coords[:-1])
    directions = sorted(
        (
            (math.dist(a, b), math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 90)
            for a, b in _edges(points)
        ),
        reverse=True,
    )
    result = [0.0]
    for _, angle in directions:
        if all(
            min(abs(angle - prior), 90 - abs(angle - prior)) >= 10 for prior in result
        ):
            result.append(angle)
        if len(result) == 4:
            break
    return result


def _rasterize(polygon, step, offset):
    """Return fully contained cells and their column/row boundary coordinates.

    ``step`` is either one uniform width or a repeating tuple of mixed widths.
    Both grids use the same cell indices, containment check and outline builder.
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    if isinstance(step, tuple):
        shift = round(5 * offset)
        xs = _mixed_axis(min_x, max_x, step, shift)
        ys = _mixed_axis(min_y, max_y, step[1:] + step[:1], shift)
    else:
        x0 = math.floor(min_x / step) * step + step * offset
        y0 = math.floor(min_y / step) * step + step * offset
        width = max(0, math.ceil((max_x - x0) / step))
        height = max(0, math.ceil((max_y - y0) / step))
        if not width or not height or width * height > 60000:
            return set(), ((), ())
        xs = x0 + np.arange(width + 1) * step
        ys = y0 + np.arange(height + 1) * step
    if (len(xs) - 1) * (len(ys) - 1) > 60000:
        return set(), (xs, ys)
    columns, rows = np.meshgrid(np.arange(len(xs) - 1), np.arange(len(ys) - 1))
    columns, rows = columns.ravel(), rows.ravel()
    squares = box(xs[columns], ys[rows], xs[columns + 1], ys[rows + 1])
    inside = covers(polygon.buffer(TOL / 2), squares)
    return set(zip(columns[inside].tolist(), rows[inside].tolist())), (xs, ys)


def _mixed_axis(low, high, widths, offset):
    """Integer distances, with several short sizes sharing the stock burden."""
    coordinates = [math.floor(low) + offset]
    index = 0
    while coordinates[-1] < high:
        coordinates.append(coordinates[-1] + widths[index % len(widths)])
        index += 1
    return np.asarray(coordinates)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        component = {seed}
        frontier = [seed]
        while frontier:
            x, y = frontier.pop()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        result.append(component)
    return sorted(result, key=lambda component: (-len(component), min(component)))


def _outline(cells, axes):
    columns, rows = np.asarray(sorted(cells), dtype=int).T
    xs, ys = axes
    squares = box(xs[columns], ys[rows], xs[columns + 1], ys[rows + 1])
    polygon = union_all(squares)
    if polygon.geom_type != "Polygon" or polygon.interiors or not polygon.is_valid:
        return None
    return polygon


def _growth_order(cells, seed):
    """Connected breadth-first prefixes cover progressively more of a room."""
    frontier = deque([seed])
    seen = {seed}
    while frontier:
        cell = frontier.popleft()
        yield cell
        x, y = cell
        for neighbor in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            if neighbor in cells and neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)


def search_grid(room, inventory, weights, deadline, consider):
    """Send validated room-following constructions to ``consider``.

    ``deadline`` is an absolute ``time.process_time()`` value. Full interior
    components are tried first. Smaller connected growth prefixes provide
    alternatives when the complete room outline needs more stock than exists.
    The caller keeps its incumbent, so a failed grid never loses a solution.
    """
    stats = GridStats()
    packer = PolygonPacker(inventory, room, weights, deadline)
    max_corners = inventory.connectors.get(ConnectorType.RIGHT, 0)
    seen = set()
    best_geometry = -math.inf

    def propose(polygon, angle, step, notch_size=None):
        nonlocal best_geometry
        if polygon is None or time.process_time() >= deadline:
            return False
        # Smoothing and notches only remove area. Even an ideal circle of the
        # original area cannot beat this area/perimeter bound. The maximum
        # over smaller areas is at an endpoint because -sqrt(area) is convex.
        area = polygon.area
        ideal_geometry = weights.A * area + weights.C * 2 * math.sqrt(math.pi * area)
        if max(0.0, ideal_geometry) + weights.G < best_geometry - 1e-7:
            return False
        polygon = _break_long_faces(
            polygon,
            min(deadline, time.process_time() + 0.8),
            notch_size=notch_size or min(step or 5, 10),
        )
        if polygon is None:
            return False
        polygon = _smooth_short_faces(
            polygon, inventory, min(deadline, time.process_time() + 1.0)
        )
        points = _corners(polygon)
        if len(points) > max_corners or polygon.length > packer.total_length + TOL:
            return False
        geometry_score = weights.A * polygon.area + weights.C * polygon.length
        # A weaker geometric score can still win by obtaining the gate bonus.
        if geometry_score + weights.G < best_geometry - 1e-7:
            return False
        world = rotate(Polygon(points), angle, origin=(0, 0))
        rounded = [(round(x, 6), round(y, 6)) for x, y in _corners(world)]
        first = min(range(len(rounded)), key=rounded.__getitem__)
        key = tuple(rounded[first:] + rounded[:first])
        if key in seen:
            return False
        seen.add(key)
        stats.candidates += 1
        construction = packer.pack(
            list(world.exterior.coords[:-1]),
            deadline=min(deadline, time.process_time() + 0.3),
        )
        if construction is None:
            return False
        stats.constructions += 1
        best_geometry = max(best_geometry, geometry_score)
        consider(construction)
        return True

    # Small cells follow narrow passages; larger cells conserve corners and
    # pieces. Integer scales preserve integer side lengths after any rotation.
    abundant_lengths = sorted(
        (
            (count, length)
            for length, count in inventory.walls.items()
            if 5 <= length <= 15 and count > 0
        ),
        reverse=True,
    )
    scales = list(
        dict.fromkeys(
            [10, 5]
            + [length for _, length in abundant_lengths]
            + [7, 8, 6, 12, 15, 20, 25, 30]
        )
    )
    # Uniform grids can require hundreds of the same short wall. Mixing
    # column and row widths distributes those compulsory faces across stock.
    short_stock = [
        (length, count) for count, length in abundant_lengths if length <= 10
    ]
    mixed_widths = []
    if len(short_stock) >= 2:
        smallest_count = min(count for _, count in short_stock)
        for length, count in sorted(short_stock):
            mixed_widths.extend(
                [length] * min(3, max(1, round(count / smallest_count)))
            )
    mixed = [tuple(mixed_widths), tuple(reversed(mixed_widths))] if mixed_widths else []
    configurations = [
        (angle, step, offset)
        for offset in (0.0, 0.5)
        for step in scales[:2] + mixed + scales[2:]
        for angle in _orientations(room)
    ]
    for angle, step, offset in configurations:
        if time.process_time() >= deadline:
            break
        local_room = rotate(room.polygon, -angle, origin=(0, 0))
        cells, axes = _rasterize(local_room, step, offset)
        actual_step = None if isinstance(step, tuple) else step
        stats.grids += 1
        for component in _components(cells)[:3]:
            if time.process_time() >= deadline:
                break
            outline = _outline(component, axes)
            notch_sizes = [min(actual_step or 5, 10)]
            if actual_step == MIN_WALL_LENGTH:
                # Larger notch legs can spare scarce minimum-length walls
                # for the room's fine boundary instead of consuming them in
                # artificial jogs along otherwise straight runs.
                notch_sizes += [
                    length
                    for length, count in sorted(inventory.walls.items())
                    if 6 <= length <= 8 and count > inventory.walls.get(5, 0)
                ]
            full_fits = False
            for notch_size in notch_sizes:
                full_fits = (
                    propose(outline, angle, actual_step, notch_size) or full_fits
                )
            if full_fits or len(component) < 4:
                continue
            # These two ends create different prefixes in winding rooms.
            for seed in (min(component), max(component)):
                order = list(_growth_order(component, seed))
                sizes = sorted(
                    {
                        max(1, round(len(order) * fraction))
                        for fraction in (0.25, 0.5, 0.75)
                    },
                    reverse=True,
                )
                for size in sizes:
                    if time.process_time() >= deadline:
                        break
                    propose(_outline(order[:size], axes), angle, actual_step)
    return stats
