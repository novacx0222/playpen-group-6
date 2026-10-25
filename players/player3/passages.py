"""Route-aligned, inventory-length proposals for entering narrow openings."""

import math
import time

from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep

from src.constants import TOL


def passage_patches(polygon, face, lengths, route, room, deadline=math.inf):
    """Aim partial faces at the next route cells, including off-center openings.

    Cross sections estimate the throat width; exact room containment checks
    every proposed rectangle. Integer offsets retain integer-length remnants
    on the original face. This is guidance, not proof of stock feasibility.
    """
    if route is None or not lengths:
        return
    a, b = face
    size = math.dist(a, b)
    ux, uy = (b[0] - a[0]) / size, (b[1] - a[1]) / size
    nx, ny = uy, -ux
    prepared = prep(polygon.buffer(TOL))
    ahead = []
    for point in route.path:
        forward = (point[0] - a[0]) * nx + (point[1] - a[1]) * ny
        if forward > TOL and not prepared.covers(Point(point)):
            ahead.append(point)
        if len(ahead) == 6:
            break
    if not ahead:
        return
    forward = [(p[0] - a[0]) * nx + (p[1] - a[1]) * ny for p in ahead]
    advance = min(max(lengths), max(min(lengths), max(forward)))
    guide = ahead[0]
    lateral = max(0.0, min(size, (guide[0] - a[0]) * ux + (guide[1] - a[1]) * uy))
    prepared_room = prep(room.buffer(TOL))
    # If the whole face can advance, keep ordinary growth. A narrow partial
    # placement alone is not enough: its leftover face lengths may be unusable.
    probe_depth = min(lengths, key=lambda d: abs(d - advance))
    probe = Polygon(
        (
            a,
            b,
            (b[0] + probe_depth * nx, b[1] + probe_depth * ny),
            (a[0] + probe_depth * nx, a[1] + probe_depth * ny),
        )
    )
    if prepared_room.covers(probe):
        return

    def sections(depth):
        start = (a[0] + depth * nx, a[1] + depth * ny)
        end = (b[0] + depth * nx, b[1] + depth * ny)
        clipped = room.intersection(LineString((start, end)))
        parts = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
        intervals = []
        for part in parts:
            if part.is_empty or part.geom_type != "LineString":
                continue
            positions = [
                (x - start[0]) * ux + (y - start[1]) * uy for x, y in part.coords
            ]
            low, high = max(0.0, min(positions)), min(size, max(positions))
            if high - low >= min(lengths) - TOL:
                intervals.append((low, high))
        intervals.sort(
            key=lambda interval: abs((interval[0] + interval[1]) / 2 - lateral)
        )
        return intervals

    throat = size
    probes = sorted(
        {min(lengths), advance, *[d for d in forward if min(lengths) <= d <= advance]}
    )
    for depth in probes:
        if time.process_time() >= deadline:
            return
        intervals = sections(depth)
        if intervals:
            throat = min(throat, intervals[0][1] - intervals[0][0])
    if throat >= size - TOL:
        return  # Ordinary face growth already has room; avoid extra branching.
    widths = sorted(
        (n for n in lengths if n <= throat + TOL and n <= size + TOL), reverse=True
    )
    depths = sorted(lengths, key=lambda depth: (abs(depth - advance), -depth))
    for depth in depths:
        if time.process_time() >= deadline:
            return
        for low, high in sections(depth):
            for width in widths:
                first, last = math.ceil(low - TOL), math.floor(high - width + TOL)
                offsets = sorted(
                    range(first, last + 1),
                    key=lambda offset: abs(offset + width / 2 - lateral),
                )
                for offset in offsets:
                    if time.process_time() >= deadline:
                        return
                    p = (a[0] + offset * ux, a[1] + offset * uy)
                    q = (p[0] + width * ux, p[1] + width * uy)
                    patch = Polygon(
                        (
                            p,
                            q,
                            (q[0] + depth * nx, q[1] + depth * ny),
                            (p[0] + depth * nx, p[1] + depth * ny),
                        )
                    )
                    if (
                        prepared_room.covers(patch)
                        and patch.difference(polygon).area > TOL
                    ):
                        yield patch
