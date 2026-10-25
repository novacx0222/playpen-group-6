"""Filling a big room with grid cells.

Lay a grid of w-by-h cells over the room, keep every cell that lies wholly inside it,
and take the outline of the biggest connected block of kept cells as the enclosure.
Its sides run along the grid, east-west ones a whole number of w long and north-south
ones a whole number of h, so it closes exactly and every side is a length the
inventory can make if those multiples are. A stepped edge along a slanting wall is
mostly one-cell sides: with w and h different, those draw on two piece lengths
instead of using up one. Two fixes make the outline legal:

- two cells that meet only at a corner would make the outline touch itself, so one
  of them is dropped;
- a straight side longer than the 30-unit face limit gets a notch: one cell along it
  is dropped, which puts two corners in the side.

Every corner takes a right-angle connector, and a stepped edge has a great many. When
the stock is short, cells are dropped until the outline fits it: on a staircase,
dropping the top cell of a one-cell step merges two steps into one and saves two
corners, so each drop is chosen to save the most corners.

Growing a shape one move at a time (freeform.py) covers a big room slowly; this
covers it in one go, at the price of a stepped edge along walls that are not lined
up with the grid.
"""

import itertools
import math
import time
from collections import deque

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

import src.constants as c

NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))
WALL_OFFSETS = 4  # grid offsets tried per cell size, wall-aligned ones included


def _component(cells: set) -> set:
    """The biggest 4-connected block of cells."""
    left = set(cells)
    best: set = set()
    while left:
        first = left.pop()
        block = {first}
        queue = deque([first])
        while queue:
            x, y = queue.popleft()
            for dx, dy in NEIGHBOURS:
                q = (x + dx, y + dy)
                if q in left:
                    left.remove(q)
                    block.add(q)
                    queue.append(q)
        if len(block) > len(best):
            best = block
    return best


def _connected(cells: set) -> bool:
    if not cells:
        return False
    first = next(iter(cells))
    seen = {first}
    queue = deque([first])
    while queue:
        x, y = queue.popleft()
        for dx, dy in NEIGHBOURS:
            q = (x + dx, y + dy)
            if q in cells and q not in seen:
                seen.add(q)
                queue.append(q)
    return len(seen) == len(cells)


def _pinches(cells: set, around=None) -> list:
    """Cells that meet another kept cell only at a corner, with both cells that
    would join them missing."""
    found = []
    for x, y in around if around is not None else cells:
        if (x, y) not in cells:
            continue
        for dx, dy in ((1, 1), (1, -1)):
            if (
                (x + dx, y + dy) in cells
                and (x + dx, y) not in cells
                and (x, y + dy) not in cells
            ):
                found.append((x, y))
    return found


def _drop_pinches(cells: set) -> None:
    while True:
        pinched = _pinches(cells)
        if not pinched:
            return
        cells.discard(pinched[0])


def _runs(cells: set) -> tuple[list, list]:
    """Every straight stretch of outline, as (runs along rows, runs along columns),
    each run the cells beside it, in order."""
    rows: list = []
    columns: list = []
    # horizontal outline: below or above a row of cells
    for side in (-1, 1):
        edge = sorted((y, x) for x, y in cells if (x, y + side) not in cells)
        run: list = []
        for y, x in edge:
            if run and run[-1] == (x - 1, y):
                run.append((x, y))
            else:
                if run:
                    rows.append(run)
                run = [(x, y)]
        if run:
            rows.append(run)
    # vertical outline: left or right of a column of cells
    for side in (-1, 1):
        edge = sorted((x, y) for x, y in cells if (x + side, y) not in cells)
        run = []
        for x, y in edge:
            if run and run[-1] == (x, y - 1):
                run.append((x, y))
            else:
                if run:
                    columns.append(run)
                run = [(x, y)]
        if run:
            columns.append(run)
    return rows, columns


def _break_long_sides(cells: set, most_x: int, most_y: int, deadline: float) -> bool:
    """Notch every side longer than the limit (`most_x` cells along a row, `most_y`
    along a column). False if one can't be."""
    while time.process_time() < deadline:
        rows, columns = _runs(cells)
        long = [(run, most_x) for run in rows if len(run) > most_x]
        long += [(run, most_y) for run in columns if len(run) > most_y]
        if not long:
            return True
        for run, most in long:
            # drop one cell no more than `most` cells along, keeping the block
            # in one piece and without making a corner-to-corner touch
            for k in range(most, 0, -1):
                cell = run[k]
                cells.discard(cell)
                x, y = cell
                near = [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
                if not _pinches(cells, near) and _connected(cells):
                    break
                cells.add(cell)
            else:
                return False
    return False


def _faces(poly: Polygon) -> tuple[tuple[float, float], list]:
    """The outline as (first corner, [(heading in 45-degree steps, length)])."""
    ring = list(orient(poly, 1.0).exterior.coords)[:-1]
    points = []
    n = len(ring)
    for i in range(n):
        a, b, d = ring[i - 1], ring[i], ring[(i + 1) % n]
        cross = (b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0])
        if abs(cross) > 1e-9:
            points.append(b)
    faces = []
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        dx, dy = q[0] - p[0], q[1] - p[1]
        heading = round(math.degrees(math.atan2(dy, dx)) / 45.0) % 8
        faces.append((heading, round(math.hypot(dx, dy))))
    return points[0], faces


def _corner_count(cells: set, x: int, y: int) -> int:
    """Whether the grid point at the lower-left of cell (x, y) is a corner of the
    outline: it is when an odd number of the four cells around it are kept."""
    around = (
        ((x - 1, y - 1) in cells)
        + ((x, y - 1) in cells)
        + ((x - 1, y) in cells)
        + ((x, y) in cells)
    )
    return around % 2


def _corners_total(cells: set) -> int:
    points = {(x + dx, y + dy) for x, y in cells for dx in (0, 1) for dy in (0, 1)}
    return sum(_corner_count(cells, x, y) for x, y in points)


def _corner_saving(cells: set, cell: tuple[int, int]) -> int:
    """How many corners dropping `cell` removes (negative if it adds some)."""
    x, y = cell
    points = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
    before = sum(_corner_count(cells, px, py) for px, py in points)
    cells.discard(cell)
    after = sum(_corner_count(cells, px, py) for px, py in points)
    cells.add(cell)
    return before - after


def _fit_corners(
    cells: set, budget: int, most_x: int, most_y: int, deadline: float
) -> bool:
    """Drop cells, those saving the most corners first, until the outline has no
    more corners than `budget`. False if it can't be done in time."""
    total = _corners_total(cells)
    while total > budget:
        if time.process_time() > deadline:
            return False
        edge = [
            (x, y)
            for x, y in cells
            if any((x + dx, y + dy) not in cells for dx, dy in NEIGHBOURS)
        ]
        ranked = sorted(
            ((_corner_saving(cells, cell), cell) for cell in edge), reverse=True
        )
        for saving, cell in ranked:
            if saving <= 0:
                return False
            cells.discard(cell)
            x, y = cell
            near = [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
            rows, columns = _runs(cells)
            fine = (
                not _pinches(cells, near)
                and all(len(run) <= most_x for run in rows)
                and all(len(run) <= most_y for run in columns)
                and _connected(cells)
            )
            if fine:
                total -= saving
                break
            cells.add(cell)
        else:
            return False
    return True


def _offsets(local, low, w, h) -> list[tuple[float, float]]:
    """Where to start the grid: the usual two offsets (flush with the room's
    lowest corner, and half a cell in), then offsets that put cell edges on the
    room's longest walls along the grid's axes, so the outline can run flush
    along them (the idea of an adaptive lattice, which puts nodes on the walls)."""
    weight_x: dict[float, float] = {}
    weight_y: dict[float, float] = {}
    for (x1, y1), (x2, y2) in itertools.pairwise(local):
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 1e-9:
            continue
        if abs(x2 - x1) <= 1e-6 * length:  # a wall along the grid's y axis
            key = round((low[0] - x1) % w, 6) % w
            weight_x[key] = weight_x.get(key, 0.0) + length
        elif abs(y2 - y1) <= 1e-6 * length:  # along the x axis
            key = round((low[1] - y1) % h, 6) % h
            weight_y[key] = weight_y.get(key, 0.0) + length
    best_x = sorted(weight_x, key=lambda k: -weight_x[k])[:2]
    best_y = sorted(weight_y, key=lambda k: -weight_y[k])[:2]
    out = [(0.0, 0.0), (0.5 * w, 0.5 * h)]
    for ox in best_x or [0.0]:
        for oy in best_y or [0.0]:
            if all(abs(ox - a) > 1e-6 or abs(oy - b) > 1e-6 for a, b in out):
                out.append((ox, oy))
    return out[:WALL_OFFSETS]


def raster_loops(
    region,
    polygon,
    headings,
    cell_sizes,
    deadline,
    corner_budget=None,
):
    """Outlines of blocks of grid cells that fill the room, as
    (start, heading, faces), with faces as (heading in 45-degree steps from
    `heading`, length). `region` is the room buffered by the containment
    tolerance and prepared; `polygon` is the room itself."""
    corners = np.array(polygon.exterior.coords)
    for heading in headings:
        t = math.radians(heading)
        cos, sin = math.cos(t), math.sin(t)
        # the room as seen from the grid's frame
        local = corners @ np.array([[cos, -sin], [sin, cos]])
        low, high = local.min(axis=0), local.max(axis=0)
        for w, h in cell_sizes:
            most_x = int(c.MAX_FACE_LENGTH // w)
            most_y = int(c.MAX_FACE_LENGTH // h)
            for ox, oy in _offsets(local, low, w, h):
                if time.process_time() > deadline:
                    return
                xs = np.arange(low[0] - ox, high[0], w)
                ys = np.arange(low[1] - oy, high[1], h)
                gx, gy = np.meshgrid(xs, ys, indexing="ij")
                x0, y0 = gx.ravel(), gy.ravel()
                square = np.array([[0, 0], [w, 0], [w, h], [0, h], [0, 0]])
                local_rings = np.stack([x0, y0], axis=1)[:, None, :] + square[None]
                world = np.stack(
                    [
                        local_rings[..., 0] * cos - local_rings[..., 1] * sin,
                        local_rings[..., 0] * sin + local_rings[..., 1] * cos,
                    ],
                    axis=-1,
                )
                inside = shapely.contains(region, shapely.polygons(world))
                ix = np.round((x0 - xs[0]) / w).astype(int)
                iy = np.round((y0 - ys[0]) / h).astype(int)
                cells = {(int(a), int(b)) for a, b, ok in zip(ix, iy, inside) if ok}
                cells = _component(cells)
                if len(cells) < 4:
                    continue
                _drop_pinches(cells)
                cells = _component(cells)
                if not _break_long_sides(cells, most_x, most_y, deadline):
                    continue
                if corner_budget is not None and not _fit_corners(
                    cells, corner_budget, most_x, most_y, deadline
                ):
                    continue
                if len(cells) < 4:
                    continue
                boxes = [
                    shapely.box(a * w, b * h, (a + 1) * w, (b + 1) * h)
                    for a, b in cells
                ]
                block = shapely.union_all(boxes)
                if block.geom_type != "Polygon" or block.interiors:
                    continue
                first, faces = _faces(block)
                # back to the room: the grid's origin is (xs[0], ys[0])
                lx = xs[0] + first[0]
                ly = ys[0] + first[1]
                start = (lx * cos - ly * sin, lx * sin + ly * cos)
                yield start, heading, faces
