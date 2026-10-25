"""Choosing grid cells under the inventory's budget.

raster.py fills the whole room with cells and then trims the outline until the
stock can build it. When the stock is the limit (too little wall for the room,
too few corners) the best enclosure is often nowhere near "the whole room,
trimmed": it is two rooms joined through a five-unit gap, or a strip run down a
corridor into the next chamber. This searches for it directly:

- the grid is lined up with one of the room's wall directions, and its lines sit
  on the walls along that direction (a cell side flush with a wall wastes
  nothing) as well as every `step` units in between, so cells come in uneven
  widths where the walls call for them;
- a set of cells is changed one cell at a time, adding a cell next to it or
  dropping one from its edge, by simulated annealing on the score;
- the set must stay one piece, with no holes and no two cells meeting only at a
  corner, so its outline is a simple polygon. Its area, its perimeter and its
  holes (by Euler's formula) are kept up to date cell by cell;
- what the stock can't do is charged for, not forbidden, and the charge rises
  with time: a side shorter than the shortest piece or longer than the face
  limit, one no pieces add up to, more wall than there is, more corners than
  there are connectors. Only sets with nothing to charge for are kept.

Sides run along grid lines whose spacing is a whole number of units, so every
side is a whole number long and the outline closes exactly.
"""

import itertools
import math
import random
import time
from collections import deque

import numpy as np
import shapely

import src.constants as c

NEIGHBOURS = ((1, 0), (0, 1), (-1, 0), (0, -1))
MAX_CELLS = 40000  # a grid with more cells than this is too fine to search
REACH = 2  # adaptive grids put lines this many shortest lengths from each wall
KEEP = 3  # best distinct cell sets kept from each grid
SEED_TRIES = 50  # random corners tried for a fresh start's rectangle
RESTART_AFTER = 4000  # moves without a new best before starting again
COOLING = 1e-3  # end temperature as a share of the starting one
# time for each grid: the room's main wall direction first, adaptive lines most
HEADING_WEIGHTS = (4, 2, 1)
ROUNDS = 12  # coolings per grid
FROM_BEST = 0.3  # share of fresh starts that begin from one of the best sets
ADAPTIVE_WEIGHT = 4
CHARGE_GROWTH = 30.0  # how much the charge per unit rises over a grid's time


def _lines(
    lo: float, hi: float, anchor: float, step: int, walls: list[float], reach: tuple
) -> list:
    """Grid lines from `lo` to `hi`: `anchor` plus whole numbers of units, every
    `step`, on each side of every wall, and `reach` units from it either way (a
    6-unit corridor holds a 5-unit side only on a line 5 from one of its walls)."""
    first = math.floor(lo - anchor) - 1
    last = math.ceil(hi - anchor) + 1
    keep = set(range(first - first % step, last + step, step))
    for w in walls:
        below = math.floor(w - anchor + 1e-6)
        above = math.ceil(w - anchor - 1e-6)
        keep.update((below, above))
        for r in reach:
            keep.update((above + r, below - r))
    return sorted(anchor + k for k in keep if first <= k <= last)


class Grid:
    """Cells between consecutive lines, in a frame turned by `heading` degrees."""

    def __init__(
        self, region, polygon, heading: float, step: int, adaptive: bool, reach=()
    ):
        t = math.radians(heading)
        self.heading = heading
        self.cos, self.sin = math.cos(t), math.sin(t)
        corners = np.array(polygon.exterior.coords)
        local = corners @ np.array([[self.cos, -self.sin], [self.sin, self.cos]])
        low, high = local.min(axis=0), local.max(axis=0)
        # walls along the grid's axes, longest first
        along_x: dict[float, float] = {}  # y of walls running along x
        along_y: dict[float, float] = {}
        for (x1, y1), (x2, y2) in itertools.pairwise(local):
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 1e-9:
                continue
            if abs(y2 - y1) <= 1e-6 * length:
                along_x[round(y1, 6)] = along_x.get(round(y1, 6), 0.0) + length
            elif abs(x2 - x1) <= 1e-6 * length:
                along_y[round(x1, 6)] = along_y.get(round(x1, 6), 0.0) + length
        ax = max(along_y, key=along_y.get) if along_y else low[0]
        ay = max(along_x, key=along_x.get) if along_x else low[1]
        walls_x = list(along_y) if adaptive else []
        walls_y = list(along_x) if adaptive else []
        reach = reach if adaptive else ()
        self.xs = _lines(low[0], high[0], ax, step, walls_x, reach)
        self.ys = _lines(low[1], high[1], ay, step, walls_y, reach)
        self.inside: set = set()
        nx, ny = len(self.xs) - 1, len(self.ys) - 1
        if nx * ny > MAX_CELLS or nx < 1 or ny < 1:
            return
        xs, ys = np.array(self.xs), np.array(self.ys)
        gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        gi, gj = gi.ravel(), gj.ravel()
        x0, x1, y0, y1 = xs[gi], xs[gi + 1], ys[gj], ys[gj + 1]
        rings = np.stack(
            [
                np.stack([x0, x1, x1, x0, x0], axis=1),
                np.stack([y0, y0, y1, y1, y0], axis=1),
            ],
            axis=-1,
        )
        world = np.stack(
            [
                rings[..., 0] * self.cos - rings[..., 1] * self.sin,
                rings[..., 0] * self.sin + rings[..., 1] * self.cos,
            ],
            axis=-1,
        )
        ok = shapely.contains(region, shapely.polygons(world))
        self.inside = {(int(i), int(j)) for i, j, k in zip(gi, gj, ok) if k}

    def width(self, i: int) -> float:
        return self.xs[i + 1] - self.xs[i]

    def height(self, j: int) -> float:
        return self.ys[j + 1] - self.ys[j]

    def world(self, x: float, y: float) -> tuple[float, float]:
        return (x * self.cos - y * self.sin, x * self.sin + y * self.cos)


class CellSet:
    """A set of cells with its area, perimeter and Euler number kept up to date."""

    def __init__(self, grid: Grid):
        self.grid = grid
        self.cells: set = set()
        self.order: list = []  # the same cells, for picking one at random
        self.index: dict = {}
        self.vertices: dict = {}  # grid point -> how many kept cells touch it
        self.area = 0.0
        self.perimeter = 0.0
        self.euler = 0  # vertices - edges + cells: 1 for one piece with no holes

    def _sides(self, cell):
        i, j = cell
        g = self.grid
        w, h = g.width(i), g.height(j)
        return (((i + 1, j), h), ((i, j + 1), w), ((i - 1, j), h), ((i, j - 1), w))

    @staticmethod
    def _points(cell):
        i, j = cell
        return ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))

    def add(self, cell) -> None:
        shared = 0
        for other, length in self._sides(cell):
            if other in self.cells:
                shared += 1
                self.perimeter -= length
            else:
                self.perimeter += length
        new_points = 0
        for p in self._points(cell):
            n = self.vertices.get(p, 0)
            new_points += n == 0
            self.vertices[p] = n + 1
        self.cells.add(cell)
        self.index[cell] = len(self.order)
        self.order.append(cell)
        self.area += self.grid.width(cell[0]) * self.grid.height(cell[1])
        self.euler += new_points - (4 - shared) + 1

    def remove(self, cell) -> None:
        self.cells.discard(cell)
        k = self.index.pop(cell)
        last = self.order.pop()
        if last != cell:
            self.order[k] = last
            self.index[last] = k
        shared = 0
        for other, length in self._sides(cell):
            if other in self.cells:
                shared += 1
                self.perimeter += length
            else:
                self.perimeter -= length
        gone = 0
        for p in self._points(cell):
            n = self.vertices[p] - 1
            if n:
                self.vertices[p] = n
            else:
                del self.vertices[p]
                gone += 1
        self.area -= self.grid.width(cell[0]) * self.grid.height(cell[1])
        self.euler -= gone - (4 - shared) + 1

    def pinched(self, cell) -> bool:
        """Whether a corner of `cell` has two kept cells meeting only there."""
        cells = self.cells
        for px, py in self._points(cell):
            a = (px - 1, py - 1) in cells
            b = (px, py - 1) in cells
            d = (px - 1, py) in cells
            e = (px, py) in cells
            if (a and e and not b and not d) or (b and d and not a and not e):
                return True
        return False

    def connected(self) -> bool:
        cells = self.cells
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

    def outline(self) -> list[tuple[int, float, tuple[int, int]]]:
        """The outline walked anticlockwise, as (heading 0..3, length, start)."""
        cells = self.cells
        step = {}
        for i, j in cells:
            if (i, j - 1) not in cells:
                step[(i, j)] = ((i + 1, j), 0)
            if (i + 1, j) not in cells:
                step[(i + 1, j)] = ((i + 1, j + 1), 1)
            if (i, j + 1) not in cells:
                step[(i + 1, j + 1)] = ((i, j + 1), 2)
            if (i - 1, j) not in cells:
                step[(i, j + 1)] = ((i, j), 3)
        g = self.grid
        start = min(step)
        # begin at a corner, so the first side is whole
        p = start
        while True:
            q, h = step[p]
            if step[q][1] != h:
                break
            p = q
            if p == start:
                break
        first = step[p][0]
        sides = []
        p = first
        while True:
            q, h = step[p]
            if h % 2 == 0:
                length = abs(g.xs[q[0]] - g.xs[p[0]])
            else:
                length = abs(g.ys[q[1]] - g.ys[p[1]])
            if sides and sides[-1][0] == h:
                sides[-1] = (h, sides[-1][1] + length, sides[-1][2])
            else:
                sides.append((h, length, p))
            p = q
            if p == first:
                break
        return sides


class CellSearch:
    def __init__(
        self,
        buildable: set[int],
        rights: int,
        total_length: int,
        area_weight: float,
        perimeter_weight: float,
        can_build,
        rng: random.Random,
    ):
        self.buildable = buildable
        self.rights = rights
        self.total_length = total_length
        self.A = area_weight
        self.C = perimeter_weight
        self.can_build = can_build
        self.rng = rng
        lengths = sorted(buildable)
        self.shortest = lengths[0]
        self.reach = tuple(lengths[:REACH])
        self.miss = [0] * (4 * c.MAX_FACE_LENGTH + 1)
        for n in range(len(self.miss)):
            if n > c.MAX_FACE_LENGTH:
                self.miss[n] = n - c.MAX_FACE_LENGTH
            elif n not in buildable:
                self.miss[n] = min(abs(n - b) for b in lengths)

    def _charge(self, state: CellSet) -> tuple[float, list | None]:
        """Units the stock can't cover, and the outline when that is zero."""
        sides = state.outline()
        over = 0.0
        for _, length, _ in sides:
            n = round(length)
            over += self.miss[n] if n < len(self.miss) else n
        over += max(0, len(sides) - self.rights)
        over += max(0.0, state.perimeter - self.total_length)
        return over, sides if over == 0 else None

    def _block(self, grid: Grid, corner, si: int, sj: int, allowed) -> list | None:
        """Cells of a rectangle from `corner`, going `si` along the rows and `sj`
        along the columns, whose sides are both buildable lengths: the smallest,
        or now and then the next one up. None unless every cell is `allowed`."""
        rng = self.rng
        i0, j0 = corner
        cols: list[int] = []
        width = 0.0
        i = i0
        while True:
            if not allowed((i, j0)):
                return None
            cols.append(i)
            width += grid.width(i)
            n = round(width)
            if n > c.MAX_FACE_LENGTH:
                return None
            if n in self.buildable and rng.random() < 0.7:
                break
            i += si
        rows: list[int] = []
        height = 0.0
        j = j0
        while True:
            if not all(allowed((k, j)) for k in cols):
                return None
            rows.append(j)
            height += grid.height(j)
            n = round(height)
            if n > c.MAX_FACE_LENGTH:
                return None
            if n in self.buildable and rng.random() < 0.7:
                return [(k, m) for k in cols for m in rows]
            j += sj

    def run(self, grid: Grid, deadline: float) -> list[tuple]:
        """The best cell sets found on `grid`, as (value, outline sides)."""
        inside = grid.inside
        if not inside:
            return []
        rng = self.rng
        pool = sorted(inside)
        best: list[tuple] = []  # (value, sides, frozenset)
        # the temperature starts at what one shortest-side square is worth, and
        # so does the charge for a unit the stock can't cover; the charge then
        # rises so that what is left at the end can be built
        t0 = max(abs(self.A) * self.shortest**2, 1e-6)
        began = time.process_time()
        span = max(deadline - began, 1e-6)

        def fresh() -> CellSet:
            """Half the time the best set so far; otherwise a rectangle of cells
            from a random cell, its sides buildable lengths: on a grid of thin
            cells a lone cell is all charge, and annealing from it rarely finds
            its way to a set with nothing to charge for."""
            state = CellSet(grid)
            if best and rng.random() < FROM_BEST:
                for cell in rng.choice(best)[2]:
                    state.add(cell)
                return state
            for _ in range(SEED_TRIES):
                si, sj = rng.choice((-1, 1)), rng.choice((-1, 1))
                block = self._block(grid, rng.choice(pool), si, sj, inside.__contains__)
                if block:
                    for cell in block:
                        state.add(cell)
                    return state
            state.add(rng.choice(pool))
            return state

        def measure(state: CellSet) -> tuple[float, float, list | None]:
            over, sides = self._charge(state)
            return self.A * state.area + self.C * state.perimeter, over, sides

        def record(state: CellSet, value: float, sides) -> bool:
            """Keep the set if it is among the best that can be built."""
            if sides is None or (len(best) >= KEEP and value <= best[-1][0] + 1e-9):
                return False
            key = frozenset(state.cells)
            if any(key == b[2] for b in best):
                return False
            if not self.can_build(tuple(round(n) for _, n, _ in sides)):
                return False
            best.append((value, sides, key))
            best.sort(key=lambda b: -b[0])
            del best[KEEP:]
            return True

        state = fresh()
        value, over, sides = measure(state)
        record(state, value, sides)
        temperature, charge_unit = t0, t0
        since = 0
        moves = 0
        round_now = 0
        while True:
            moves += 1
            if moves % 64 == 0:
                now = time.process_time()
                if now >= deadline:
                    break
                # several rounds, each cooling all the way from a fresh start:
                # one long cooling settles on the first good region it finds
                done = (now - began) / span * ROUNDS
                if int(done) > round_now:
                    round_now = int(done)
                    state = fresh()
                    value, over, sides = measure(state)
                    record(state, value, sides)
                    since = 0
                done -= int(done)
                temperature = t0 * COOLING**done
                charge_unit = t0 * (1 + CHARGE_GROWTH * done)
            # a cell of the set: drop it, or add one of its neighbours, or the
            # same with a block of cells whose sides are buildable lengths (on a
            # grid of thin cells one cell at a time is all charge)
            cell = rng.choice(state.order)
            kind = rng.random()
            dx, dy = NEIGHBOURS[rng.randrange(4)]
            if kind < 0.5:
                adding = kind < 0.25
                if adding:
                    cell = (cell[0] + dx, cell[1] + dy)
                    if cell in state.cells or cell not in inside:
                        continue
                changed = [cell]
            else:
                adding = kind < 0.75
                si = dx or rng.choice((-1, 1))
                sj = dy or rng.choice((-1, 1))
                if adding:
                    cell = (cell[0] + dx, cell[1] + dy)
                    block = self._block(
                        grid,
                        cell,
                        si,
                        sj,
                        lambda q, cells=state.cells: q in inside and q not in cells,
                    )
                else:
                    block = self._block(grid, cell, si, sj, state.cells.__contains__)
                if not block:
                    continue
                changed = block
            if not adding and len(changed) >= len(state.order):
                continue
            for q in changed:
                (state.add if adding else state.remove)(q)
            fine = state.euler == 1 and not any(state.pinched(q) for q in changed)
            if fine and not adding:
                fine = state.connected()
            if fine:
                new_value, new_over, sides = measure(state)
                delta = (new_value - charge_unit * new_over) - (
                    value - charge_unit * over
                )
                fine = delta >= 0 or rng.random() < math.exp(delta / temperature)
            if not fine:
                for q in changed:
                    (state.remove if adding else state.add)(q)
                continue
            value, over = new_value, new_over
            since += 1
            if record(state, value, sides):
                since = 0
            if since > RESTART_AFTER:
                state = fresh()
                value, over, sides = measure(state)
                record(state, value, sides)
                since = 0
        return [(v, sides) for v, sides, _ in best]


def cell_loops(region, polygon, headings, steps, search: CellSearch, deadline: float):
    """Outlines of the best cell sets on each grid, as (start, heading, faces),
    with faces as (heading in 45-degree steps from `heading`, length).
    `headings` come longest walls first, and the first get the most time."""
    grids = [
        (heading, step, adaptive, HEADING_WEIGHTS[min(k, len(HEADING_WEIGHTS) - 1)]
         * (ADAPTIVE_WEIGHT if adaptive else 1))
        for k, heading in enumerate(headings)
        for adaptive in (True, False)
        for step in steps
    ]  # fmt: skip
    left = sum(g[3] for g in grids)
    for heading, step, adaptive, weight in grids:
        now = time.process_time()
        if now >= deadline:
            return
        grid = Grid(region, polygon, heading, step, adaptive, search.reach)
        slot = now + (deadline - now) * weight / left
        left -= weight
        for _, sides in search.run(grid, slot):
            _, _, (pi, pj) = sides[0]
            start = grid.world(grid.xs[pi], grid.ys[pj])
            faces = tuple((2 * h, round(n)) for h, n, _ in sides)
            yield start, heading, faces
