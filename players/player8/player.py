"""Group 8 player.

Tries a lot of candidate shapes (grid-traced outlines for big rooms,
rectangles, cut-corner octagons and hexagons, L/U/T shapes, and everything
right-angled grown by bumps), builds every side out of real inventory pieces,
fits the shape into the room and keeps the best valid one. Every candidate is
checked with the simulator's own validator, so we only ever return something
valid (or None). Phases are bounded by work, not by the clock, so the result
does not depend on how fast the machine is.
"""

import bisect
import heapq
import math
import random
import time

import numpy as np
import shapely
from shapely.geometry import Point, Polygon

from players.player0 import Player
from src.constants import MIN_WALL_LENGTH, TOL
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType

try:  # a prepared polygon makes the repeated containment tests a lot faster
    from shapely.prepared import prep
except ImportError:
    prep = None

# CPU seconds we give ourselves. The simulator's limit is 300 (the submission
# checker's 60 s smoke runs only report a timeout as a warning). Every phase is
# bounded by work, not by the clock, so the same input gives the same answer on
# any machine; the deadline is a safety net for machines far slower than ours.
TIME_BUDGET = 290.0
TEMPLATE_CHECKS = 5_000_000  # containment checks per template pass (two passes)
MAX_GRIDS = 150  # grids the outline phase may try (it normally finishes them all)
MAX_REFINE_SEEDS = (
    24  # root shapes grown by bumps, best first (the spiral needs its fifth)
)
POSE_SEEDS = 8  # best rectangles also grown from other places they fit ...
POSES_PER_SEED = 20  # ... this many places each
DOMINANCE_OK = {
    "rect",
    "oct",
    "hex",
    "L",
    "T",
    "stair",
    "stair2",
}  # larger parameters contain the smaller shape
# Random restarts keep paying off late (improvements measured at restart
# 20,000+ on some rooms), so the caps are generous; on this machine a room
# of restarts runs at 200-400 per second.
MAX_RESTARTS = 80_000  # random restarts in total ...
IDLE_RESTARTS = 20_000  # ... and consecutive restarts without an improvement

# Exact placement (Placer.exact_place): an angle is tested exactly when the room
# has few enough walls for it to be cheap; each test is charged this many
# containment checks against the template budget.
EXACT_LIMIT = 1000  # room walls x shape sides
EXACT_MAX_ANGLES = 32  # rotations tested exactly per shape (longest walls first)
EXACT_FAST_CHECKS = 1500  # anchored poses tried before the exact pass
EXACT_COST = 25  # containment checks charged per room edge of an exact fit test
MAX_FAILED_MEMO = 400
# families walked after all the others (their own best-first segment), so a
# large new family can never starve the proven ones of the template budget
LATE_FAMILIES = {"stair", "stair2"}
MAX_PARTS = 6  # pieces on one face: 30 / 5
NODE_LIMIT = 4000  # backtracking budget for one allocation
RELAX_MAX_SIDES = 24  # the cheap feasibility check is only trusted up to this

WALL = "wall"
GATE = "gate"

# --- small geometry helpers ---------------------------------------------------


def _rot(x, y, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return c * x - s * y, s * x + c * y


def _area(pts):
    """Signed area (positive when the points run counter-clockwise)."""
    n = len(pts)
    return (
        sum(
            pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
            for i in range(n)
        )
        / 2
    )


def _inside(pts, x, y):
    """Ray casting point-in-polygon."""
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < xi + (y - yi) * (xj - xi) / (yj - yi):
            inside = not inside
        j = i
    return inside


def _seg_dist(x, y, edge):
    (ax, ay), (bx, by) = edge
    dx, dy = bx - ax, by - ay
    seg = dx * dx + dy * dy
    t = 0.0 if seg == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg))
    return math.hypot(x - ax - t * dx, y - ay - t * dy)


# --- shapes -------------------------------------------------------------------
# A shape is a closed counter-clockwise loop of sides, each one pointing in a
# multiple of 45 degrees and with an integer length. Diagonal sides move
# sqrt(2)/2 per unit, which is irrational, so a loop only closes when the axis
# sides cancel each other and the diagonal sides cancel each other:
#   E == W, N == S, NE == SW, NW == SE

_R2 = math.sqrt(0.5)
UNIT = [
    (1, 0),
    (_R2, _R2),
    (0, 1),
    (-_R2, _R2),
    (-1, 0),
    (-_R2, -_R2),
    (0, -1),
    (_R2, -_R2),
]

# turn between two sides (in 45 degree steps, counter-clockwise) -> connector
TURN_CONNECTOR = {
    1: Connector(ConnectorType.DIAGONAL),
    2: Connector(ConnectorType.RIGHT),
    6: Connector(ConnectorType.RIGHT, reflex=True),
    7: Connector(ConnectorType.DIAGONAL, reflex=True),
}
TURN_SIGNED = {1: 1, 2: 2, 6: -2, 7: -1}


class Shape:
    def __init__(self, family, params, dirs, lengths):
        self.family = family
        self.params = params
        self.dirs = dirs
        self.lengths = lengths
        n = len(dirs)
        self.turns = [(dirs[i] - dirs[i - 1]) % 8 for i in range(n)]
        self.right = sum(t in (2, 6) for t in self.turns)
        self.diagonal = sum(t in (1, 7) for t in self.turns)
        self.axis_only = all(d % 2 == 0 for d in dirs)
        self.perimeter = float(sum(lengths))

        pts = [(0.0, 0.0)]
        for d, length in zip(dirs, lengths):
            ux, uy = UNIT[d]
            pts.append((pts[-1][0] + ux * length, pts[-1][1] + uy * length))
        self.closes = math.hypot(*pts[-1]) < 1e-6
        self.vertices = pts[:-1]
        self.area = _area(self.vertices)
        self.turns_ok = all(t in TURN_CONNECTOR for t in self.turns) and (
            sum(TURN_SIGNED.get(t, 0) for t in self.turns) == 8
        )

    def valid(self):
        return self.turns_ok and self.closes and self.area > 0

    def symmetry(self):
        """The smallest rotation (90, 180 or 360 degrees) that maps the shape
        onto itself; rotations that differ by it give the same placements."""
        seq = list(zip(self.dirs, self.lengths))
        n = len(seq)
        for step, deg in ((2, 90), (4, 180)):
            rot = [((d + step) % 8, length) for d, length in seq]
            if any(rot[k:] + rot[:k] == seq for k in range(n)):
                return deg
        return 360

    def interior_angle(self, i):
        return 180.0 - 45.0 * TURN_SIGNED[self.turns[i]]

    def world_vertices(self, theta, tx, ty):
        return [
            (tx + x, ty + y) for x, y in (_rot(x, y, theta) for x, y in self.vertices)
        ]

    def construction(self, sides, theta, tx, ty):
        """Rotate by theta, move so the first vertex sits at (tx, ty), and lay
        out the pieces chosen for each side."""
        start = self.world_vertices(theta, tx, ty)[0]
        pieces, connectors = [], []
        for turn, side in zip(self.turns, sides):
            for j, (kind, length) in enumerate(side):
                piece_type = PieceType.GATE if kind == GATE else PieceType.WALL
                pieces.append(Piece(piece_type, length))
                if j == 0:
                    connectors.append(TURN_CONNECTOR[turn])
                else:
                    connectors.append(Connector(ConnectorType.STRAIGHT))
        return Construction(
            start=start,
            start_heading=theta + 45.0 * self.dirs[0],
            pieces=pieces,
            connectors=connectors,
        )


def rectangle(a, b):
    return Shape("rect", (a, b), [0, 2, 4, 6], [a, b, a, b])


def octagon(a, b, c, d):
    """Rectangle with all four corners cut (c on NE/SW, d on NW/SE)."""
    return Shape("oct", (a, b, c, d), list(range(8)), [a, c, b, d, a, c, b, d])


def hexagon(a, b, c):
    """Rectangle with two opposite corners cut."""
    return Shape("hex", (a, b, c), [0, 1, 2, 4, 5, 6], [a, c, b, a, c, b])


def hexagon_tilted(a, b, c):
    """Same idea as the hexagon but the cuts sit on the other diagonal."""
    return Shape("hex2", (a, b, c), [0, 2, 3, 4, 6, 7], [a, b, c, a, b, c])


def l_shape(a1, a2, b1, b2):
    """Bottom a1+a2 wide, right arm b1 high, left arm b1+b2 high and a1 wide."""
    return Shape(
        "L", (a1, a2, b1, b2), [0, 2, 4, 2, 4, 6], [a1 + a2, b1, a2, b2, a1, b1 + b2]
    )


def u_shape(a1, a2, a3, b1, b2):
    """U opening upwards: base a1+a2+a3 by b1, arms a1 and a3 going up b2 more."""
    return Shape(
        "U",
        (a1, a2, a3, b1, b2),
        [0, 2, 4, 6, 4, 2, 4, 6],
        [a1 + a2 + a3, b1 + b2, a3, b2, a2, b2, a1, b1 + b2],
    )


def t_shape(a1, a2, a3, b1, b2):
    """Base a1+a2+a3 by b1 with a stem (a2 wide, b2 high) over the middle."""
    return Shape(
        "T",
        (a1, a2, a3, b1, b2),
        [0, 2, 4, 2, 4, 6, 4, 6],
        [a1 + a2 + a3, b1, a3, b2, a2, b2, a1, b1],
    )


def z_shape(a, b, c, d):
    """Two a by b rectangles, the upper one shifted left by c and up by d (a Z).
    Needs c < a and d < b so that they overlap."""
    return Shape("Z", (a, b, c, d), [0, 2, 4, 2, 4, 6, 0, 6], [a, b, c, d, a, b, c, d])


def step_shape(a, b, c, d):
    """A thin snake: a bar a long and d thick, a bar b high and c thick hanging
    from its right end, and the same again rotated by 180 degrees, so that it
    threads corridors narrower than any rectangle that pays. Needs c < a and
    d < b."""
    return Shape(
        "step", (a, b, c, d), [0, 2, 4, 6, 4, 6, 0, 2], [a, b, c, d, a, b, c, d]
    )


def stair(p, q, c, s):
    """A p-wide block with its top-right corner cut by a diagonal c and its
    bottom-left corner cut by a diagonal c followed by a straight s: the two
    steps hug a wall that is steeper than 45 degrees (3 right and 4 diagonal
    connectors, one of them reflex)."""
    return Shape(
        "stair", (p, q, c, s), [4, 6, 7, 6, 0, 2, 3], [p, q, c, s, p, q + s, c]
    )


def stair2(p, q, c, s):
    """The mirror image of `stair` (the placer rotates but never mirrors)."""
    return Shape(
        "stair2", (p, q, c, s), [5, 6, 0, 2, 1, 2, 4], [c, q + s, p, s, c, q, p]
    )


def band(s, k, d, h):
    """A strip h high whose two long sides are k+1 runs of s separated by k
    inward V-dips with legs d (two diagonal connectors and one reflex right
    connector each): how a face longer than 30 is broken in a thin room."""
    dirs = [0, 1, 7] * k + [0, 2] + [4, 5, 3] * k + [4, 6]
    lengths = [s, d, d] * k + [s, h] + [s, d, d] * k + [s, h]
    return Shape("band", (s, k, d, h), dirs, lengths)


def band2(s1, s2, k, d, h):
    """A band whose runs are s1 and s2 (k = 1) or s1, s2, s1 (k = 2): the
    middle run is often the one that has to be a different length."""
    runs = [s1, s2] if k == 1 else [s1, s2, s1]
    dirs, lengths = [], []
    for j, r in enumerate(runs):
        if j:
            dirs += [1, 7]
            lengths += [d, d]
        dirs.append(0)
        lengths.append(r)
    dirs.append(2)
    lengths.append(h)
    for j, r in enumerate(reversed(runs)):
        if j:
            dirs += [5, 3]
            lengths += [d, d]
        dirs.append(4)
        lengths.append(r)
    dirs.append(6)
    lengths.append(h)
    return Shape("band2", (s1, s2, k, d, h), dirs, lengths)


TEMPLATES = {
    "rect": rectangle,
    "oct": octagon,
    "hex": hexagon,
    "hex2": hexagon_tilted,
    "L": l_shape,
    "U": u_shape,
    "T": t_shape,
    "Z": z_shape,
    "step": step_shape,
    "band": band,
    "band2": band2,
    "stair": stair,
    "stair2": stair2,
}

# --- piece allocation ---------------------------------------------------------


class Stock:
    """How many pieces of each length can be used as wall, plus a cache of the
    ways to make a given length out of them."""

    def __init__(self, counts):
        self.counts = counts
        self.lengths = sorted(counts, reverse=True)
        self.ways = {}
        self.closable = {}

    def partitions(self, total):
        """Every multiset of pieces adding up to total, fewest pieces first."""
        if total in self.ways:
            return self.ways[total]
        found = []

        def rec(left, start, chosen):
            if left == 0:
                found.append(tuple(chosen))
                return
            if len(chosen) >= MAX_PARTS:
                return
            for k in range(start, len(self.lengths)):
                n = self.lengths[k]
                if n > left or chosen.count(n) >= self.counts[n]:
                    continue
                chosen.append(n)
                rec(left - n, k, chosen)
                chosen.pop()

        if total > 0:
            rec(total, 0, [])
        # fewest pieces first, then prefer the lengths we have plenty of
        found.sort(key=lambda p: (len(p), sum(1.0 / self.counts[n] for n in p)))
        self.ways[total] = found
        return found

    def fewest(self, total):
        ways = self.partitions(total)
        return len(ways[0]) if ways else 99


def _usable(counts):
    """Pieces long enough to use, as {length: count}."""
    return {int(n): k for n, k in counts.items() if n >= MIN_WALL_LENGTH and k > 0}


class Allocator:
    """Turns the side lengths of a shape into actual pieces.

    Each side is built from one or more pieces joined by straight connectors.
    One gate (or two, on different sides, for the bonus) goes somewhere in the
    shape. Spare gates are real pieces too, so when plain walls can't make a
    side they can stand in as walls.
    """

    def __init__(self, walls, gates, straight):
        self.walls = _usable(walls)
        self.gates = _usable(gates)
        self.straight = max(0, straight)
        everything = dict(self.walls)
        for n, k in self.gates.items():
            everything[n] = everything.get(n, 0) + k
        self.plain = Stock(self.walls)
        self.pool = Stock(everything)
        # borrowing a gate for a one-gate enclosure isn't worth it when a
        # two-gate one (with the bonus) is on offer; the player switches it off
        self.borrow_for_single = True
        self.cache = {}

    def side_ok(self, total, stock=None):
        """Can a side this long be built at all, with or without a gate?"""
        stock = stock or self.plain
        if stock.partitions(total):
            return True
        return bool(self.gate_options(total, stock))

    def gate_options(self, total, stock=None):
        """Gate lengths that can sit on a side of this length."""
        stock = stock or self.plain
        return [
            g
            for g in sorted(self.gates, reverse=True)
            if g == total
            or (total - g >= MIN_WALL_LENGTH and stock.partitions(total - g))
        ]

    def allocate(self, sides, n_gates=1):
        """Pieces for every side (gate first on its side) or None."""
        key = (tuple(sides), n_gates)
        if key not in self.cache:
            self.cache[key] = self._search(list(sides), n_gates)
        return self.cache[key]

    def _search(self, sides, n_gates):
        if not self._closable(sides):
            return None
        stocks = [self.plain]
        spare = sum(self.gates.values()) - n_gates
        if spare >= 1 and (n_gates >= 2 or self.borrow_for_single):
            stocks.append(self.pool)
        for stock in stocks:
            found = self._place_gates(stock, sides, n_gates)
            if found is not None:
                return found
        return None

    def _closable(self, sides):
        """Necessary condition: the sides can be built from everything we have
        even if nothing is set aside as the gate. One cheap search that weeds
        out most hopeless shapes before the slow gate placement search."""
        if len(sides) > RELAX_MAX_SIDES:
            return True
        key = tuple(sorted(sides))
        if key not in self.pool.closable:
            filled = self._fill(
                self.pool,
                sides,
                dict(self.pool.counts),
                self.straight,
                [0] * len(sides),
            )
            self.pool.closable[key] = filled is not None
        return self.pool.closable[key]

    def _place_gates(self, stock, sides, n_gates):
        if n_gates < 1 or sum(self.gates.values()) < n_gates:
            return None
        if any(not self.side_ok(s, stock) for s in sides):
            return None

        # (side, gate length) options, a gate that is a whole side first
        options = []
        seen = set()
        for i, s in enumerate(sides):
            for g in self.gate_options(s, stock):
                if (s, g) in seen and n_gates == 1:
                    continue
                seen.add((s, g))
                options.append((0 if g == s else 1, -g, i, g))
        options.sort()

        tries = 0
        if n_gates == 1:
            for _, _, i, g in options[:12]:
                found = self._with_gates(stock, sides, {i: g})
                if found is not None:
                    return found
            return None

        for a in range(len(options)):
            for b in range(a + 1, len(options)):
                _, _, i, g = options[a]
                _, _, j, h = options[b]
                if i == j or (g == h and self.gates[g] < 2):
                    continue
                tries += 1
                if tries > 25:
                    return None
                found = self._with_gates(stock, sides, {i: g, j: h})
                if found is not None:
                    return found
        return None

    def _with_gates(self, stock, sides, gate_at):
        walls = dict(stock.counts)
        if stock is self.pool:
            for g in gate_at.values():
                walls[g] -= 1  # keep these for the gates themselves
        rest = [s - gate_at.get(i, 0) for i, s in enumerate(sides)]
        # a gate sharing its side with walls needs one more straight joint
        extra = [int(i in gate_at and rest[i] > 0) for i in range(len(sides))]
        filled = self._fill(stock, rest, walls, self.straight, extra)
        if filled is None:
            return None

        # use real walls first, spare gates only for what is left over
        walls_left = dict(self.walls)
        out = []
        for i, parts in enumerate(filled):
            side = [(GATE, gate_at[i])] if i in gate_at else []
            for n in parts:
                if walls_left.get(n, 0) > 0:
                    walls_left[n] -= 1
                    side.append((WALL, n))
                else:
                    side.append((GATE, n))
            out.append(side)
        return out

    def _fill(self, stock, lengths, walls, joints, extra):
        """Backtracking: wall pieces for every length (0 needs nothing), longest
        side first. `joints` is how many straight connectors we have left."""
        n = len(lengths)
        order = sorted(range(n), key=lambda i: -lengths[i])
        # fewest joints the sides from position k on will need
        need = [0] * (n + 1)
        for k in range(n - 1, -1, -1):
            i = order[k]
            own = 0 if lengths[i] == 0 else stock.fewest(lengths[i]) - 1 + extra[i]
            need[k] = need[k + 1] + own
        choice = [None] * n
        nodes = 0

        def rec(k, joints, prev_key, prev_idx):
            nonlocal nodes
            if k == n:
                return True
            nodes += 1
            if nodes > NODE_LIMIT or need[k] > joints:
                return False
            i = order[k]
            length = lengths[i]
            if length == 0:
                choice[i] = ()
                return rec(k + 1, joints, (), 0)
            ways = stock.partitions(length)
            key = (length, extra[i])
            # equal sides are interchangeable, so don't revisit earlier options
            for idx in range(prev_idx if key == prev_key else 0, len(ways)):
                way = ways[idx]
                cost = len(way) - 1 + extra[i]
                if cost > joints:
                    break
                taken = []
                for m in way:
                    if walls.get(m, 0) <= 0:
                        break
                    walls[m] -= 1
                    taken.append(m)
                if len(taken) == len(way):
                    choice[i] = way
                    if rec(k + 1, joints - cost, key, idx):
                        return True
                for m in taken:
                    walls[m] += 1
            return False

        if not rec(0, joints, (), 0):
            return None
        return [list(c) for c in choice]


# --- placement ----------------------------------------------------------------
# Good placements almost always touch the room's walls and line up with them, so
# we put each corner of the shape into each corner of the room with one side
# lying along a wall, then slide it along that wall. Only if none of that works
# do we sweep positions and angles.


def _angle(dx, dy):
    return math.degrees(math.atan2(dy, dx))


def _wrap(a):
    """Angle into (-180, 180]."""
    a = math.fmod(a, 360.0)
    if a <= -180:
        a += 360
    elif a > 180:
        a -= 360
    return a


class Placer:
    def __init__(self, room, out_of_time):
        self.room = room
        self.out_of_time = out_of_time
        pts = list(room.polygon.exterior.coords)[:-1]
        if _area(pts) < 0:
            pts.reverse()
        n = len(pts)
        self.pts = pts
        self.edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
        self.out_angle = [_angle(b[0] - a[0], b[1] - a[1]) for a, b in self.edges]
        self.in_angle = [self.out_angle[i - 1] for i in range(n)]
        self.interior = [
            180.0 - _wrap(self.out_angle[i] - self.in_angle[i]) for i in range(n)
        ]
        self.edge_len = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in self.edges]
        self.bounds = room.polygon.bounds
        padded = room.polygon.buffer(TOL)
        self.checker = prep(padded) if prep is not None else padded
        self.max_radius = self._max_radius()
        self.checks = 0  # containment tests so far: the template walk's unit of work
        self.diameter = max(
            math.hypot(a[0] - b[0], a[1] - b[1]) for a in pts for b in pts
        )
        # the walls' directions modulo 45 degrees (clustered to 0.05), longest
        # walls first: a shape side parallel to a wall is where tight fits are
        weight = {}
        for ang, length in zip(self.out_angle, self.edge_len):
            base = round((ang % 45.0) / 0.05) * 0.05
            if base >= 45.0 - 1e-9:
                base = 0.0
            weight[base] = weight.get(base, 0.0) + length
        self.wall_bases = [b for b, _ in sorted(weight.items(), key=lambda t: -t[1])]
        self._extents = {}  # direction -> the room's extent along it
        self._pts_arr = np.asarray(pts, dtype=float)
        self._edge_a = self._pts_arr
        self._edge_b = np.roll(self._pts_arr, -1, axis=0)

    def _max_radius(self):
        """Upper bound on the largest circle that fits in the room (sampled on a
        grid plus the sampling error). A shape whose own inner circle is bigger
        can't fit anywhere."""
        minx, miny, maxx, maxy = self.bounds
        steps = 70
        hx, hy = (maxx - minx) / steps, (maxy - miny) / steps
        best = 0.0
        for a in range(steps + 1):
            for b in range(steps + 1):
                x, y = minx + a * hx, miny + b * hy
                if _inside(self.pts, x, y):
                    best = max(best, min(_seg_dist(x, y, e) for e in self.edges))
        return best + math.hypot(hx, hy) / 2 + 1e-6

    def point_ok(self, x, y):
        """Inside the room or within a hair of its boundary (cheap pre-filter)."""
        return _inside(self.pts, x, y) or any(
            _seg_dist(x, y, e) <= 1e-3 for e in self.edges
        )

    def fits(self, world):
        self.checks += 1
        minx, miny, maxx, maxy = self.bounds
        for x, y in world:
            if not (
                minx - 1e-3 <= x <= maxx + 1e-3 and miny - 1e-3 <= y <= maxy + 1e-3
            ):
                return False
        if not all(self.point_ok(x, y) for x, y in world):
            return False
        return self.checker.contains(Polygon(world))

    def candidates(self, shape, step45):
        """(theta, tx, ty) placements to try, best guesses first: the anchored
        poses, then the coarse sweep."""
        yield from self._anchored(shape)
        yield from self._sweep(shape, step45)

    def _anchored(self, shape):
        """Poses with a shape corner in a room corner (a side along either
        wall), then a convex shape corner sliding along every wall, longest
        walls first. Cheap, and where most placements are."""
        n = len(shape.vertices)
        shape_out = [45.0 * d for d in shape.dirs]
        shape_in = [shape_out[i - 1] for i in range(n)]
        pairs = []
        for j in range(len(self.pts)):
            for i in range(n):
                gap = self.interior[j] - shape.interior_angle(i)
                if gap >= -1e-6:
                    pairs.append((gap, j, i))
        pairs.sort()
        for _, j, i in pairs:
            rx, ry = self.pts[j]
            vx, vy = shape.vertices[i]
            for theta in (
                self.out_angle[j] - shape_out[i],
                self.in_angle[j] - shape_in[i],
            ):
                x, y = _rot(vx, vy, theta)
                yield theta, rx - x, ry - y
        convex = [i for i in range(n) if shape.interior_angle(i) < 180.0]
        for j in sorted(range(len(self.pts)), key=lambda j: -self.edge_len[j]):
            rx, ry = self.pts[j]
            ex, ey = _rot(1.0, 0.0, self.out_angle[j])
            for i in convex:
                theta = self.out_angle[j] - shape_out[i]
                x, y = _rot(shape.vertices[i][0], shape.vertices[i][1], theta)
                free = self.edge_len[j] - shape.lengths[i]
                for frac in (0.5, 0.25, 0.75, 1.0):
                    d = free * frac
                    if abs(d) > 1e-6:
                        yield theta, rx + ex * d - x, ry + ey * d - y

    def _sweep(self, shape, step45):
        """Last resort: centre the shape on interior points at room-aligned
        angles."""
        period = 45.0 if step45 else 90.0
        angles = []
        for base in sorted({round(a % period, 6) for a in self.out_angle}):
            k = 0.0
            while k < 360.0:
                angles.append(base + k)
                k += period
        angles += [float(a) for a in range(0, 360, 10)]

        n = len(shape.vertices)
        cx = sum(v[0] for v in shape.vertices) / n
        cy = sum(v[1] for v in shape.vertices) / n
        minx, miny, maxx, maxy = self.bounds
        rp = self.room.polygon.representative_point()
        points = [(rp.x, rp.y)]
        steps = 7
        for a in range(steps):
            for b in range(steps):
                x = minx + (maxx - minx) * (a + 0.5) / steps
                y = miny + (maxy - miny) * (b + 0.5) / steps
                if self.point_ok(x, y):
                    points.append((x, y))
        for px, py in points:
            for theta in angles:
                x, y = _rot(cx, cy, theta)
                yield theta, px - x, py - y

    def angle_set(self, shape):
        """Rotations worth an exact test: a side of the shape parallel to a
        wall of the room (longest walls first), kept only when the shape's
        extent fits the room's extent along the frame's four directions."""
        out = []
        turns = {90: 2, 180: 4, 360: 8}[shape.symmetry()]
        for base in self.wall_bases:
            for k in range(turns):
                theta = (base + 45.0 * k) % 360.0
                pts = shape.world_vertices(theta, 0.0, 0.0)
                if self._extent_ok(pts, theta):
                    out.append((theta, pts))
                    if len(out) >= EXACT_MAX_ANGLES:
                        return out
        return out

    def _extent_ok(self, pts, theta):
        for k in range(4):
            phi = (theta + 45.0 * k) % 180.0
            ux, uy = math.cos(math.radians(phi)), math.sin(math.radians(phi))
            proj = [x * ux + y * uy for x, y in pts]
            if max(proj) - min(proj) > self._room_extent(phi) + 1e-6:
                return False
        return True

    def _room_extent(self, phi):
        key = round(phi, 3)
        got = self._extents.get(key)
        if got is None:
            ux, uy = math.cos(math.radians(phi)), math.sin(math.radians(phi))
            proj = [x * ux + y * uy for x, y in self.pts]
            got = max(proj) - min(proj)
            self._extents[key] = got
        return got

    def feasible(self, pts, convex):
        """Every translation of the shape's first vertex that puts the shape
        (vertices `pts`, first at the origin) inside the room: the room shifted
        by a point of the shape, minus the translations at which the shape's
        boundary meets the room's boundary. For a convex shape that obstacle is
        one convex hull per room edge; in general one parallelogram per pair of
        edges. Exact for simple polygons."""
        sp = np.asarray(pts, dtype=float)
        if convex:
            # hull of (edge endpoints) minus (every shape vertex)
            quads = [
                shapely.MultiPoint(np.concatenate([a - sp, b - sp])).convex_hull
                for a, b in zip(self._edge_a, self._edge_b)
            ]
            obstacle = shapely.union_all(quads)
        else:
            s0, s1 = sp[:, None, :], np.roll(sp, -1, axis=0)[:, None, :]
            a, b = self._edge_a[None, :, :], self._edge_b[None, :, :]
            quads = np.stack([a - s0, a - s1, b - s1, b - s0], axis=2).reshape(-1, 4, 2)
            ab, es = b - a, s1 - s0
            cross = (ab[..., 0] * es[..., 1] - ab[..., 1] * es[..., 0]).reshape(-1)
            quads = quads[np.abs(cross) > 1e-9]  # parallel edges: a measure-zero set
            obstacle = (
                shapely.union_all(shapely.polygons(quads)) if len(quads) else None
            )
        c = Polygon(pts).representative_point()
        inside = Polygon(self._pts_arr - (c.x, c.y))
        return inside if obstacle is None else inside.difference(obstacle)

    def exact_place(self, shape, theta, pts):
        """A pose at this rotation if the shape fits anywhere at it, found
        exactly; a zero-slack fit is allowed the validator's tolerance."""
        convex = all(t in (1, 2) for t in shape.turns)
        # charge the real work: one hull per room edge for a convex shape, one
        # parallelogram per pair of edges otherwise
        self.checks += (
            EXACT_COST * len(self.edges) * (1 if convex else max(1, len(pts) // 4))
        )
        try:
            region = self.feasible(pts, convex)
            if region.is_empty:
                c = Polygon(pts).representative_point()
                r = max(math.hypot(x - c.x, y - c.y) for x, y in pts) or 1.0
                f = 1.0 - 0.9 * TOL / r
                shrunk = [(c.x + (x - c.x) * f, c.y + (y - c.y) * f) for x, y in pts]
                region = self.feasible(shrunk, convex)
                if region.is_empty:
                    return None
            t = region.representative_point()
        except Exception:  # noqa: BLE001 - a geometry error on a pathological room is "no fit here"
            return None
        # prefer a pose on the room's integer lattice in this rotation (flush
        # with walls, which is what later bump growth needs), else the point
        vx, vy = self.pts[0]
        dx, dy = _rot(t.x - vx, t.y - vy, -theta)
        tried = set()
        for fx in (round, math.floor, math.ceil):
            for fy in (round, math.floor, math.ceil):
                sx, sy = fx(dx), fy(dy)
                if (sx, sy) in tried:
                    continue
                tried.add((sx, sy))
                wx, wy = _rot(sx, sy, theta)
                cand = (vx + wx, vy + wy)
                if region.contains(Point(cand)) and self.fits(
                    shape.world_vertices(theta, *cand)
                ):
                    return theta, cand[0], cand[1]
        if self.fits(shape.world_vertices(theta, t.x, t.y)):
            return theta, t.x, t.y
        return None

    def place(self, shape, max_checks=4000):
        """A (theta, tx, ty) that puts the whole shape inside the room, or None.
        Anchored poses first (cheap, deduplicated by the polygon they produce),
        then one exact test per candidate rotation, which finds fits with no
        slack at all; rooms with very many walls fall back to the coarse sweep."""
        exact = len(self.edges) * len(shape.vertices) <= EXACT_LIMIT
        cap = EXACT_FAST_CHECKS if exact else max_checks
        seen, polys = set(), set()
        for theta, tx, ty in self._anchored(shape):
            key = (round(theta % 360.0, 4), round(tx, 4), round(ty, 4))
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > cap or (len(seen) % 64 == 0 and self.out_of_time()):
                break
            world = shape.world_vertices(theta, tx, ty)
            pkey = tuple(sorted((round(x, 3), round(y, 3)) for x, y in world))
            if pkey in polys:
                continue
            polys.add(pkey)
            if self.fits(world):
                return theta, tx, ty
        if exact:
            for theta, pts in self.angle_set(shape):
                if self.out_of_time():
                    return None
                pos = self.exact_place(shape, theta, pts)
                if pos is not None:
                    return pos
            return None
        for theta, tx, ty in self._sweep(shape, shape.diagonal > 0):
            key = (round(theta % 360.0, 4), round(tx, 4), round(ty, 4))
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > max_checks or (len(seen) % 64 == 0 and self.out_of_time()):
                return None
            if self.fits(shape.world_vertices(theta, tx, ty)):
                return theta, tx, ty
        return None


# --- bump refinement ----------------------------------------------------------
# Start from a valid right-angled enclosure and keep pushing out a w by h
# rectangle from one of its sides into free floor whenever that raises the
# score. Widths and depths come from lengths the inventory can build, so the
# loop still closes exactly. This is what lets us follow the arms and bowls of
# letter, spiral and U-shaped rooms.


class Refiner:
    def __init__(self, placer, lengths, weights, max_corners, out_of_time):
        self.placer = placer
        self.R = sorted(lengths)
        self.Rset = set(lengths)
        self.weights = weights
        self.max_corners = max_corners
        self.out_of_time = out_of_time

    def refine(self, shape, theta, tx, ty, max_moves=400, rng=None, top_k=1):
        """Yields (shape, theta, tx, ty) after every accepted move. Normally the
        best move is taken; with `rng` one of the `top_k` best is drawn at random
        (weighted by gain) so repeated runs grow in different orders."""
        if not shape.axis_only:
            return
        # work in the shape's own frame: rotated by -theta, origin at (0, 0)
        verts = [_rot(x, y, -theta) for x, y in shape.world_vertices(theta, tx, ty)]
        room = [_rot(x, y, -theta) for x, y in self.placer.pts]
        self.room = room
        room_edges = [(room[i], room[(i + 1) % len(room)]) for i in range(len(room))]
        # a side's candidate bumps only change when the outline changes near it,
        # so they are cached by the side's endpoints and dropped around each move
        self._moves = {}
        for _ in range(max_moves):
            if self.out_of_time():
                return
            best = self._best_move(verts, room_edges, rng, top_k)
            if best is None:
                return
            _, i, o, w, h, post = best
            self._forget(verts, i, o, w, h)
            verts = self._apply(verts, i, o, w, h, post)
            got = self._to_shape(verts, theta)
            if got is None:
                return
            yield got

    @staticmethod
    def _side_key(p, q):
        return (round(p[0], 6), round(p[1], 6), round(q[0], 6), round(q[1], 6))

    def _best_move(self, verts, room_edges, rng, top_k):
        """The best bump as (gain, side, offset, width, depth, post), or None."""
        n = len(verts)
        greedy = rng is None or top_k <= 1
        best = None
        cands = []
        for i in range(n):
            if i % 16 == 0 and self.out_of_time():
                return None
            key = self._side_key(verts[i], verts[(i + 1) % n])
            moves = self._moves.get(key)
            if moves is None:
                moves = self._side_moves(verts, i, room_edges)
                self._moves[key] = moves
            for gain, o, w, h, post, corners in moves:
                if n + corners > self.max_corners:
                    continue
                if greedy:
                    if best is None or gain > best[0]:
                        best = (gain, i, o, w, h, post)
                else:
                    cands.append((gain, i, o, w, h, post))
        if not greedy and cands:
            cands.sort(key=lambda c: -c[0])
            top = cands[:top_k]
            best = rng.choices(top, weights=[c[0] for c in top])[0]
        if best is None or best[0] <= 1e-9:
            return None
        return best

    def _side_moves(self, verts, i, room_edges):
        """Every profitable bump over side i: (gain, offset, width, depth,
        post, corners added)."""
        n = len(verts)
        gain_w, gain_p = self.weights.A, self.weights.C
        p, q = verts[i], verts[(i + 1) % n]
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        li = round(length)
        ux, uy = (q[0] - p[0]) / length, (q[1] - p[1]) / length
        nx, ny = uy, -ux  # outward (right of travel) for a CCW loop
        prev, nxt = verts[i - 1], verts[(i + 2) % n]
        # a convex corner lets the bump start flush with it (offset 0)
        p_convex = (p[0] - prev[0]) * nx + (p[1] - prev[1]) * ny > 1e-9
        q_convex = (nxt[0] - q[0]) * nx + (nxt[1] - q[1]) * ny < -1e-9
        prev_len = round(math.hypot(p[0] - prev[0], p[1] - prev[1]))
        next_len = round(math.hypot(nxt[0] - q[0], nxt[1] - q[1]))

        # everything near this side, in its frame (u along, v outwards)
        room_obs = self._near(room_edges, p, ux, uy, nx, ny, li)
        # the two neighbouring sides end exactly at this side's ends, so they
        # only matter for a bump flush with them, which is handled by _depth
        skip = {i, (i - 1) % n, (i + 1) % n}
        own = [(verts[k], verts[(k + 1) % n]) for k in range(n) if k not in skip]
        poly_obs = self._near(own, p, ux, uy, nx, ny, li)

        out = []
        offsets = [0] + [o for o in self.R if o <= li - 5]
        for o in offsets:
            for w in self.R:
                post = li - o - w
                if post < 0:
                    break
                if post != 0 and post not in self.Rset:
                    continue
                # new perimeter per unit of depth: a flush end at a reflex corner
                # fills a notch and adds nothing there
                units = 2
                if o == 0 and not p_convex:
                    units -= 1
                if post == 0 and not q_convex:
                    units -= 1
                if gain_w * w + units * gain_p <= 0:
                    continue
                cap = self._clearance(room_obs, poly_obs, o, o + w)
                if cap < 5:
                    continue
                # the floor just outside this side must be room
                um = o + w / 2
                if not _inside(
                    self.room,
                    p[0] + ux * um + nx * 1e-3,
                    p[1] + uy * um + ny * 1e-3,
                ):
                    continue
                h = self._depth(cap, o, post, prev_len, next_len, p_convex, q_convex)
                if h is None:
                    continue
                corners = (2 if o > 0 else 0) + (2 if post > 0 else 0)
                gain = gain_w * w * h + units * gain_p * h
                if gain > 1e-9:
                    out.append((gain, o, w, h, post, corners))
        return out

    def _forget(self, verts, i, o, w, h):
        """Drop the cached moves of every side the bump over side i could
        affect: its neighbours, and anything within the bump horizon."""
        n = len(verts)
        for k in (i - 1, i, i + 1):
            self._moves.pop(self._side_key(verts[k % n], verts[(k + 1) % n]), None)
        p, q = verts[i], verts[(i + 1) % n]
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        ux, uy = (q[0] - p[0]) / length, (q[1] - p[1]) / length
        nx, ny = uy, -ux
        xs, ys = [], []
        for du, dv in ((o, 0), (o + w, 0), (o, h), (o + w, h)):
            xs.append(p[0] + ux * du + nx * dv)
            ys.append(p[1] + uy * du + ny * dv)
        pad = 31.5
        x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
        for key in list(self._moves):
            kx0, kx1 = min(key[0], key[2]), max(key[0], key[2])
            ky0, ky1 = min(key[1], key[3]), max(key[1], key[3])
            if kx1 >= x0 and kx0 <= x1 and ky1 >= y0 and ky0 <= y1:
                del self._moves[key]

    @staticmethod
    def _near(edges, p, ux, uy, nx, ny, li):
        """Edges that could get in the way of a bump over this side, as
        (au, av, bu, bv) in the side's frame."""
        out = []
        for a, b in edges:
            au = (a[0] - p[0]) * ux + (a[1] - p[1]) * uy
            av = (a[0] - p[0]) * nx + (a[1] - p[1]) * ny
            bu = (b[0] - p[0]) * ux + (b[1] - p[1]) * uy
            bv = (b[0] - p[0]) * nx + (b[1] - p[1]) * ny
            if max(av, bv) <= 1e-9 or min(av, bv) > 31:
                continue
            if max(au, bu) < -1e-9 or min(au, bu) > li + 1e-9:
                continue
            out.append((au, av, bu, bv))
        return out

    @staticmethod
    def _v_range(au, av, bu, bv, lo, hi, closed):
        """(v_min, v_max) of the part of the edge with u in [lo, hi] (or in the
        open interval), or None if no part of it is there."""
        e = 1e-9
        if abs(bu - au) < 1e-12:
            hit = (lo - e <= au <= hi + e) if closed else (lo + e < au < hi - e)
            return (min(av, bv), max(av, bv)) if hit else None
        t0, t1 = (lo - au) / (bu - au), (hi - au) / (bu - au)
        tlo, thi = max(0.0, min(t0, t1)), min(1.0, max(t0, t1))
        if (tlo > thi + 1e-12) if closed else (thi - tlo <= 1e-12):
            return None
        v1 = av + (bv - av) * tlo
        v2 = av + (bv - av) * thi
        return min(v1, v2), max(v1, v2)

    def _clearance(self, room_obs, poly_obs, u0, u1):
        """How deep a bump over [u0, u1] can go: it may touch the room's walls
        but has to stay clear of the rest of the enclosure."""
        cap = 30.0
        for au, av, bu, bv in room_obs:
            r = self._v_range(au, av, bu, bv, u0, u1, closed=False)
            if r is None or r[1] <= 1e-9:
                continue
            cap = min(cap, max(r[0], 0.0))
            if cap <= 0:
                return 0.0
        for au, av, bu, bv in poly_obs:
            r = self._v_range(au, av, bu, bv, u0, u1, closed=True)
            if r is None or r[1] <= 1e-7:
                continue
            cap = min(cap, max(r[0], 0.0) - 1e-6)
            if cap <= 0:
                return 0.0
        return cap

    def _depth(self, cap, o, post, prev_len, next_len, p_convex, q_convex):
        """Deepest bump we can build. If it is flush with a neighbouring side,
        that side grows (convex corner) or shrinks (reflex corner: the bump
        fills the notch) and must stay a buildable length."""
        for h in reversed(self.R):
            if h > cap + 1e-9:
                continue
            if o == 0 and (prev_len + h if p_convex else prev_len - h) not in self.Rset:
                continue
            if (
                post == 0
                and (next_len + h if q_convex else next_len - h) not in self.Rset
            ):
                continue
            return h
        return None

    @staticmethod
    def _apply(verts, i, o, w, h, post):
        n = len(verts)
        p, q = verts[i], verts[(i + 1) % n]
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        ux, uy = (q[0] - p[0]) / length, (q[1] - p[1]) / length
        nx, ny = uy, -ux
        a = (p[0] + ux * o, p[1] + uy * o)
        b = (a[0] + nx * h, a[1] + ny * h)
        c = (b[0] + ux * w, b[1] + uy * w)
        d = (c[0] - nx * h, c[1] - ny * h)
        new = verts[: i + 1]
        if o > 0:
            new.append(a)
        new += [b, c]
        if post > 0:
            new.append(d)
        new += verts[i + 1 :]
        # drop vertices that became straight
        out = []
        m = len(new)
        for k in range(m):
            x0, y0 = new[k - 1]
            x1, y1 = new[k]
            x2, y2 = new[(k + 1) % m]
            if abs((x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)) > 1e-6:
                out.append(new[k])
        return out

    @staticmethod
    def _to_shape(verts, theta):
        n = len(verts)
        dirs, lengths = [], []
        for k in range(n):
            (x0, y0), (x1, y1) = verts[k], verts[(k + 1) % n]
            length = math.hypot(x1 - x0, y1 - y0)
            if abs(length - round(length)) > 1e-6:
                return None
            ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 360.0
            if abs(ang - 45.0 * round(ang / 45.0)) > 1e-4:
                return None
            dirs.append(round(ang / 45.0) % 8)
            lengths.append(round(length))
        shape = Shape("bumped", (n,), dirs, lengths)
        if not shape.valid():
            return None
        tx, ty = _rot(verts[0][0], verts[0][1], theta)
        return shape, theta, tx, ty


# --- grid outlines ------------------------------------------------------------
# A face can't be longer than 30, so a plain rectangle tops out at 30 x 30. In a
# big room with lots of pieces a many-cornered outline that follows the room does
# much better. We lay a grid over the room, grow a blob of cells outward from the
# roomiest spot (never letting it get a hole or pinch), break every straight run
# longer than 30 with a one-cell bump or notch, and turn the outline into sides.
# Every side is a whole number of cells and every corner is 90 degrees, so the
# loop closes exactly.

N4 = [(1, 0), (0, 1), (-1, 0), (0, -1)]
# the eight neighbours of a cell, in order: E, NE, N, NW, W, SW, S, SE
RING = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


class Region:
    """A set of grid cells kept 4-connected, hole-free and free of pinches."""

    def __init__(self, inside):
        self.inside = inside
        self.cells = set()

    def _ring(self, c):
        x, y = c
        return [(x + dx, y + dy) in self.cells for dx, dy in RING]

    def can_add(self, c):
        if c in self.cells or c not in self.inside:
            return False
        if not self.cells:
            return True
        r = self._ring(c)
        if not any(r[k] for k in (0, 2, 4, 6)):
            return False
        # a diagonal neighbour that only touches c at a corner is a pinch
        for k in (1, 3, 5, 7):
            if r[k] and not r[k - 1] and not r[(k + 1) % 8]:
                return False
        return self._connected(r)

    def can_remove(self, c):
        if c not in self.cells or len(self.cells) <= 1:
            return False
        r = self._ring(c)
        if all(r[k] for k in (0, 2, 4, 6)):
            return False  # an inner cell, removing it leaves a hole
        for k in (1, 3, 5, 7):
            if not r[k] and r[k - 1] and r[(k + 1) % 8]:
                return False
        return self._connected(r)

    @staticmethod
    def _connected(r):
        """Are the neighbours of c that are in the region joined to each other
        around c, through the corner cells?"""
        edges = [k for k in (0, 2, 4, 6) if r[k]]
        parent = {k: k for k in edges}

        def find(a):
            while parent[a] != a:
                a = parent[a]
            return a

        for k in edges:
            nxt = (k + 2) % 8
            if r[nxt] and r[(k + 1) % 8]:
                parent[find(k)] = find(nxt)
        return len({find(k) for k in edges}) <= 1

    def outline(self):
        """Boundary edges counter-clockwise as (dir, start, inner, outer), dir in
        45 degree units (0 E, 2 N, 4 W, 6 S). None unless it is a single loop."""
        nxt = {}
        for x, y in self.cells:
            if (x, y - 1) not in self.cells:
                nxt[(x, y)] = (0, (x + 1, y), (x, y), (x, y - 1))
            if (x + 1, y) not in self.cells:
                nxt[(x + 1, y)] = (2, (x + 1, y + 1), (x, y), (x + 1, y))
            if (x, y + 1) not in self.cells:
                nxt[(x + 1, y + 1)] = (4, (x, y + 1), (x, y), (x, y + 1))
            if (x - 1, y) not in self.cells:
                nxt[(x, y + 1)] = (6, (x, y), (x, y), (x - 1, y))
        if not nxt:
            return None
        start = v = min(nxt)
        edges = []
        while True:
            d, w, inner, outer = nxt[v]
            edges.append((d, v, inner, outer))
            v = w
            if v == start:
                break
            if len(edges) > len(nxt):
                return None
        # else there is a hole
        return edges if len(edges) == len(nxt) else None


def _runs(edges):
    """Group an outline into straight runs (lists of edges)."""
    n = len(edges)
    k = 0
    while k < n and edges[k][0] == edges[k - 1][0]:
        k += 1
    if k == n:
        return None
    edges = edges[k:] + edges[:k]
    out = [[edges[0]]]
    for e in edges[1:]:
        if e[0] == out[-1][-1][0]:
            out[-1].append(e)
        else:
            out.append([e])
    return out


def _clip(pts, box):
    """Sutherland-Hodgman: the polygon cut down to an axis-aligned box."""
    x0, y0, x1, y1 = box

    def clip_edge(poly, inside, cross):
        out = []
        for i in range(len(poly)):
            a, b = poly[i - 1], poly[i]
            ia, ib = inside(a), inside(b)
            if ib:
                if not ia:
                    out.append(cross(a, b))
                out.append(b)
            elif ia:
                out.append(cross(a, b))
        return out

    def at_x(xv):
        return lambda a, b: (xv, a[1] + (b[1] - a[1]) * (xv - a[0]) / (b[0] - a[0]))

    def at_y(yv):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (yv - a[1]) / (b[1] - a[1]), yv)

    poly = list(pts)
    for inside, cross in (
        (lambda p: p[0] >= x0, at_x(x0)),
        (lambda p: p[0] <= x1, at_x(x1)),
        (lambda p: p[1] >= y0, at_y(y0)),
        (lambda p: p[1] <= y1, at_y(y1)),
    ):
        if not poly:
            break
        poly = clip_edge(poly, inside, cross)
    return poly


def _crossings(pts, y):
    """Intervals of x where the horizontal line at height y is inside."""
    xs = []
    for i in range(len(pts)):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % len(pts)]
        if (ay > y) != (by > y):
            xs.append(ax + (y - ay) * (bx - ax) / (by - ay))
    xs.sort()
    return list(zip(xs[::2], xs[1::2]))


def _overlap(a, b):
    """Intersection of two sorted lists of intervals."""
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


class Grid:
    """Cut positions along x and y (in the rotated frame) and the cells that lie
    completely inside the room."""

    def __init__(self, theta, xc, yc, inside):
        self.theta = theta
        self.xc = xc
        self.yc = yc
        self.w = [xc[i + 1] - xc[i] for i in range(len(xc) - 1)]
        self.h = [yc[j + 1] - yc[j] for j in range(len(yc) - 1)]
        self.inside = inside

    def world(self, gx, gy):
        return _rot(gx, gy, self.theta)

    def centre(self, c):
        return (self.xc[c[0]] + self.xc[c[0] + 1]) / 2, (
            self.yc[c[1]] + self.yc[c[1] + 1]
        ) / 2

    def corners(self, c):
        i, j = c
        return [
            self.world(self.xc[i + a], self.yc[j + b])
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))
        ]

    def edge_len(self, d, cell):
        return self.w[cell[0]] if d in (0, 4) else self.h[cell[1]]

    def area(self, cells):
        return sum(self.w[x] * self.h[y] for x, y in cells)


class GridEngine:
    def __init__(self, placer, allocator, connectors, weights, out_of_time):
        self.placer = placer
        self.alloc = allocator
        self.right = connectors["right"]
        self.weights = weights
        self.out_of_time = out_of_time
        self.total_length = sum(n * k for n, k in allocator.pool.counts.items())
        self.side_lengths = [
            n for n in range(5, 31) if allocator.side_ok(n, allocator.pool)
        ]

    # grids to try

    def frame_angles(self, limit=3):
        """The room's dominant wall directions (mod 90), longest walls first."""
        weight = {}
        for ang, length in zip(self.placer.out_angle, self.placer.edge_len):
            a = round(ang % 90.0, 6)
            if a > 90.0 - 1e-6:
                a = 0.0
            weight[a] = weight.get(a, 0.0) + length
        return [a for a, _ in sorted(weight.items(), key=lambda t: -t[1])[:limit]]

    def cell_sizes(self, limit=3):
        sizes = []
        for s in range(5, 16):
            mults = [k * s for k in range(1, 30 // s + 1)]
            pool = self.alloc.pool
            if all(self.alloc.side_ok(m, pool) for m in mults) and any(
                self.alloc.gate_options(m, pool) for m in mults
            ):
                sizes.append(s)
                if len(sizes) >= limit:
                    break
        return sizes

    @staticmethod
    def _uniform_cuts(lo, hi, s, off):
        cuts = [off + s * math.floor((lo - off) / s + 1e-9)]
        while cuts[-1] < hi - 1e-9:
            cuts.append(cuts[-1] + s)
        return cuts

    def _slab_cuts(self, pts, gmax, rank=0, free_start=False):
        """Cut positions along y (gaps are buildable side lengths <= gmax),
        picked by dynamic programming to maximise the room area that sits in
        full-height columns of each slab, so slabs line up with the room's bars
        and arms. `rank` picks which vertex fraction the cuts sit on (0 = the
        most common one); with `free_start` the first cut may sit anywhere
        within one gap of the room's bottom instead of being pinned to it."""
        gaps = [g for g in self.side_lengths if g <= gmax]
        if not gaps:
            return None
        ys = [p[1] for p in pts]
        lo, hi = min(ys), max(ys)
        weight, exact = {}, {}
        for y in ys:
            r = y - math.floor(y)
            f = round(r, 6) % 1.0
            weight[f] = weight.get(f, 0.0) + 1
            exact.setdefault(f, r % 1.0)
        fractions = [f for f, _ in sorted(weight.items(), key=lambda t: -t[1])]
        if rank >= len(fractions):
            return None
        base = math.floor(lo) + exact[fractions[rank]]
        if base > lo + 1e-9:
            base -= 1
        span = math.ceil(hi - base - 1e-9)
        if span <= 0 or span > 2000:
            return None
        vert_y = sorted(set(ys))
        eps = 1e-7
        cache = {}

        def value(a, b):
            """Room area in the columns that are inside for the whole slab."""
            if (a, b) not in cache:
                ya, yb = base + a, base + b
                iv = _overlap(_crossings(pts, ya + eps), _crossings(pts, yb - eps))
                k = bisect.bisect_right(vert_y, ya)
                while iv and k < len(vert_y) and vert_y[k] < yb:
                    iv = _overlap(iv, _crossings(pts, vert_y[k] - eps))
                    iv = _overlap(iv, _crossings(pts, vert_y[k] + eps))
                    k += 1
                cache[a, b] = (yb - ya) * sum(end - start for start, end in iv)
            return cache[a, b]

        top = span + max(gaps)
        dp = [-1.0] * (top + 1)
        back = [-1] * (top + 1)
        dp[0] = 0.0
        if free_start:
            for t in range(1, min(max(gaps), top + 1)):
                dp[t] = 0.0
        for t in range(1, top + 1):
            if t % 64 == 0 and self.out_of_time():
                return None
            for g in gaps:
                if t - g >= 0 and dp[t - g] >= 0:
                    val = dp[t - g] + (value(t - g, t) if t - g < span else 0.0)
                    if val > dp[t] + 1e-9:
                        dp[t] = val
                        back[t] = t - g
        ends = [t for t in range(span, top + 1) if dp[t] >= 0]
        if not ends:
            return None
        t = max(ends, key=lambda e: (dp[e], -e))
        cuts = []
        while t >= 0:
            cuts.append(base + t)
            t = back[t] if t > 0 else -1
        return cuts[::-1]

    def grids(self):
        """(theta, xcuts, ycuts): room-aligned adaptive cuts first, then
        windows of big rooms, then uniform cells."""
        sizes = self.cell_sizes()
        for theta in self.frame_angles():
            local = [_rot(x, y, -theta) for x, y in self.placer.pts]
            xs = [p[0] for p in local]
            ys = [p[1] for p in local]
            swapped = [(y, x) for x, y in local]
            seen = set()
            for gmax in (10, 14):
                for rank, free in ((0, False), (0, True), (1, False), (1, True)):
                    yc = self._slab_cuts(local, gmax, rank, free)
                    xc = self._slab_cuts(swapped, gmax, rank, free)
                    if xc and yc:
                        key = (
                            tuple(round(c, 6) for c in xc),
                            tuple(round(c, 6) for c in yc),
                        )
                        if key not in seen:
                            seen.add(key)
                            yield theta, xc, yc
            yield from self._window_grids(theta, local)
            for s in sizes:
                for ox, oy in self._offsets(local, s):
                    yield (
                        theta,
                        self._uniform_cuts(min(xs), max(xs), s, ox),
                        self._uniform_cuts(min(ys), max(ys), s, oy),
                    )

    def _window_grids(self, theta, local, size=56.0, step=28.0):
        """Big rooms: grids fitted to one window at a time, so the cut lines can
        follow that part's arms instead of a compromise for the whole room."""
        xs = [p[0] for p in local]
        ys = [p[1] for p in local]
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        if maxx - minx <= size + 1e-6 and maxy - miny <= size + 1e-6:
            return
        windows = []
        wx = minx
        while True:
            wy = miny
            while True:
                box = (wx, wy, min(wx + size, maxx), min(wy + size, maxy))
                clipped = _clip(local, box)
                if len(clipped) >= 3 and abs(_area(clipped)) > 50:
                    windows.append((abs(_area(clipped)), clipped))
                if wy + size >= maxy - 1e-6:
                    break
                wy += step
            if wx + size >= maxx - 1e-6:
                break
            wx += step
        windows.sort(key=lambda t: -t[0])
        for _, clipped in windows[:10]:
            for free in (False, True):
                yc = self._slab_cuts(clipped, 10, 0, free)
                xc = self._slab_cuts([(y, x) for x, y in clipped], 10, 0, free)
                if xc and yc:
                    yield theta, xc, yc

    @staticmethod
    def _offsets(local, s):
        def common(values):
            count, exact = {}, {}
            for v in values:
                r = v % s
                if r > s - 1e-6:
                    r = 0.0
                key = round(r, 6)  # rounded only to count; the exact residue is kept
                count[key] = count.get(key, 0) + 1
                exact.setdefault(key, r)
            return [exact[o] for o, _ in sorted(count.items(), key=lambda t: -t[1])[:2]]

        xo = common([p[0] for p in local])
        yo = common([p[1] for p in local])
        return [(x, y) for x in xo for y in yo][:3]

    def rasterize(self, theta, xc, yc):
        """Which cells are completely inside the room (cells the outline runs
        through are blocked)."""
        nx, ny = len(xc) - 1, len(yc) - 1
        if nx <= 0 or ny <= 0 or nx * ny > 60000:
            return None
        local = [_rot(x, y, -theta) for x, y in self.placer.pts]
        n = len(local)
        eps = 9e-5  # just under the validator's tolerance: walls 1e-4 off a cut still count as on it
        step = (
            min(
                min(xc[i + 1] - xc[i] for i in range(nx)),
                min(yc[j + 1] - yc[j] for j in range(ny)),
            )
            / 24.0
        )

        def locate(cuts, v):
            k = bisect.bisect_right(cuts, v) - 1
            on_cut = (0 <= k < len(cuts) and abs(v - cuts[k]) < eps) or (
                k + 1 < len(cuts) and abs(cuts[k + 1] - v) < eps
            )
            return k, on_cut

        blocked = set()
        for i in range(n):
            ax, ay = local[i]
            bx, by = local[(i + 1) % n]
            steps = max(1, int(math.hypot(bx - ax, by - ay) / step) + 1)
            for k in range(steps + 1):
                t = k / steps
                ci, on_u = locate(xc, ax + (bx - ax) * t)
                cj, on_v = locate(yc, ay + (by - ay) * t)
                if not on_u and not on_v:
                    blocked.add((ci, cj))

        inside = set()
        for j in range(ny):
            ym = (yc[j] + yc[j + 1]) / 2
            for a, b in _crossings(local, ym):
                for i in range(nx):
                    if a < (xc[i] + xc[i + 1]) / 2 < b and (i, j) not in blocked:
                        inside.add((i, j))
        return Grid(theta, xc, yc, inside)

    # growing a blob

    @staticmethod
    def seeds(grid, count=2):
        """Roomiest cells (furthest from the edge), kept apart from each other."""
        inside = grid.inside
        frontier = [
            c
            for c in inside
            if any((c[0] + dx, c[1] + dy) not in inside for dx, dy in RING)
        ]
        dist = dict.fromkeys(frontier, 1)
        for c in frontier:  # the list grows while we walk it
            for dx, dy in RING:
                d = (c[0] + dx, c[1] + dy)
                if d in inside and d not in dist:
                    dist[d] = dist[c] + 1
                    frontier.append(d)

        ranked = sorted(
            inside, key=lambda c: (-dist.get(c, 0), -grid.w[c[0]] * grid.h[c[1]], c)
        )
        out = []
        for c in ranked:
            if all(
                max(abs(c[0] - o[0]), abs(c[1] - o[1])) > 2 * dist.get(o, 1)
                for o in out
            ):
                out.append(c)
            if len(out) >= count:
                break
        return out

    @staticmethod
    def _corner_delta(cells, c):
        """Corners of the blob at the four vertices of cell c."""
        x, y = c
        total = 0
        for vx, vy in ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)):
            k = (
                ((vx - 1, vy - 1) in cells)
                + ((vx, vy - 1) in cells)
                + ((vx - 1, vy) in cells)
                + ((vx, vy) in cells)
            )
            total += k in (1, 3)
        return total

    def grow(self, grid, seed, max_corners):
        """Add cells nearest the seed first. Returns the order they were added
        in and (count, area, perimeter, corners) after each one."""
        region = Region(grid.inside)
        cells = region.cells
        sx, sy = grid.centre(seed)

        def dist(c):
            cx, cy = grid.centre(c)
            return max(abs(cx - sx), abs(cy - sy))

        order, stats = [], []
        corners = 0
        area = perim = 0.0
        heap = [(0.0, 0, seed)]
        queued = {seed}
        while heap:
            _, tries, c = heapq.heappop(heap)
            if c in cells:
                continue
            if not region.can_add(c):
                if tries < 3:  # may become addable once its neighbours are in
                    heapq.heappush(heap, (dist(c) + 10.0 * (tries + 1), tries + 1, c))
                continue
            i, j = c
            w, h = grid.w[i], grid.h[j]
            before = self._corner_delta(cells, c)
            for (dx, dy), length in (
                ((1, 0), h),
                ((-1, 0), h),
                ((0, 1), w),
                ((0, -1), w),
            ):
                perim += -length if (i + dx, j + dy) in cells else length
            cells.add(c)
            corners += self._corner_delta(cells, c) - before
            area += w * h
            order.append(c)
            stats.append((len(order), area, perim, corners))
            if corners > max_corners:
                break
            for dx, dy in N4:
                d = (i + dx, j + dy)
                if d in grid.inside and d not in cells and d not in queued:
                    queued.add(d)
                    heapq.heappush(heap, (dist(d), 0, d))
        return order, stats

    # finishing a blob

    def break_long_runs(self, region, grid):
        """Every straight run must be <= 30: add or remove one cell along each
        longer one. False if some run can't be fixed."""
        for _ in range(4 * len(region.cells) + 50):
            if self.out_of_time():
                return False
            edges = region.outline()
            runs = _runs(edges) if edges else None
            if runs is None:
                return False
            if len(runs) > self.right:
                return False  # already more corners than right connectors
            run = next(
                (
                    r
                    for r in runs
                    if sum(grid.edge_len(e[0], e[2]) for e in r) > 30 + 1e-6
                ),
                None,
            )
            if run is None:
                return True
            total = sum(grid.edge_len(e[0], e[2]) for e in run)
            # cheapest first: dropping a cell at either end of the run shortens
            # it (and the run around the corner) for two new corners, not four
            fixed = False
            for k in (len(run) - 1, 0):
                cell = run[k][2]
                if total - grid.edge_len(
                    run[k][0], cell
                ) <= 30 + 1e-6 and region.can_remove(cell):
                    region.cells.discard(cell)
                    fixed = True
                    break
            if fixed:
                continue
            # else break after the longest prefix that is still <= 30
            prefix = 0.0
            last_ok = 0
            for k, e in enumerate(run):
                prefix += grid.edge_len(e[0], e[2])
                if prefix > 30 + 1e-6:
                    break
                last_ok = k
            for k in range(min(last_ok + 1, len(run) - 2), 0, -1):
                _, _, inner, outer = run[k]
                if region.can_add(outer):
                    region.cells.add(outer)
                    fixed = True
                    break
                if region.can_remove(inner):
                    region.cells.discard(inner)
                    fixed = True
                    break
            if not fixed and not self._cut_off(region, grid, run):
                return False
        return False

    def _cut_off(self, region, grid, run):
        """Last resort for a run that can't be bumped or notched: drop one cell
        along it and whatever part of the blob that disconnects, choosing the
        cut that loses the least area."""
        best = None
        for edge in run:
            rest = set(region.cells)
            rest.discard(edge[2])
            if not rest:
                continue
            keep = max(_components(rest), key=grid.area)
            trial = Region(region.inside)
            trial.cells = set(keep)
            if trial.outline() is None or _has_pinch(keep):
                continue
            lost = grid.area(region.cells - keep)
            if best is None or lost < best[0]:
                best = (lost, keep)
        if best is None:
            return False
        region.cells = set(best[1])
        return True

    def to_shape(self, region, grid):
        edges = region.outline()
        runs = _runs(edges) if edges else None
        if runs is None:
            return None
        lengths = []
        for r in runs:
            total = sum(grid.edge_len(e[0], e[2]) for e in r)
            if abs(total - round(total)) > 1e-6:
                return None
            lengths.append(round(total))
        if any(n < 5 or n > 30 for n in lengths):
            return None
        shape = Shape("grid", (len(region.cells),), [r[0][0] for r in runs], lengths)
        if not shape.valid():
            return None
        vx, vy = runs[0][0][1]
        tx, ty = grid.world(grid.xc[vx], grid.yc[vy])
        return shape, tx, ty

    def run(self, try_shape, best_score):
        """Grow blobs on every grid and hand the shapes to try_shape(shape,
        theta, tx, ty) -> (score, ok, reason)."""
        w = self.weights
        best = best_score
        if self.right < 8:
            return  # with fewer corners a grid outline is a rectangle or an L, which the templates cover
        for n_grid, (theta, xc, yc) in enumerate(self.grids()):
            if n_grid >= MAX_GRIDS or self.out_of_time():
                return
            grid = self.rasterize(theta, xc, yc)
            if grid is None or not grid.inside:
                continue
            starts = self.seeds(grid)
            if 20 <= len(grid.inside) <= 1500:
                # a few more seeds spread over the room: nearest-first growth
                # from the roomiest cell misses blobs that start elsewhere
                cells = sorted(grid.inside)
                stride = max(3, len(cells) // 12)
                starts += [c for c in cells[::stride] if c not in starts]
            for seed in starts:
                if self.out_of_time():
                    return
                order, stats = self.grow(grid, seed, self.right)
                est = [
                    (1000 + w.A * area + w.C * perim, k)
                    for k, area, perim, corners in stats
                    if corners <= self.right and perim <= self.total_length
                ]
                if not est:
                    continue
                est.sort(reverse=True)
                if est[0][0] + w.G <= best:
                    continue

                # the biggest blob first, and if that doesn't work out, bisect
                # on how many cells to keep
                k_best = est[0][1]
                res = self._evaluate(order[:k_best], grid, try_shape)
                if res is not None:
                    best = max(best, res)
                    continue
                lo, hi, evals = 1, k_best - 1, 0
                while lo <= hi and evals < 12 and not self.out_of_time():
                    mid = (lo + hi + 1) // 2
                    evals += 1
                    res = self._evaluate(order[:mid], grid, try_shape)
                    if res is None:
                        hi = mid - 1
                    else:
                        best = max(best, res)
                        lo = mid + 1

    def _evaluate(self, cells, grid, try_shape):
        inside = set(grid.inside)
        region = Region(inside)
        region.cells.update(cells)
        for _ in range(4):
            if not self.break_long_runs(region, grid):
                return None
            got = self.to_shape(region, grid)
            if got is None:
                return None
            shape, tx, ty = got
            if shape.right > self.right:
                return None
            score, ok, reason = try_shape(shape, grid.theta, tx, ty)
            if ok:
                return score
            if "fit" not in (reason or ""):
                return None
            # the raster was a bit optimistic: drop the cells that poke out
            bad = [c for c in region.cells if not self.placer.fits(grid.corners(c))]
            if not bad:
                return None
            for c in bad:
                inside.discard(c)
                if not region.can_remove(c):
                    return None
                region.cells.discard(c)
        return None


def _components(cells):
    left = set(cells)
    out = []
    while left:
        stack = [left.pop()]
        comp = set(stack)
        while stack:
            x, y = stack.pop()
            for dx, dy in N4:
                d = (x + dx, y + dy)
                if d in left:
                    left.discard(d)
                    comp.add(d)
                    stack.append(d)
        out.append(comp)
    return out


def _has_pinch(cells):
    """Two cells that touch only at a corner."""
    for x, y in cells:
        if (
            (x + 1, y + 1) in cells
            and (x + 1, y) not in cells
            and (x, y + 1) not in cells
        ):
            return True
        if (
            (x - 1, y + 1) in cells
            and (x - 1, y) not in cells
            and (x, y + 1) not in cells
        ):
            return True
    return False


# --- the player ---------------------------------------------------------------


class Player8(Player):
    """Score-driven search over many enclosure shapes: grid outlines first (they
    are cheap and finish on their own), then the templates, growing every
    right-angled shape found by bumps, and finally random restarts. Each phase
    is bounded by work rather than by the clock, so the answer does not depend
    on the machine. Returns the best valid enclosure, or None."""

    def build_enclosure(self):
        self.start = time.process_time()
        self.deadline = TIME_BUDGET
        inv = self.inventory
        self.conn = {
            "right": inv.connectors.get(ConnectorType.RIGHT, 0),
            "diagonal": inv.connectors.get(ConnectorType.DIAGONAL, 0),
            "straight": inv.connectors.get(ConnectorType.STRAIGHT, 0),
        }
        self.alloc = Allocator(inv.walls, inv.gates, self.conn["straight"])
        if not self.alloc.gates:
            self.note = "No gate of length >= 5 in the inventory, nothing valid exists"
            return None
        self.placer = Placer(self.room, self._done)
        self.best_score = 0.0
        self.best = None
        self.best_desc = ""
        two_gates = self.weights.G > 0 and sum(self.alloc.gates.values()) >= 2
        self.gate_counts = [2, 1] if two_gates else [1]
        self.alloc.borrow_for_single = (
            True  # 2-gate builds are tried first and compared on real score
        )
        self.seeds = []  # right-angled enclosures to grow from, best first
        self._refined = set()  # seeds that have already been grown
        self._posed = set()  # (shape, pose) pairs already grown
        self._cands = None  # template candidates, worked through in order
        self._rect_fit_memo = {}  # (a, b) -> whether an a x b rectangle fits the room
        self._next = 0
        self._restarts = 0
        self.upper_bound = self._upper_bound()

        try:
            # grid outlines first: cheap, they finish on their own, and they
            # are the best seeds for bump growth, so nothing may outscore them
            # before they are evaluated (the grid skips blobs it cannot beat)
            engine = GridEngine(
                self.placer, self.alloc, self.conn, self.weights, self._done
            )
            engine.run(self._try_grid_shape, self.best_score)
            self._room_trace()
            self._refine()
            self._templates(TEMPLATE_CHECKS)
            self._refine()
            # whatever templates the first pass did not get to
            self._templates(TEMPLATE_CHECKS)
            self._refine()
            self._explore()
        except Exception as e:  # noqa: BLE001 - keep what we have, don't crash the run
            self.note = f"search stopped early ({type(e).__name__}: {e})"

        if self.best is None:
            self.note = self.note or "No valid enclosure with a positive score found"
            return None
        self.note = (
            f"{self.best_desc}, expected score {self.best_score:.1f} "
            f"(cpu {time.process_time() - self.start:.1f}s, "
            f"{self.placer.checks} checks, {self._restarts} restarts"
            f"{', at the upper bound' if self._at_bound() else ''})"
        )
        return self.best

    def _out_of_time(self):
        return time.process_time() - self.start > self.deadline

    def _at_bound(self):
        return self.best_score >= self.upper_bound - 1e-6

    def _done(self):
        """Stop searching: the safety-net deadline has passed, or nothing can
        beat what we have."""
        return self._at_bound() or self._out_of_time()

    def _upper_bound(self):
        """No enclosure can score more than this: its area is at most the
        room's and at most what the pieces' total length can enclose
        (isoperimetric), its perimeter is charged nothing, and the second-gate
        bonus is assumed whenever two gates exist."""
        w = self.weights
        total = sum(n * k for n, k in self.alloc.pool.counts.items())
        area = min(self.room.polygon.area, total * total / (4 * math.pi))
        return 1000.0 + w.A * area + (w.G if len(self.gate_counts) > 1 else 0.0)

    def _score(self, area, perimeter, bonus):
        w = self.weights
        return 1000.0 + w.A * area + w.C * perimeter + (w.G if bonus else 0.0)

    def _connectors_ok(self, shape):
        return (
            shape.right <= self.conn["right"]
            and shape.diagonal <= self.conn["diagonal"]
        )

    def _side_lengths(self):
        pool = self.alloc.pool  # walls plus spare gates used as walls
        return [n for n in range(5, 31) if self.alloc.side_ok(n, pool)]

    def _commit(self, shape, pieces, theta, tx, ty, desc):
        """Build it, check it with the simulator's own rules, keep it if best."""
        construction = shape.construction(pieces, theta, tx, ty)
        result = validate_construction(construction, self.room, self.inventory)
        if not result.valid:
            return 0.0, False, result.reason
        score = score_construction(result, self.weights)
        if shape.axis_only and "grown" not in desc:
            self.seeds.append((score, shape, theta, tx, ty))
            self.seeds.sort(key=lambda t: -t[0])
        if score > self.best_score:
            self.best_score = score
            self.best = construction
            self.best_desc = desc
        return score, True, "valid"

    def _commit_any(self, shape, theta, tx, ty, label, skip_worse=False):
        """Allocate pieces for the shape (two gates if we can, else one) and
        commit the first that works. Returns (score, ok, reason)."""
        if not self._connectors_ok(shape):
            return 0.0, False, "connectors"
        reason = "pieces"
        for n_gates in self.gate_counts:
            if skip_worse:
                # a spare gate used as a wall on another face earns the bonus too
                bonus = n_gates == 2 or (
                    self.weights.G > 0 and sum(self.alloc.gates.values()) >= 2
                )
                est = self._score(shape.area, shape.perimeter, bonus)
                if est <= self.best_score:
                    # can't beat our best; say "ok" so the caller stops shrinking
                    return est, True, "worse"
            pieces = self.alloc.allocate(shape.lengths, n_gates)
            if pieces is None:
                continue
            desc = label + (" with 2 gates" if n_gates == 2 else "")
            score, ok, reason = self._commit(shape, pieces, theta, tx, ty, desc)
            if ok:
                return score, ok, reason
        return 0.0, False, reason

    # templates

    def _template_candidates(self):
        """(upper-bound score, family, params) for every template the connector
        stock allows, best first."""
        lengths = self._side_lengths()
        if not lengths:
            return []
        buildable = set(lengths)
        room_diag = self.placer.diameter + 1e-6  # the room's true diameter
        room_area = self.room.polygon.area + 1e-6
        right, diag = self.conn["right"], self.conn["diagonal"]
        bonus = self.weights.G if self.gate_counts[0] == 2 else 0.0
        max_r = self.placer.max_radius
        r2 = math.sqrt(0.5)
        out = []

        def add(family, params, area, perim, diameter, radius):
            if area > room_area or diameter > room_diag or radius > max_r:
                return
            bound = self._score(area, perim, False) + bonus
            if bound > self.best_score:
                out.append((bound, family, params))

        if right >= 4:
            for i, a in enumerate(lengths):
                for b in lengths[i:]:
                    add("rect", (a, b), a * b, 2 * (a + b), math.hypot(a, b), a / 2)
        if diag >= 8:
            for a in lengths:
                for b in lengths:
                    for c in lengths:
                        for d in lengths:
                            if (b, a, d, c) < (a, b, c, d):
                                continue  # the same octagon turned by 90 degrees
                            k = (c + d) * r2
                            w, h = a + k, b + k
                            area = w * h - (c * c + d * d) / 2
                            r = min(w, h, (w + h - max(c, d) * 2 * r2) * r2) / 2
                            # true diameter: the longest of the four antipodal pairs
                            sd, td = (c - d) * r2, (c + d) * r2
                            dm = math.sqrt(
                                max(
                                    (a + sd) ** 2 + (b + td) ** 2,
                                    (a - sd) ** 2 + (b + td) ** 2,
                                    (a + td) ** 2 + (b - sd) ** 2,
                                    (a + td) ** 2 + (b + sd) ** 2,
                                )
                            )
                            add(
                                "oct",
                                (a, b, c, d),
                                area,
                                2 * (a + b + c + d),
                                dm,
                                r,
                            )
        if right >= 2 and diag >= 4:
            for a in lengths:
                for b in lengths:
                    for c in lengths:
                        k = c * r2
                        w, h = a + k, b + k
                        area = w * h - c * c / 2
                        r = min(w, h, (w + h - c * 2 * r2) * r2) / 2
                        for family in (
                            "hex",
                        ):  # hex2(b, a, c) is hex(a, b, c) turned by 90 degrees
                            add(
                                family,
                                (a, b, c),
                                area,
                                2 * (a + b + c),
                                math.hypot(w, h),
                                r,
                            )
        if right >= 6:
            for a1 in lengths:
                for a2 in lengths:
                    if a1 + a2 not in buildable:
                        continue
                    for b1 in lengths:
                        for b2 in lengths:
                            if b1 + b2 not in buildable:
                                continue
                            w, h = a1 + a2, b1 + b2
                            area = w * b1 + a1 * b2
                            r = max(min(w, b1), min(a1, h)) / 2
                            dm = math.hypot(w, h)
                            add("L", (a1, a2, b1, b2), area, 2 * (w + h), dm, r)
        if right >= 8:
            for a1 in lengths:
                for a2 in lengths:
                    for a3 in lengths:
                        w = a1 + a2 + a3
                        if w not in buildable:
                            continue
                        for b1 in lengths:
                            for b2 in lengths:
                                if b1 + b2 not in buildable:
                                    continue
                                h = b1 + b2
                                dm = math.hypot(w, h)
                                p = (a1, a2, a3, b1, b2)
                                u_area = w * b1 + (a1 + a3) * b2
                                u_r = max(min(w, b1), min(max(a1, a3), h)) / 2
                                t_r = max(min(w, b1), min(a2, h)) / 2
                                t_dm = max(
                                    math.hypot(w, b1),
                                    math.hypot(a1 + a2, h),
                                    math.hypot(a2 + a3, h),
                                )
                                add("U", p, u_area, 2 * (w + h) + 2 * b2, dm, u_r)
                                add("T", p, w * b1 + a2 * b2, 2 * (w + h), t_dm, t_r)
            # Z and S: two a x b rectangles offset by (c, d); the bounding box
            # reaches twice the 30-unit face limit
            for a in lengths:
                if a < 8:
                    continue
                for c in lengths:
                    if c >= a:
                        break
                    for b in lengths:
                        if b < 8:
                            continue
                        for d in lengths:
                            if d >= b:
                                break
                            dm = math.hypot(a + c, b + d)
                            per = 2 * (a + b + c + d)
                            add(
                                "Z",
                                (a, b, c, d),
                                (a + c) * (b + d) - 2 * c * d,
                                per,
                                dm,
                                min(a, b) / 2,
                            )
                            # the snake: bars d and c thick
                            add(
                                "step",
                                (a, b, c, d),
                                b * (a + c) - d * (a - c),
                                per,
                                dm,
                                min(c, d) / 2,
                            )
        if right >= 3 and diag >= 4:
            # staircase corners: a diagonal plus a straight hug walls steeper than 45 degrees
            for p_ in lengths:
                for q in lengths:
                    for s_ in lengths:
                        if q + s_ not in buildable:
                            continue
                        if not self._rect_fits(p_, q + s_):
                            continue  # the stair contains a p x (q + s) rectangle
                        for c in lengths:
                            if c > 10:
                                break
                            k = c * r2
                            area = (p_ + k) * (q + s_ + k) - c * s_ * r2 - c * c / 2
                            per = 2 * (p_ + q + s_ + c)
                            # true diameter over the seven vertices
                            pts = [
                                (0.0, 0.0),
                                (-p_, 0.0),
                                (-p_, -q),
                                (-p_ + k, -q - k),
                                (-p_ + k, -q - k - s_),
                                (k, -q - k - s_),
                                (k, -k),
                            ]
                            dm = max(
                                math.hypot(x1 - x0, y1 - y0)
                                for i, (x0, y0) in enumerate(pts)
                                for x1, y1 in pts[i + 1 :]
                            )
                            r = min(p_, q) / 2
                            add("stair", (p_, q, c, s_), area, per, dm, r)
                            add("stair2", (p_, q, c, s_), area, per, dm, r)
        # bands: long thin shapes whose faces are kept under 30 by V-dips
        for k in (1, 2, 3, 4):
            if right < 4 + 2 * k or diag < 4 * k:
                continue
            for s_ in lengths:
                for d in lengths:
                    if d > 10:
                        break
                    for h in lengths:
                        if h <= d * math.sqrt(2) + 1e-9:
                            continue  # the dips from both sides would meet
                        w = (k + 1) * s_ + k * d * math.sqrt(2)
                        area = w * h - k * d * d
                        per = 2 * (k + 1) * s_ + 4 * k * d + 2 * h
                        add(
                            "band",
                            (s_, k, d, h),
                            area,
                            per,
                            math.hypot(w, h),
                            min(s_, h) / 2,
                        )
        # bands with unequal runs (s1, s2) or (s1, s2, s1)
        for k in (1, 2):
            if right < 4 + 2 * k or diag < 4 * k:
                continue
            for s1 in lengths:
                for s2 in lengths:
                    if s1 == s2:
                        continue
                    axis = s1 + s2 if k == 1 else 2 * s1 + s2
                    for d in lengths:
                        if d > 10:
                            break
                        for h in lengths:
                            if h <= d * math.sqrt(2) + 1e-9:
                                continue
                            w = axis + k * d * math.sqrt(2)
                            area = w * h - k * d * d
                            per = 2 * axis + 4 * k * d + 2 * h
                            add(
                                "band2",
                                (s1, s2, k, d, h),
                                area,
                                per,
                                math.hypot(w, h),
                                min(s1, s2, h) / 2,
                            )
        out.sort(key=lambda t: (t[1] in LATE_FAMILIES, -t[0]))
        return out

    def _templates(self, work):
        """Work through the candidates, best upper bound first, until `work`
        containment checks have been spent. We remember where we stopped so a
        later call can carry on. A template that does not fit is tried smaller
        (committing any size that does), and the smallest size that still does
        not fit is remembered so that every larger member of the family can be
        skipped (for the families where larger parameters contain the shape)."""
        if self._cands is None:
            self._cands = self._template_candidates()
            self._next = 0
            first_late = next(
                (i for i, c in enumerate(self._cands) if c[1] in LATE_FAMILIES),
                len(self._cands),
            )
            self._seg_ends = sorted({first_late, len(self._cands)})
            self._failed = {}  # family -> minimal parameter sets that didn't fit
            self._lengths_sorted = self._side_lengths()
        spent = self.placer.checks
        while self._next < len(self._cands):
            bound, family, params = self._cands[self._next]
            if bound <= self.best_score:
                # best first within its segment, so skip to the next segment
                self._next = next(e for e in self._seg_ends if e > self._next)
                continue
            if self._done() or self.placer.checks - spent > work:
                return
            self._next += 1
            # a rectangle fits the same either way round
            key = tuple(sorted(params)) if family == "rect" else params
            if family in DOMINANCE_OK and any(
                all(k >= f for k, f in zip(key, small))
                for small in self._failed.get(family, ())
            ):
                continue  # a smaller one of this family already failed to fit
            shape = TEMPLATES[family](*params)
            if not shape.valid() or not self._connectors_ok(shape):
                continue
            placed = self._try_template(family, params, shape)
            if placed is None or placed:
                continue
            if self._done():
                self._next -= 1  # ran out of time, not a real miss
                return
            self._shrink(family, params)

    def _try_template(self, family, params, shape):
        """Allocate and place one template: True if committed, False if it
        does not fit, None if it cannot be built at all."""
        placement, placed = None, False
        for n_gates in self.gate_counts:
            if (
                self._score(shape.area, shape.perimeter, n_gates == 2)
                <= self.best_score
            ):
                continue
            pieces = self.alloc.allocate(shape.lengths, n_gates)
            if pieces is None:
                continue
            if not placed:
                placed = True
                placement = self.placer.place(shape)
            if placement is None:
                return False
            desc = f"{family} {params}" + (" with 2 gates" if n_gates == 2 else "")
            self._commit(shape, pieces, *placement, desc)
        return True if placed else None

    def _shrink(self, family, params):
        """A template that did not fit: bisect a uniform scale below it, commit
        any size that fits, and remember the smallest size that did not."""
        lengths = self._lengths_sorted
        if not lengths:
            return
        lo, hi, failed = 0.0, 1.0, params
        for _ in range(3):
            scale = (lo + hi) / 2
            small = []
            for i, p in enumerate(params):
                if (family == "band" and i == 1) or (family == "band2" and i == 2):
                    small.append(p)  # the number of dips is not a length
                    continue
                want = max(lengths[0], p * scale)
                j = bisect.bisect_right(lengths, want + 1e-9) - 1
                small.append(lengths[max(0, j)])
            small = tuple(small)
            if small == params:
                break
            shape = TEMPLATES[family](*small)
            if not shape.valid() or not self._connectors_ok(shape):
                hi = scale
                continue
            placed = self._try_template(family, small, shape)
            if placed is None or self._done():
                hi = scale
                continue
            if placed:
                lo = scale
            else:
                hi = scale
                failed = small
        if family not in DOMINANCE_OK:
            return
        key = tuple(sorted(failed)) if family == "rect" else failed
        memo = self._failed.setdefault(family, [])
        if any(all(k >= f for k, f in zip(key, small)) for small in memo):
            return  # a smaller failure already implies this one
        memo[:] = [m for m in memo if not all(m_ >= k for m_, k in zip(m, key))]
        if len(memo) < MAX_FAILED_MEMO:
            memo.append(key)

    def _rect_fits(self, a, b):
        """Whether an a x b rectangle fits somewhere in the room (memoised): a
        template that contains such a rectangle cannot fit when it does not."""
        key = (min(a, b), max(a, b))
        hit = self._rect_fit_memo.get(key)
        if hit is None:
            hit = self.placer.place(TEMPLATES["rect"](*key)) is not None
            self._rect_fit_memo[key] = hit
        return hit

    # the room itself

    def _room_trace(self):
        """The room's own outline, and its insets by 1 to 3 units, as candidates
        whenever every wall is on the 45-degree grid with a buildable length:
        in a room of short walls nothing beats following them."""
        polys = [self.room.polygon]
        for k in (1, 2, 3):
            try:
                inset = self.room.polygon.buffer(-k, join_style=2)
            except Exception:  # noqa: BLE001 - an inset that cannot be built is simply not a candidate
                inset = None
            if (
                inset is not None
                and not inset.is_empty
                and inset.geom_type == "Polygon"
            ):
                polys.append(inset)
        for poly in polys:
            pts = list(poly.exterior.coords)[:-1]
            if _area(pts) < 0:
                pts.reverse()
            n = len(pts)
            base = _angle(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
            dirs, lengths, ok = [], [], True
            for i in range(n):
                (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % n]
                length = math.hypot(x1 - x0, y1 - y0)
                turn = ((_angle(x1 - x0, y1 - y0) - base) % 360.0) / 45.0
                if abs(turn - round(turn)) > 1e-4 or abs(length - round(length)) > 1e-6:
                    ok = False
                    break
                d = round(turn) % 8
                if dirs and dirs[-1] == d:  # a collinear corner: one side
                    lengths[-1] += round(length)
                else:
                    dirs.append(d)
                    lengths.append(round(length))
            if not ok or len(dirs) < 4:
                continue
            if dirs[0] == dirs[-1]:  # the first and last side are one side
                lengths[0] += lengths.pop()
                dirs.pop()
            if any(not 5 <= length <= 30 for length in lengths):
                continue
            shape = Shape("trace", (len(dirs),), dirs, lengths)
            if shape.valid():
                self._commit_any(
                    shape, base, pts[0][0], pts[0][1], f"room trace ({len(dirs)} sides)"
                )

    # grid outlines

    def _try_grid_shape(self, shape, theta, tx, ty):
        label = f"grid outline ({len(shape.lengths)} sides)"
        return self._commit_any(shape, theta, tx, ty, label, skip_worse=True)

    # bump growth

    def _refine(self):
        if self.conn["right"] < 6:
            return
        refiner = Refiner(
            self.placer,
            self._side_lengths(),
            self.weights,
            self.conn["right"],
            self._done,
        )
        for _, shape, theta, tx, ty in list(self.seeds)[:MAX_REFINE_SEEDS]:
            key = (shape.family, tuple(shape.lengths))
            if key in self._refined:
                continue
            self._refined.add(key)
            self._posed.add(self._pose_key(shape, theta, tx, ty))
            if self._done():
                return
            self._grow(refiner, shape, theta, tx, ty)
        # the best rectangles are grown from other places they fit as well: a
        # seed's first pose says little about how far it can grow
        rects = [s for s in self.seeds if s[1].family == "rect"][:POSE_SEEDS]
        for _, shape, _, _, _ in rects:
            poses, scanned = 0, 0
            for theta, tx, ty in self.placer._anchored(shape):
                scanned += 1
                if self._done() or poses >= POSES_PER_SEED or scanned > 600:
                    break
                key = self._pose_key(shape, theta, tx, ty)
                if key in self._posed:
                    continue
                if not self.placer.fits(shape.world_vertices(theta, tx, ty)):
                    continue
                self._posed.add(key)
                poses += 1
                self._grow(refiner, shape, theta, tx, ty)

    @staticmethod
    def _pose_key(shape, theta, tx, ty):
        return (shape.family, tuple(shape.lengths)) + tuple(
            sorted(
                (round(x, 3), round(y, 3))
                for x, y in shape.world_vertices(theta, tx, ty)
            )
        )

    def _grow(self, refiner, shape, theta, tx, ty, rng=None, top_k=1):
        """Grow the shape by bumps, committing every valid step. False means we
        ran out of time."""
        fails = 0
        for grown, th, gx, gy in refiner.refine(
            shape, theta, tx, ty, rng=rng, top_k=top_k
        ):
            if self._done():
                return False
            if not self._connectors_ok(grown):
                break
            label = f"{shape.family} grown by bumps ({len(grown.lengths)} sides)"
            _, ok, _ = self._commit_any(grown, th, gx, gy, label)
            fails = 0 if ok else fails + 1
            if fails >= 3:
                break
        return True

    def _random_start(self, rng, small, angles):
        """A small random square somewhere in the room, or None."""
        minx, miny, maxx, maxy = self.placer.bounds
        for _ in range(60):
            a, b = rng.choice(small), rng.choice(small)
            shape = rectangle(a, b)
            if not shape.valid() or not self._connectors_ok(shape):
                continue
            px, py = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if not self.placer.point_ok(px, py):
                continue
            theta = rng.choice(angles) + 90.0 * rng.randrange(4)
            x, y = _rot(a / 2, b / 2, theta)
            tx, ty = px - x, py - y
            # snap to the room's integer lattice in this rotation: a start that
            # is flush with a wall grows further than one floating 0.37 off it
            vx, vy = self.placer.pts[0]
            dx, dy = _rot(tx - vx, ty - vy, -theta)
            wx, wy = _rot(round(dx), round(dy), theta)
            for cand in ((vx + wx, vy + wy), (tx, ty)):
                if self.placer.fits(shape.world_vertices(theta, *cand)):
                    return shape, theta, cand[0], cand[1]
        return None

    def _explore(self):
        """Random restarts: drop a small square somewhere, then grow it by
        randomly chosen bumps. One greedy growth gets stuck in a single local
        optimum; different growth orders find the winding shapes (star arms,
        letter strokes) it misses. Stops after MAX_RESTARTS, or after
        IDLE_RESTARTS in a row that did not improve the best score."""
        if self.conn["right"] < 6 or self._done():
            return
        lengths = self._side_lengths()
        if not lengths:
            return
        rng = random.Random(8)
        refiner = Refiner(
            self.placer, lengths, self.weights, self.conn["right"], self._done
        )
        angles = sorted({round(a % 90.0, 4) for a in self.placer.out_angle}) or [0.0]
        misses = idle = 0
        while (
            not self._done() and self._restarts < MAX_RESTARTS and idle < IDLE_RESTARTS
        ):
            self._restarts += 1
            before = self.best_score
            if self.seeds and rng.random() < 0.34:
                _, *start = rng.choice(self.seeds[:MAX_REFINE_SEEDS])
            else:
                start = self._random_start(rng, lengths[:4], angles)
            if start is None:
                # no square fits (tight room): grow from something we already have
                if not self.seeds:
                    misses += 1
                    if misses > 200:
                        return
                    continue
                _, *start = rng.choice(self.seeds)
            if not self._grow(refiner, *start, rng=rng, top_k=rng.choice((2, 3, 5))):
                return
            idle = 0 if self.best_score > before else idle + 1
