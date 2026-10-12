"""Group 6: beam search over closed chains of faces with 0/45/90 degree turns.

Ideas
-----
* A *face* is a straight run of pieces joined by straight connectors. Its
  length is what matters for the outline; which pieces make it up is decided
  afterwards (gate faces preferred, so the second-gate bonus is always taken
  when the inventory allows it).
* Coordinates are exact: x = m + n*(sqrt(2)/2) with integer (m, n). A loop
  closes iff all four integers are zero, so diagonal faces whose irrational
  parts do not cancel are never produced.
* Beam search grows chains one face at a time (small shapes first); every
  closed outline is fit-checked against the room in order of best possible
  score and the best valid one so far is kept. A CPU timer stops the search
  and returns that best-so-far.
"""

import math
import time
from itertools import pairwise

from shapely import Polygon

from players.player0 import Player
from players.player6.polygon import find_fit
from src.constants import BASELINE_SCORE
from src.enclosure import Construction
from src.pieces import Connector, ConnectorType, Piece, PieceType

SQ = math.sqrt(2) / 2

# heading h (multiples of 45 degrees) -> per unit length: (dx_m, dx_n, dy_m, dy_n)
# with x = m + n * sqrt(2)/2
DIRS = (
    (1, 0, 0, 0),
    (0, 1, 0, 1),
    (0, 0, 1, 0),
    (0, -1, 0, 1),
    (-1, 0, 0, 0),
    (0, -1, 0, -1),
    (0, 0, -1, 0),
    (0, 1, 0, -1),
)
# turns, in 45 degree units, tried after each face. +/-2 use a right
# connector (flipped for -2), +/-1 a diagonal connector (flipped for -1).
TURNS = (2, -2, 1, -1)

GATE = 0
WALL = 1


def _ccw(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(p1, p2, p3, p4, eps=1e-9):
    d1 = _ccw(p3, p4, p1)
    d2 = _ccw(p3, p4, p2)
    d3 = _ccw(p1, p2, p3)
    d4 = _ccw(p1, p2, p4)
    return ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and (
        (d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)
    )


class Chain:
    """A partial boundary: faces laid end to end from the origin."""

    __slots__ = ("diag", "faces", "perim", "pos", "right", "straights", "turn", "verts")

    def __init__(self, pos, turn, faces, verts, perim, right, diag, straights):
        self.pos = pos  # exact (xm, xn, ym, yn)
        self.turn = turn  # cumulative turn, 45-degree units
        self.faces = faces  # tuple of (length, turn_after)
        self.verts = verts  # float vertices: start of each face, then the end
        self.perim = perim
        self.right = right
        self.diag = diag
        self.straights = straights  # lower bound on straight connectors used


class Player6(Player):
    """Group 6: beam search over exact 45-degree-turn face chains."""

    # Stop searching this many CPU seconds into the run (hard limit is 300s).
    TIME_BUDGET = 240.0
    MAX_FACES = 12
    BEAM_WIDTH = 150  # first pass; widened x4 while time allows
    PERIM_BUCKET = 8  # beam keeps shapes of every size, not just the biggest

    class OutOfTime(Exception):
        pass

    def __init__(self, *args, **kwargs) -> None:
        # process_time() is what the simulator's CPU limit counts; the
        # constructor counts toward it, so the clock starts here.
        self._start = time.process_time()
        super().__init__(*args, **kwargs)

    def out_of_time(self) -> bool:
        return time.process_time() - self._start >= self.TIME_BUDGET

    # ---- inventory / face library -------------------------------------

    def prepare_inventory(self) -> None:
        inv = self.inventory
        self.straight_count = inv.connectors.get(ConnectorType.STRAIGHT, 0)
        self.right_count = inv.connectors.get(ConnectorType.RIGHT, 0)
        self.diag_count = inv.connectors.get(ConnectorType.DIAGONAL, 0)

        self.stock: dict[tuple[int, int], int] = {}
        for length, n in inv.walls.items():
            if n > 0:
                self.stock[(WALL, length)] = n
        for length, n in inv.gates.items():
            if n > 0:
                self.stock[(GATE, length)] = n
        self.total_length = sum(k[1] * n for k, n in self.stock.items())
        self.total_gates = sum(n for k, n in self.stock.items() if k[0] == GATE)

        # comps[length] = candidate compositions of a face of that length:
        # (pieces, gate_count, straight_connectors_needed), gate faces first
        # (fewest gates, then fewest pieces).
        keys = sorted(self.stock)
        comps: dict[int, list] = {}

        def rec(i, chosen, total):
            if i == len(keys):
                if chosen and total >= 5:
                    pieces = tuple(chosen)
                    gates = sum(1 for t, _ in pieces if t == GATE)
                    comps.setdefault(total, []).append((pieces, gates, len(pieces) - 1))
                return
            t, length = keys[i]
            most = min(
                self.stock[keys[i]],
                (30 - total) // length,
                self.straight_count + 1 - len(chosen),
            )
            for c in range(most + 1):
                rec(i + 1, chosen + [(t, length)] * c, total + c * length)

        rec(0, [], 0)
        for options in comps.values():
            options.sort(key=lambda c: (0 if c[1] > 0 else 1, c[1], len(c[0])))
        self.comps = comps
        self.lengths = sorted(comps)
        self.min_straights = {
            length: min(c[2] for c in cs) for length, cs in comps.items()
        }
        self._realize_cache: dict[tuple, list | None] = {}

    def realize(self, lengths: tuple[int, ...]):
        """Assign a piece composition to each face length (a sorted tuple).
        Gate-containing compositions are tried first. Returns a list aligned
        with `lengths`, or None if the inventory cannot supply them."""
        if lengths in self._realize_cache:
            return self._realize_cache[lengths]
        order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
        remaining = dict(self.stock)
        assign: list = [None] * len(lengths)
        budget = [4000]

        def dfs(k, straights_left):
            if k == len(order):
                return True
            budget[0] -= 1
            if budget[0] < 0:
                return False
            i = order[k]
            for comp in self.comps[lengths[i]]:
                pieces, _, s = comp
                if s > straights_left:
                    continue
                need: dict = {}
                ok = True
                for p in pieces:
                    need[p] = need.get(p, 0) + 1
                    if need[p] > remaining.get(p, 0):
                        ok = False
                        break
                if not ok:
                    continue
                for p, c in need.items():
                    remaining[p] -= c
                assign[i] = comp
                if dfs(k + 1, straights_left - s):
                    return True
                for p, c in need.items():
                    remaining[p] += c
            return False

        result = assign if dfs(0, self.straight_count) else None
        self._realize_cache[lengths] = result
        return result

    # ---- chain search --------------------------------------------------

    def _closed_score(self, verts, perim) -> float:
        n = len(verts)
        area2 = 0.0
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            area2 += x1 * y2 - x2 * y1
        closing = math.hypot(verts[-1][0] - verts[0][0], verts[-1][1] - verts[0][1])
        return self.weights.A * (area2 / 2) + self.weights.C * (perim + closing)

    def expand(self, ch: Chain, closed: list) -> list[Chain]:
        out: list[Chain] = []
        nf = len(ch.faces) + 1
        rf = self.MAX_FACES - nf
        dxm, dxn, dym, dyn = DIRS[ch.turn % 8]
        px, py = ch.verts[-1]
        for length in self.lengths:
            perim = ch.perim + length
            if perim > self.total_length:
                break
            straights = ch.straights + self.min_straights[length]
            if straights > self.straight_count:
                continue
            xm, xn, ym, yn = ch.pos
            npos = (
                xm + dxm * length,
                xn + dxn * length,
                ym + dym * length,
                yn + dyn * length,
            )
            nx, ny = npos[0] + npos[1] * SQ, npos[2] + npos[3] * SQ
            if math.hypot(nx, ny) > self.room_diam:
                continue
            lengths_key = tuple(sorted([f[0] for f in ch.faces] + [length]))
            if self.realize(lengths_key) is None:
                continue
            for t in TURNS:
                if abs(t) == 2:
                    right, diag = ch.right + 1, ch.diag
                    if right > self.right_count:
                        continue
                else:
                    right, diag = ch.right, ch.diag + 1
                    if diag > self.diag_count:
                        continue
                nturn = ch.turn + t
                if abs(8 - nturn) > 2 * rf:
                    continue
                if npos == (0, 0, 0, 0) and nturn == 8:
                    if nf >= 4:
                        closed.append(((*ch.faces, (length, t)), perim))
                    continue
                # Centrally symmetric outline: a half-chain turning 180
                # degrees, repeated, closes by itself (second half is the
                # first rotated), whatever the lengths.
                if (
                    nturn == 4
                    and 2 * nf <= self.MAX_FACES
                    and 2 * right <= self.right_count
                    and 2 * diag <= self.diag_count
                ):
                    half = (*ch.faces, (length, t))
                    closed.append((half + half, 2 * perim))
                if nf >= self.MAX_FACES:
                    continue
                if diag == self.diag_count and (npos[1] or npos[3]):
                    continue  # irrational part can no longer be cancelled
                # can we still get home?
                if math.hypot(nx, ny) > min(30 * rf, self.total_length - perim):
                    continue
                # new face must not cross earlier ones
                v = ch.verts
                if any(
                    _segments_cross(v[i], v[i + 1], (px, py), (nx, ny))
                    for i in range(len(v) - 2)
                ):
                    continue
                if not self.may_fit([*v, (nx, ny)]):
                    continue
                out.append(
                    Chain(
                        npos,
                        nturn,
                        (*ch.faces, (length, t)),
                        [*v, (nx, ny)],
                        perim,
                        right,
                        diag,
                        straights,
                    )
                )
        return out

    def may_fit(self, pts) -> bool:
        """Necessary condition: at one of the rotations we try, the chain's
        bounding box is no bigger than the room's bounding box."""
        for c, s in self._rot:
            xs = [x * c - y * s for x, y in pts]
            ys = [x * s + y * c for x, y in pts]
            if (
                max(xs) - min(xs) <= self.room_w + 1e-9
                and max(ys) - min(ys) <= self.room_h + 1e-9
            ):
                return True
        return False

    def potential(self, perim: float, area: float) -> float:
        bonus = self.weights.G if self.total_gates >= 2 else 0.0
        return self.weights.A * area + self.weights.C * perim + bonus

    def fit_angles(self) -> list[float]:
        """Rotations worth trying. Outline edges are multiples of 45 degrees,
        so a side lies flush on a room wall of direction a when the rotation
        is a mod 45 plus any multiple of 45 (the shape is not symmetric, so
        all eight are different placements)."""
        if getattr(self, "_angles", None) is None:
            coords = list(self.room.polygon.exterior.coords)
            base = {0.0}
            for (x1, y1), (x2, y2) in pairwise(coords):
                a = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 45.0
                base.add(round(a, 6) % 45.0)
            self._angles = sorted(
                {(b + 45.0 * j) % 360.0 for b in base for j in range(8)}
            )
        return self._angles

    @staticmethod
    def canonical(faces):
        """Rotate the cyclic (length, turn) list to its smallest rotation."""
        faces = list(faces)
        k = min(range(len(faces)), key=lambda i: faces[i:] + faces[:i])
        return tuple(faces[k:] + faces[:k])

    @staticmethod
    def walk(faces):
        """Float vertices (start of each face) of a face cycle from heading 0."""
        pos = (0, 0, 0, 0)
        turn = 0
        verts = []
        for length, t in faces:
            xm, xn, ym, yn = pos
            verts.append((xm + xn * SQ, ym + yn * SQ))
            dxm, dxn, dym, dyn = DIRS[turn % 8]
            pos = (
                xm + dxm * length,
                xn + dxn * length,
                ym + dym * length,
                yn + dyn * length,
            )
            turn += t
        return verts

    def search(self):
        self.prepare_inventory()
        room_poly = self.room.polygon
        rx0, ry0, rx1, ry1 = room_poly.bounds
        self.room_diam = math.hypot(rx1 - rx0, ry1 - ry0)
        self.room_w, self.room_h = rx1 - rx0, ry1 - ry0
        self._rot = [
            (math.cos(math.radians(a)), math.sin(math.radians(a)))
            for a in self.fit_angles()
        ]
        room_area = room_poly.area
        if (
            self.right_count + self.diag_count < 4
            or not self.lengths
            or self.total_gates < 1
        ):
            return None

        best = None  # (score, faces, per-face compositions, fit)
        seen: set = set()
        fit_cache: dict = {}

        def process(closed):
            nonlocal best
            cands = []
            for faces, _perim in closed:
                key = Player6.canonical(faces)
                if key in seen:
                    continue
                seen.add(key)
                poly = Polygon(Player6.walk(key))
                if not poly.is_valid or poly.area <= 0 or poly.area > room_area:
                    continue
                cands.append((self.potential(poly.length, poly.area), key, poly))
            cands.sort(key=lambda c: -c[0])
            for pot, key, poly in cands:
                if self.out_of_time():
                    raise Player6.OutOfTime()
                if best is not None and pot <= best[0]:
                    break
                lengths = tuple(sorted(f[0] for f in key))
                assign = self.realize(lengths)
                if assign is None:
                    continue
                pool: dict[int, list] = {}
                for ln, comp in zip(lengths, assign):
                    pool.setdefault(ln, []).append(comp)
                per_face = [pool[f[0]].pop(0) for f in key]
                gate_faces = sum(1 for c in per_face if c[1] > 0)
                if gate_faces == 0:
                    continue
                if key not in fit_cache:
                    fit_cache[key] = find_fit(
                        room_poly, poly, angles=self.fit_angles(), stop=self.out_of_time
                    )
                    if self.out_of_time():
                        raise Player6.OutOfTime()
                fit = fit_cache[key]
                if fit is None:
                    continue
                score = (
                    self.weights.A * poly.area
                    + self.weights.C * poly.length
                    + (self.weights.G if gate_faces >= 2 else 0.0)
                )
                if best is None or score > best[0]:
                    best = (score, key, per_face, fit)

        # Iterative widening: a cheap narrow beam first (answer within a
        # moment), then ever wider ones, until the beam stops truncating
        # (search is then exhaustive) or the CPU budget is spent.
        try:
            width = self.BEAM_WIDTH
            while True:
                self._truncated = False
                self._width = width
                beam = [Chain((0, 0, 0, 0), 0, (), [(0.0, 0.0)], 0, 0, 0, 0)]
                for _level in range(self.MAX_FACES):
                    closed: list = []
                    nxt: dict = {}
                    for ch in beam:
                        if self.out_of_time():
                            raise Player6.OutOfTime()
                        for c in self.expand(ch, closed):
                            key = (c.pos, c.turn, tuple(sorted(f[0] for f in c.faces)))
                            sc = self._closed_score(c.verts, c.perim)
                            old = nxt.get(key)
                            if old is None or sc > old[0]:
                                nxt[key] = (sc, c)
                    process(closed)
                    beam = self.select_beam(list(nxt.values()))
                    if not beam:
                        break
                if not self._truncated:
                    break
                width *= 4
        except Player6.OutOfTime:
            self.note = "CPU budget reached; returning best found so far"
        return best

    def select_beam(self, scored):
        """Top `_width` states, split evenly over perimeter bands so small
        outlines survive next to the (higher scoring, maybe unfittable) big
        ones; leftover slots go to the best of the rest."""
        buckets: dict[int, list] = {}
        for sc, c in scored:
            buckets.setdefault(int(c.perim // self.PERIM_BUCKET), []).append((sc, c))
        if not buckets:
            return []
        if len(scored) > self._width:
            self._truncated = True
        per = max(1, self._width // len(buckets))
        keep, rest = [], []
        for b in buckets.values():
            b.sort(key=lambda s: -s[0])
            keep += b[:per]
            rest += b[per:]
        room_left = self._width - len(keep)
        if room_left > 0:
            rest.sort(key=lambda s: -s[0])
            keep += rest[:room_left]
        return [c for _, c in keep]

    # ---- building the answer --------------------------------------------

    def build_enclosure(self) -> Construction | None:
        best = self.search()
        # a valid enclosure scores BASELINE + best[0]; None scores 0
        if best is None or BASELINE_SCORE + best[0] <= 0:
            return None
        _, faces, per_face, (fit, _params) = best

        coords = list(fit.exterior.coords)
        (x1, y1), (x2, y2) = coords[0], coords[1]
        start_heading = math.degrees(math.atan2(y2 - y1, x2 - x1))

        pieces: list[Piece] = []
        after: list[Connector] = []  # joint after each piece
        for (_length, t), (comp_pieces, _g, _s) in zip(faces, per_face):
            for i, (ptype, plen) in enumerate(comp_pieces):
                pieces.append(
                    Piece(
                        piece_type=PieceType.GATE if ptype == GATE else PieceType.WALL,
                        length=plen,
                    )
                )
                if i < len(comp_pieces) - 1:
                    after.append(Connector(connector_type=ConnectorType.STRAIGHT))
            kind = ConnectorType.RIGHT if abs(t) == 2 else ConnectorType.DIAGONAL
            after.append(Connector(connector_type=kind, reflex=(t < 0)))

        # connectors[i] is the joint *before* piece i (connectors[0] closes
        # the loop), so rotate the "after" list by one.
        connectors = after[-1:] + after[:-1]
        return Construction(
            start=(x1, y1),
            start_heading=start_heading,
            pieces=pieces,
            connectors=connectors,
        )
