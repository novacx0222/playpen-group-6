"""Group 6: beam search over closed chains of faces with 0/45/90 degree turns.

Ideas
-----
* A *face* is a straight run of pieces joined by straight connectors. Its
  length is what matters for the outline; which pieces make it up is decided
  afterwards. Fitted candidates get a bounded retry for placing gates on
  separate faces, retaining the original legal plan if the retry stops early.
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
from collections import Counter

from shapely import Polygon

from players.player0 import Player
from players.player6.polygon import find_fit, room_search_angles
from src.constants import BASELINE_SCORE
from src.enclosure import Construction, score_construction, validate_construction
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


class DiagonalSearch(Player):
    """Group 6: beam search over exact 45-degree-turn face chains."""

    # Stop searching this many CPU seconds into the run (hard limit is 300s).
    TIME_BUDGET = 20.0
    MAX_FACES = 12
    BEAM_WIDTH = 150  # first pass; widened x4 while time allows
    PERIM_BUCKET = 8  # beam keeps shapes of every size, not just the biggest
    REALIZE_NODE_LIMIT = 4000
    STATE_VARIANTS = 2  # Distinct boundary prefixes retained per resource state.
    GATE_BONUS_SECONDS = 0.01  # Per closed-candidate retry, within the beam budget.
    GATE_BONUS_FRACTION = 0.05  # All retries together use at most 5% of beam CPU.

    class OutOfTime(Exception):
        pass

    def __init__(
        self, *args, time_budget=20.0, face_library=None, incumbent_score=0.0, **kwargs
    ) -> None:
        self.TIME_BUDGET = time_budget
        self.face_library = face_library
        self.incumbent_score = incumbent_score
        # process_time() is what the simulator's CPU limit counts; the
        # constructor counts toward it, so the clock starts here.
        self._start = time.process_time()
        self.history = []  # (CPU seconds since start, full simulator score)
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

        if self.face_library is None:
            rec(0, [], 0)
        else:
            for length, options in self.face_library.items():
                comps[length] = [
                    (tuple(pieces), sum(t == GATE for t, _ in pieces), len(pieces) - 1)
                    for pieces in options
                ]
        for options in comps.values():
            options.sort(key=lambda c: (0 if c[1] > 0 else 1, c[1], len(c[0])))
        self.comps = comps
        self.lengths = sorted(comps)
        self.min_straights = {
            length: min(c[2] for c in cs) for length, cs in comps.items()
        }
        stock_index = {piece: i for i, piece in enumerate(self.stock)}
        self._stock_counts = tuple(self.stock.values())
        # Material demand never changes between calls: count repeated pieces
        # once instead of rebuilding a dictionary at every DFS branch.
        self._realize_options = {
            length: [
                (comp, tuple((stock_index[p], n) for p, n in Counter(comp[0]).items()))
                for comp in options
            ]
            for length, options in comps.items()
        }
        self._realize_cache: dict[tuple, list | None] = {}
        self._gated_realize_cache: dict[tuple, list | None] = {}
        self._gate_bonus_cpu = 0.0

    def realize(self, lengths: tuple[int, ...], *, min_gate_faces=0, stop=None):
        """Assign a piece composition to each face length (a sorted tuple).
        Gate-containing compositions are tried first. Returns a list aligned
        with `lengths`, or None if no assignment was found. Only exhaustive
        failures are cached; a CPU/node cutoff leaves the query retryable.
        Gate constraints use a separate cache and count faces, not gate pieces.
        """
        cache = self._gated_realize_cache if min_gate_faces else self._realize_cache
        cache_key = (lengths, min_gate_faces) if min_gate_faces else lengths
        if cache_key in cache:
            return cache[cache_key]
        if min_gate_faces > min(self.total_gates, len(lengths)) or any(
            length not in self._realize_options for length in lengths
        ):
            cache[cache_key] = None
            return None
        # Assign the most constrained faces first, preserving gate-first
        # composition order and the caller's original face ordering.
        order = sorted(
            range(len(lengths)),
            key=lambda i: (len(self._realize_options[lengths[i]]), -lengths[i]),
        )
        remaining = list(self._stock_counts)
        assign: list = [None] * len(lengths)
        failed = set()
        nodes_left = self.REALIZE_NODE_LIMIT

        def dfs(k, straights_left, gates_needed):
            nonlocal nodes_left
            if self.out_of_time() or (stop is not None and stop()):
                return None  # Unknown, rather than proven infeasible.
            if k == len(order):
                return gates_needed == 0
            state = (k, straights_left, tuple(remaining))
            if min_gate_faces:
                state += (gates_needed,)
            if state in failed:
                return False
            if len(order) - k < gates_needed:
                return False
            if nodes_left <= 0:
                return None
            nodes_left -= 1
            i = order[k]
            for comp, need in self._realize_options[lengths[i]]:
                straights = comp[2]
                if straights > straights_left or any(remaining[p] < n for p, n in need):
                    continue
                for p, n in need:
                    remaining[p] -= n
                assign[i] = comp
                outcome = dfs(
                    k + 1,
                    straights_left - straights,
                    max(0, gates_needed - (comp[1] > 0)),
                )
                for p, n in need:
                    remaining[p] += n
                if outcome is not False:
                    return outcome
            failed.add(state)
            return False

        outcome = dfs(0, self.straight_count, min_gate_faces)
        result = assign if outcome is True else None
        if outcome is not None:
            cache[cache_key] = result
        return result

    def _improve_gate_assignment(self, lengths, assignment):
        """Try the second-face bonus without losing a known legal fallback."""
        if (
            self.weights.G <= 0
            or self.total_gates < 2
            or sum(comp[1] > 0 for comp in assignment) >= 2
        ):
            return assignment
        key = (lengths, 2)
        if key in self._gated_realize_cache:
            return self._gated_realize_cache[key] or assignment
        started = time.process_time()
        seconds = min(
            self.GATE_BONUS_SECONDS,
            self.TIME_BUDGET * self.GATE_BONUS_FRACTION - self._gate_bonus_cpu,
            self.TIME_BUDGET - (started - self._start),
        )
        if seconds <= 0:
            return assignment

        def stopped():
            return time.process_time() >= started + seconds

        try:
            return self.realize(lengths, min_gate_faces=2, stop=stopped) or assignment
        finally:
            self._gate_bonus_cpu += time.process_time() - started

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
            if self.out_of_time():
                raise DiagonalSearch.OutOfTime()
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
        """Use the same eight room-oriented angles as the orthogonal search."""
        if getattr(self, "_angles", None) is None:
            self._angles = room_search_angles(self.room.polygon)
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

        def threshold():
            return max(
                self.incumbent_score - BASELINE_SCORE,
                best[0] if best is not None else -math.inf,
            )

        def consider(key, per_face, fit):
            nonlocal best
            construction = self.to_construction(key, per_face, fit)
            result = validate_construction(construction, self.room, self.inventory)
            if not result.valid:
                return False
            score = score_construction(result, self.weights) - BASELINE_SCORE
            if score > threshold():
                best = (score, key, per_face, fit)
                self.history.append(
                    (time.process_time() - self._start, BASELINE_SCORE + score)
                )
            return True

        def order_assignment(key, lengths, assignment):
            pool = {}
            for length, comp in zip(lengths, assignment):
                pool.setdefault(length, []).append(comp)
            return [pool[length].pop(0) for length, _ in key]

        def process(closed):
            cands = []
            for faces, _perim in closed:
                key = DiagonalSearch.canonical(faces)
                if key in seen:
                    continue
                seen.add(key)
                poly = Polygon(DiagonalSearch.walk(key))
                if not poly.is_valid or poly.area <= 0 or poly.area > room_area:
                    continue
                cands.append((self.potential(poly.length, poly.area), key, poly))
            cands.sort(key=lambda c: -c[0])
            for pot, key, poly in cands:
                if self.out_of_time():
                    raise DiagonalSearch.OutOfTime()
                if pot <= threshold():
                    break
                lengths = tuple(sorted(f[0] for f in key))
                assign = self.realize(lengths)
                if assign is None:
                    continue
                if not any(comp[1] > 0 for comp in assign):
                    assign = self.realize(lengths, min_gate_faces=1)
                    if assign is None:
                        continue
                if key not in fit_cache:
                    fit_cache[key] = find_fit(
                        room_poly, poly, angles=self.fit_angles(), stop=self.out_of_time
                    )
                    if self.out_of_time():
                        raise DiagonalSearch.OutOfTime()
                fit = fit_cache[key]
                if fit is None:
                    continue
                per_face = order_assignment(key, lengths, assign)
                # Publish the legal baseline before spending time on the bonus.
                if not consider(key, per_face, fit) or pot <= threshold():
                    continue
                improved = self._improve_gate_assignment(lengths, assign)
                if improved is not assign:
                    consider(key, order_assignment(key, lengths, improved), fit)

        # Iterative widening: start narrow, then widen until the global beam
        # stops truncating or the CPU budget is spent. The per-state outline
        # cap remains a heuristic even when the global beam is wide enough.
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
                            raise DiagonalSearch.OutOfTime()
                        for c in self.expand(ch, closed):
                            self._keep_chain(nxt, c)
                    process(closed)
                    beam = self.select_beam(
                        [
                            candidate
                            for variants in nxt.values()
                            for candidate in variants
                        ]
                    )
                    if not beam:
                        break
                if not self._truncated:
                    break
                width *= 4
        except DiagonalSearch.OutOfTime:
            self.note = "CPU budget reached; returning best found so far"
        return best

    def _keep_chain(self, groups, chain):
        """Keep a bounded set of exact, distinct outlines for each resource state.

        Equal endpoints, headings and lengths do not imply equal futures:
        connector budgets and the existing boundary affect legal completions.
        Face sequences identify outlines exactly, without float rounding.
        """
        key = (
            chain.pos,
            chain.turn,
            tuple(sorted(length for length, _ in chain.faces)),
            chain.right,
            chain.diag,
            chain.straights,
        )
        variants = groups.get(key)
        if variants is None:
            groups[key] = [(self._closed_score(chain.verts, chain.perim), chain)]
            return
        if any(previous.faces == chain.faces for _, previous in variants):
            return
        score = self._closed_score(chain.verts, chain.perim)
        if len(variants) >= self.STATE_VARIANTS and score <= variants[-1][0]:
            return
        variants.append((score, chain))
        variants.sort(key=lambda candidate: -candidate[0])
        del variants[self.STATE_VARIANTS :]

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
        _, faces, per_face, fit = best
        return self.to_construction(faces, per_face, fit)

    @staticmethod
    def to_construction(faces, per_face, fit):
        """Keep the outline's traversal order and rotate outgoing connectors."""
        _, (theta, tx, ty) = fit
        x1, y1 = tx, ty
        start_heading = theta

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
