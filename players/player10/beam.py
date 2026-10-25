"""Free-form enclosures: a beam search that walks the boundary inside the room.

The walk is counter-clockwise. Each step adds one face (a length the inventory can
make) after a corner of +-90 or +-45 degrees. Faces are only kept if they lie inside
the room and stay clear of the earlier path, so containment is pruned as we go.

Closure is exact: headings are multiples of 45 degrees and lengths are integers, so
the loop closes iff the lengths walked along each of the axes 0/45/90/135 cancel
(sqrt(2) is irrational). `Walk.lattice` tracks those four integers.

Pieces are not tracked here; closed shapes are handed to Assigner.assign, which
works for any list of face lengths.
"""

import heapq
import math
import time
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.geometry.polygon import orient

from players.player10.shapes import FREE, Shape
from src.pieces import ConnectorType

TURNS = (90, 45, -45, -90)  # degrees, left (+) is counter-clockwise
MAX_FACES = 14
WIDTHS = (30, 120, 480)  # beam widths, narrowest first (anytime behaviour)


@dataclass(frozen=True)
class Walk:
    points: tuple  # corners so far; points[-1] is where the walk is now
    heading: int  # in 45-degree steps from the base angle, mod 8
    turned: int  # degrees turned at the corners so far
    lattice: tuple  # exact lengths walked along the 0/45/90/135 axes
    faces: tuple
    turns: tuple  # turns[i] = corner at the start of face i (turns[0] set on closing)
    rights: int
    diagonals: int
    # kept as the walk grows, so ranking a walk needs no pass over all its corners
    twice: float = 0.0  # twice the shoelace area of the path so far
    length: int = 0  # total length of its faces


def _step(h: int, length: int) -> list[int]:
    v = [0, 0, 0, 0]
    v[h % 4] = length if h < 4 else -length
    return v


def _wrap(angle: float) -> float:
    return (angle + 180) % 360 - 180


class Walker:
    def __init__(
        self,
        room_search,
        lengths,
        total_length,
        rights,
        diagonals,
        weights,
        can_supply=None,
        diverse=False,
    ) -> None:
        self.region = room_search.region
        self.lengths = sorted(lengths)
        self.length_set = set(lengths)
        self.total_length = total_length
        self.max_rights, self.max_diagonals = rights, diagonals
        self.A, self.C = weights.A, weights.C
        self.max_faces = MAX_FACES
        self.can_supply = can_supply
        self.diverse = diverse

    # ------------------------------------------------------------ scoring

    def _value(self, w: Walk) -> float:
        """Score if the walk were closed by a straight chord: what the beam ranks by."""
        (x0, y0), (x1, y1) = w.points[0], w.points[-1]
        area = 0.5 * abs(w.twice + x1 * y0 - x0 * y1)
        perimeter = w.length + math.dist(w.points[-1], w.points[0])
        return self.A * area + self.C * perimeter

    # ---------------------------------------------------------- expansion

    def _extend(self, w: Walk, base: float) -> list[Walk]:
        n = len(w.faces)
        used = sum(w.faces)
        earlier = LineString(w.points[:-1]) if len(w.points) > 2 else None
        lengths = [L for L in self.lengths if used + L <= self.total_length]
        if not lengths:
            return []
        out = []
        for turn in (0,) if n == 0 else TURNS:
            right = abs(turn) == 90
            rights = w.rights + (1 if turn and right else 0)
            diagonals = w.diagonals + (1 if turn and not right else 0)
            if rights > self.max_rights or diagonals > self.max_diagonals:
                continue
            turned = w.turned + turn
            # can the remaining faces still turn enough to come round?
            if turned + (self.max_faces - n) * 90 < 360:
                continue
            h = (w.heading + turn // 45) % 8
            t = math.radians(base + 45 * h)
            x, y = w.points[-1]
            ends = np.array(
                [(x + L * math.cos(t), y + L * math.sin(t)) for L in lengths]
            )
            starts = np.tile([x, y], (len(lengths), 1))
            segs = shapely.linestrings(np.stack([starts, ends], axis=1))
            ok = shapely.contains(self.region, segs)
            if earlier is not None:
                ok &= ~shapely.intersects(segs, earlier)
            for L, end, good in zip(lengths, ends, ok):
                if good:
                    faces = w.faces + (L,)
                    if self.can_supply is not None and not self.can_supply(faces):
                        continue
                    lattice = tuple(a + b for a, b in zip(w.lattice, _step(h, L)))
                    out.append(
                        Walk(
                            w.points + ((float(end[0]), float(end[1])),),
                            h,
                            turned,
                            lattice,
                            faces,
                            w.turns + (turn,),
                            rights,
                            diagonals,
                            w.twice + x * float(end[1]) - float(end[0]) * y,
                            w.length + L,
                        )
                    )
        return out

    def _closing(self, w: Walk):
        """(faces, turns) if one more face brings the walk exactly back to its start."""
        if len(w.points) < 4:
            return None
        need = [-v for v in w.lattice]
        nonzero = [k for k in range(4) if need[k]]
        if len(nonzero) != 1:
            return None
        k = nonzero[0]
        length = abs(need[k])
        if length not in self.length_set or sum(w.faces) + length > self.total_length:
            return None
        h = k if need[k] > 0 else k + 4
        t1, t2 = _wrap((h - w.heading) * 45), _wrap(-h * 45)
        if t1 not in TURNS or t2 not in TURNS or w.turned + t1 + t2 != 360:
            return None
        rights = w.rights + sum(abs(t) == 90 for t in (t1, t2))
        diagonals = w.diagonals + sum(abs(t) == 45 for t in (t1, t2))
        if rights > self.max_rights or diagonals > self.max_diagonals:
            return None
        seg = LineString([w.points[-1], w.points[0]])
        if not shapely.contains(self.region, seg):
            return None
        touch = shapely.intersection(seg, LineString(w.points[:-1]))
        if not touch.equals(Point(w.points[0])):
            return None
        return w.faces + (length,), (int(t2),) + w.turns[1:] + (int(t1),)

    # ------------------------------------------------------------- search

    def run(self, start, base: float, width: int, deadline: float):
        """Closed polygons found by a beam of `width` walks from `start`, whose first
        face heads along `base` degrees."""
        beam = [Walk((start,), 0, 0, (0, 0, 0, 0), (), (), 0, 0)]
        closed = []
        for _ in range(self.max_faces):
            nxt: dict = {}
            for w in beam:
                if time.process_time() > deadline:
                    return closed
                for child in self._extend(w, base):
                    done = self._closing(child)
                    if done:
                        closed.append(done)
                    value = self._value(child)
                    key = (child.lattice, child.heading, child.turned, round(value, 1))
                    if self.diverse:
                        # Equal endpoints can have very different remaining stock
                        # and geometry.  Keeping the face multiset prevents a broad
                        # rectangle from erasing every irregular path that could
                        # later close and fit a non-convex room.
                        key += (tuple(sorted(child.faces)),)
                    old = nxt.get(key)
                    if old is None or value > old[0]:
                        nxt[key] = (value, child)
            beam = self._select(list(nxt.values()), width)
            if not beam:
                break
        return closed

    def _select(self, scored: list[tuple[float, Walk]], width: int) -> list[Walk]:
        """Keep high-value paths without losing every non-rectangular topology.

        The usual short room walk benefits from a pure score beam.  The deeper
        diversity-enabled walk groups states by material used, number of faces,
        and diagonal corners.  These resources distinguish useful concave and
        angled loops from the rectangles favored by the partial-chord score.
        """
        if not self.diverse:
            return [
                w for _, w in heapq.nlargest(width, scored, key=lambda item: item[0])
            ]
        buckets: dict[tuple[int, int, int], list[tuple[float, Walk]]] = {}
        for value, walk in scored:
            key = (sum(walk.faces) // 8, len(walk.faces), walk.diagonals)
            buckets.setdefault(key, []).append((value, walk))
        if not buckets:
            return []
        share = max(1, width // len(buckets))
        kept: list[tuple[float, Walk]] = []
        rest: list[tuple[float, Walk]] = []
        for bucket in buckets.values():
            bucket.sort(key=lambda item: item[0], reverse=True)
            kept.extend(bucket[:share])
            rest.extend(bucket[share:])
        if len(kept) < width:
            kept.extend(
                heapq.nlargest(width - len(kept), rest, key=lambda item: item[0])
            )
        elif len(kept) > width:
            kept = heapq.nlargest(width, kept, key=lambda item: item[0])
        return [walk for _, walk in kept]


def _make_shape(faces: tuple, turns: tuple) -> Shape:
    draft = Shape(FREE, faces, faces, tuple(float(t) for t in turns), 0.0, 0.0)
    pts = np.array(draft.vertices())
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return Shape(FREE, faces, faces, draft.turns, float(area), float(sum(faces)))


def free_shapes(
    room_search,
    assigner,
    inventory,
    weights,
    deadline,
    *,
    widths=WIDTHS,
    max_faces=MAX_FACES,
    max_starts=None,
    diverse_starts=False,
    diverse_beam=False,
):
    """Yield (shape, start, heading) for closed polygons that fit the room, widening
    the beam each round and best-ranked first within a round. The caller checks the
    pieces (Assigner) and the simulator's validator."""
    walker = Walker(
        room_search,
        assigner.lengths,
        assigner.total_length,
        inventory.connectors.get(ConnectorType.RIGHT, 0),
        inventory.connectors.get(ConnectorType.DIAGONAL, 0),
        weights,
        assigner.can_supply,
        diverse=diverse_beam,
    )
    walker.max_faces = max_faces
    ring = list(orient(room_search.polygon, 1.0).exterior.coords)[:-1]
    starts = []
    for i, p in enumerate(ring):
        q = ring[(i + 1) % len(ring)]
        # first face runs along the wall leaving this corner (interior on the left)
        length = math.dist(p, q)
        base = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
        starts.append((length, i, p, base))
        if diverse_starts:
            midpoint = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
            # A high-area loop need not begin at a room corner or follow the
            # boundary. The +45 heading points into a CCW room; containment in
            # Walker._extend still rejects variants that point outside locally.
            starts.extend(
                (
                    (length, i, midpoint, base),
                    (length, i, p, base + 45.0),
                    (length, i, midpoint, base + 45.0),
                )
            )
    if diverse_starts:
        order = len(starts)
        for centre in room_search.centres:
            point = (float(centre[0]), float(centre[1]))
            for base in room_search.base_headings(45):
                starts.append((0.0, order, point, base))
                order += 1
        unique = {}
        for item in starts:
            key = (round(item[2][0], 6), round(item[2][1], 6), round(item[3] % 360, 6))
            unique.setdefault(key, item)
        starts = list(unique.values())
    if max_starts is not None:
        # Long boundary edges are the most useful anchors in large rooms and
        # allow an extended pass to spend its budget on a representative subset.
        starts = sorted(starts, key=lambda item: (-item[0], item[1]))[:max_starts]

    seen = set()
    for width in widths:
        batch = []
        for _, _, start, base in starts:
            if time.process_time() > deadline:
                break
            for faces, turns in walker.run(start, base, width, deadline):
                key = (start, round(base, 6), faces, turns)
                if key not in seen:
                    seen.add(key)
                    batch.append((_make_shape(faces, turns), start, base))
        batch.sort(key=lambda t: -(weights.A * t[0].area + weights.C * t[0].perimeter))
        yield from batch
        if time.process_time() > deadline:
            return
