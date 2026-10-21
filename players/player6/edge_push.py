"""Edge-push compaction: slide a straight face outward until the room stops it.

A face whose two neighbours are perpendicular can slide along them. The face
keeps its length; the neighbours grow (or shrink) by the same amount, so the
loop still closes and no new faces appear. Only the two neighbours need new
pieces, taken from the spare inventory plus the pieces they release.

Example: a 20x20 square in a 23x23 room. Pushing the east face out by 3 turns
its north and south faces from 20 into 23 long.

Every accepted move is simulator-valid and strictly raises the real score.
"""

import math
import time
from dataclasses import replace

from shapely import Polygon
from shapely.prepared import prep

from players.player6.diagonal_search import DIRS
from players.player6.local_search import (
    Local45Search,
    LocalSearchStats,
    exact_vertices,
)
from src.constants import TOL
from src.enclosure import score_construction, validate_construction
from src.pieces import Piece, PieceType

MAX_PIECES = 6


def _signed_area(points):
    return 0.5 * sum(
        x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )


def compose(length, pool, straights, wanted_gates):
    """Pieces summing to `length` from `pool` ({(is_gate, len): count}).

    Prefers keeping `wanted_gates` gate pieces, then fewer pieces. Returns a
    list of (is_gate, len) or None."""
    keys = sorted(pool, key=lambda k: (-k[1], not k[0]))
    limit = min(MAX_PIECES, straights + 1)
    best = [None, None]

    def rank(chosen):
        gates = sum(1 for g, _ in chosen if g)
        return (-min(gates, wanted_gates), len(chosen))

    def rec(i, remaining, chosen):
        if remaining == 0:
            if chosen and (best[0] is None or rank(chosen) < best[1]):
                best[0], best[1] = chosen, rank(chosen)
            return
        if i == len(keys):
            return
        key = keys[i]
        most = min(pool[key], remaining // key[1], limit - len(chosen))
        for used in range(most, -1, -1):
            rec(i + 1, remaining - used * key[1], chosen + [key] * used)

    rec(0, length, [])
    return best[0]


class EdgePush:
    MAX_PUSH = 29

    def __init__(self, room, inventory, weights, *, time_budget=5.0, stop=None):
        self.room = room
        self.inventory = inventory
        self.weights = weights
        self.deadline = time.process_time() + max(0.0, time_budget)
        self.external_stop = stop
        self.stats = LocalSearchStats()

    def stopped(self):
        return time.process_time() >= self.deadline or (
            self.external_stop is not None and self.external_stop()
        )

    def improve(self, seed):
        started = time.process_time()
        result = validate_construction(seed, self.room, self.inventory)
        if not result.valid:
            return None
        best, best_score = seed, score_construction(result, self.weights)
        prepared_room = prep(self.room.polygon.buffer(TOL))
        improved = False
        start = 0
        while not self.stopped():
            found = self._sweep(best, result, best_score, prepared_room, start)
            if found is None:
                break
            candidate, resume, _scanned = found
            if candidate is None:
                break
            best, result, best_score = candidate
            improved = True
            self.stats.accepted += 1
            start = resume
        self.stats.cpu_seconds = time.process_time() - started
        return best if improved else None

    def _sweep(self, seed, result, best_score, prepared_room, start=0):
        """Scan faces cyclically from `start`; return ((construction, result,
        score), next face to scan, faces scanned) for the first improving push,
        (None, ...) if a full pass finds none, or None if out of time."""
        faces, origin, angle = Local45Search.boundary(seed, result)
        n = len(faces)
        if n < 4 or any(f.heading % 2 for f in faces):
            return None, 0, 0  # rectilinear boundaries only
        vertices = exact_vertices(faces)
        if vertices is None:
            return None, 0, 0
        ccw = _signed_area(vertices) > 0
        walls, gates, straights = Local45Search.spare_inventory(seed, self.inventory)
        pool = {}
        for length, count in walls.items():
            pool[(False, length)] = count
        for length, count in gates.items():
            pool[(True, length)] = count
        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        for step in range(n):
            i = (start + step) % n
            if self.stopped():
                return None
            p, nx = (i - 1) % n, (i + 1) % n
            if (faces[i].heading - faces[p].heading) % 8 not in (2, 6):
                continue
            if (faces[nx].heading - faces[i].heading) % 8 not in (2, 6):
                continue
            di, dp, dn = (
                (DIRS[faces[k].heading][0], DIRS[faces[k].heading][2])
                for k in (i, p, nx)
            )
            normal = (di[1], -di[0]) if ccw else (-di[1], di[0])
            grow_prev = normal[0] * dp[0] + normal[1] * dp[1]
            grow_next = -(normal[0] * dn[0] + normal[1] * dn[1])
            gain_per = self.weights.A * faces[i].length + self.weights.C * (
                grow_prev + grow_next
            )
            if gain_per <= 0:
                continue
            for delta in range(1, self.MAX_PUSH + 1):
                new_prev = faces[p].length + delta * grow_prev
                new_next = faces[nx].length + delta * grow_next
                if not (5 <= new_prev <= 30 and 5 <= new_next <= 30):
                    break
                moved = list(vertices)
                moved[i] = (
                    vertices[i][0] + delta * normal[0],
                    vertices[i][1] + delta * normal[1],
                )
                j = (i + 1) % n
                moved[j] = (
                    vertices[j][0] + delta * normal[0],
                    vertices[j][1] + delta * normal[1],
                )
                shift = (
                    (delta * normal[0], delta * normal[1])
                    if i in (0, n - 1)
                    else (0, 0)
                )
                poly = Polygon(
                    [
                        (
                            origin[0] + (x - 0) * cos_a - (y - 0) * sin_a,
                            origin[1] + (x - 0) * sin_a + (y - 0) * cos_a,
                        )
                        for x, y in moved
                    ]
                )
                if not poly.is_valid or not prepared_room.covers(poly):
                    break
                self.stats.attempted += 1
                released = {}
                extra_straights = 0
                for idx in (p, nx):
                    for piece in faces[idx].pieces:
                        key = (piece.is_gate, piece.length)
                        released[key] = released.get(key, 0) + 1
                    extra_straights += len(faces[idx].pieces) - 1
                work = dict(pool)
                for key, count in released.items():
                    work[key] = work.get(key, 0) + count
                work = {k: v for k, v in work.items() if v > 0}
                avail = straights + extra_straights
                new_pieces = {}
                ok = True
                for idx, target in ((p, new_prev), (nx, new_next)):
                    wanted = sum(1 for pc in faces[idx].pieces if pc.is_gate)
                    comp = compose(target, work, avail, wanted)
                    if comp is None:
                        ok = False
                        break
                    for key in comp:
                        work[key] -= 1
                    avail -= len(comp) - 1
                    new_pieces[idx] = tuple(
                        Piece(
                            PieceType.GATE if g else PieceType.WALL,
                            length,
                        )
                        for g, length in comp
                    )
                if not ok:
                    continue
                self.stats.feasible += 1
                new_faces = list(faces)
                new_faces[p] = replace(faces[p], length=new_prev, pieces=new_pieces[p])
                new_faces[nx] = replace(
                    faces[nx], length=new_next, pieces=new_pieces[nx]
                )
                new_origin = (
                    origin[0] + shift[0] * cos_a - shift[1] * sin_a,
                    origin[1] + shift[0] * sin_a + shift[1] * cos_a,
                )
                candidate = Local45Search.to_construction(new_faces, new_origin, angle)
                check = validate_construction(candidate, self.room, self.inventory)
                if not check.valid:
                    continue
                score = score_construction(check, self.weights)
                if score > best_score + 1e-9:
                    return (candidate, check, score), (i + 1) % n, step + 1
        return None, 0, n
