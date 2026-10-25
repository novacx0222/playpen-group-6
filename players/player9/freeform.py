"""Growing an enclosure that already fits into a free-form one.

A loop is a cycle of faces, each with a heading (a multiple of 45 degrees from the
loop's own heading) and a whole-number length. It closes exactly when the lengths
along each of the four axes cancel. One move keeps that true by construction:

    slide a run of consecutive corners by `d` units along one of the 8 directions

Where the run is cut from the rest, a face parallel to the slide just changes length,
and anywhere else a new face of length `d` appears. Pushing a side out, stepping a
long side (which is how a side gets past the 30-unit face limit), notching, and
cutting corners diagonally are all this one move. The start point and heading of the
loop carry over, so a grown loop needs no new search for where it goes.

`grow` does three things with it, in turn:

1. climb: take the best slide that raises the score, until there is none;
2. stretch and repair: a side stops growing at the 30-unit face limit, and the step
   that would let it continue costs perimeter before it pays; with odd pieces, every
   step between two buildable loops may need a length nothing in stock makes. So for
   a while a side that can't be built is allowed, at a charge for each unit it is off
   by, low at first: the loop spreads through the room on sides that are too long or
   the wrong length, and as the charge rises the cheapest fixes (a notch, a step, a
   unit more or less) are bought one by one until every side can be built;
3. kick: from each loop that is stuck, try the slides that add a face whatever they
   cost, climb from each, and keep what ends up better.

Only loops whose every side can be built are ever given pieces and shown to the
simulator's validator; the others exist only inside step 2.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import shapely

import src.constants as c

DIRECTIONS = tuple(
    (math.cos(math.radians(45 * k)), math.sin(math.radians(45 * k))) for k in range(8)
)
# how far a run may slide in one climbing move, where the inventory can make
# use of it: big steps first, and the small ones only when no big step helps
BIG_SLIDES = (5, 10, 20)
SMALL_SLIDES = (1, 2, 3)
SHARP = (3, 5)  # turns of 135 degrees, in 45-degree steps mod 8
NEAR_FACE = 0.75  # a room corner this close to a face marks a place to cut it
MAX_FACES = 80
# A run may be cut at any two corners when the loop has at most this many faces,
# and otherwise only at corners this many faces apart: a long run moving is the
# same as several short ones, and trying every pair costs too much.
ALL_CORNERS_UP_TO = 24
CORNER_REACH = 8
ASSIGN_TRIES = 40  # how many of a step's best fitting candidates get pieces assigned
KICKS = 8  # face-adding moves tried from each stuck loop
BEAM = 3
OVER_LIMIT_MAX = 150  # how long a side may get while the limit is only a charge
CHARGE_GROWTH = 1.7
CHARGE_STEPS = 13
STRETCHED = 3  # how many climbed loops go through stretch and repair
STRETCH_SHARE = 0.7  # of the time left after climbing; kicks get the rest
MODES = (False, True)  # stretch with only over-long sides charged, then any side
REPAIR_GRACE = 2.0  # CPU seconds a stretch cut short gets to make its loop buildable
# the best loop's first stretch gets the biggest slice of the time: a big room
# needs it, and what a small room leaves over passes to the next in line
SLICE_WEIGHTS = (4, 2, 2, 1, 1, 1)
EPS = 1e-6

Faces = tuple[tuple[int, int], ...]  # (heading index 0..7, length)


@dataclass(frozen=True)
class Loop:
    start: tuple[float, float]  # where face 0 begins, in the room
    heading: float  # degrees: the room direction of heading index 0
    faces: Faces
    score: float  # the true score, or the most it could score while `over` > 0
    over: int = 0  # units by which its faces miss lengths that can be built


def turn(before: int, after: int) -> int:
    """The signed turn from one heading index to another, in 45-degree steps."""
    return (after - before + 3) % 8 - 3


def _normalise(pos: tuple[float, float], faces: list) -> tuple[tuple, list]:
    """Merge neighbouring faces that lie on one line (same or opposite headings)
    into their net, so every joint left is a real corner. `pos` is where the first
    face starts and moves if that face is merged across the loop's seam."""
    out: list[tuple[int, int]] = []
    for h, length in faces:
        while out and (out[-1][0] - h) % 4 == 0:
            ph, pl = out.pop()
            if ph == h:
                length += pl
            elif pl > length:
                h, length = ph, pl - length
            else:
                length -= pl
            if length == 0:
                break
        if length:
            out.append((h, length))
    while len(out) >= 2 and (out[-1][0] - out[0][0]) % 4 == 0:
        h, length = out.pop()
        ux, uy = DIRECTIONS[h]
        pos = (pos[0] - length * ux, pos[1] - length * uy)
        fh, fl = out[0]
        if fh == h:
            out[0] = (h, fl + length)
        elif fl > length:
            out[0] = (fh, fl - length)
        elif fl < length:
            out[0] = (h, length - fl)
        else:
            out.pop(0)
    return pos, out


def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def corners(pos: tuple[float, float], faces) -> list[tuple[float, float]]:
    """The loop's corners in its own frame, the first at `pos`."""
    x, y = pos
    points = []
    for h, length in faces:
        points.append((x, y))
        x += length * DIRECTIONS[h][0]
        y += length * DIRECTIONS[h][1]
    return points


def area_of(points: list[tuple[float, float]]) -> float:
    total = 0.0
    px, py = points[-1]
    for x, y in points:
        total += px * y - x * py
        px, py = x, y
    return total / 2.0


class Grower:
    def __init__(
        self,
        region,  # the room, buffered by the containment tolerance and prepared
        room_corners,  # the room's corners, as an (n, 2) array
        buildable: set[int],  # face lengths the inventory can make
        rights: int,
        diagonals: int,
        total_length: int,
        score: Callable[[float, float], float],  # (area, perimeter) -> most it scores
        accept: Callable[[tuple[float, float], float, Faces], float | None],
        can_build: Callable[[tuple[int, ...]], bool],
        max_faces: int = MAX_FACES,
    ):
        self.region = region
        self.room_corners = room_corners
        self._remembered: dict[tuple, list] = {}
        self._remembered_for: tuple | None = None
        self.buildable = buildable
        self.rights = rights
        self.diagonals = diagonals
        self.total_length = total_length
        self.score = score
        # assigns pieces and has the simulator validate; returns the true score
        self.accept = accept
        # whether the inventory has pieces for these face lengths
        self.can_build = can_build
        self.max_faces = max_faces
        self.directions = range(8) if diagonals > 0 else range(0, 8, 2)
        self.smallest = tuple(sorted(buildable)[:2])
        self.longest = max(buildable)
        # A slide of d adds faces of length d or changes a face's length by d,
        # so only a d that is a buildable length, or the gap between two, can
        # lead anywhere: with nothing but 12-unit walls, sliding by 5 never does.
        lengths = sorted(buildable)
        useful = set(lengths) | {b - a for a in lengths for b in lengths if b > a}
        big = set(BIG_SLIDES) | set(self.smallest)
        self.exact_slides = tuple(
            step
            for step in (
                tuple(sorted(big & useful)),
                tuple(d for d in SMALL_SLIDES if d in useful),
            )
            if step
        )
        # while unbuildable sides are only charged for, any slide can lead somewhere
        self.loose_slides = (tuple(sorted(big)), SMALL_SLIDES)
        # how far each length is from one that can be built
        self.miss = [0] * (OVER_LIMIT_MAX + 1)
        for length in range(1, OVER_LIMIT_MAX + 1):
            if length > c.MAX_FACE_LENGTH:
                self.miss[length] = length - c.MAX_FACE_LENGTH
            elif length not in buildable:
                self.miss[length] = min(abs(length - b) for b in lengths)
        # two diagonal slides at once move a run straight sideways by a notch in
        # each cut: the way to break a long side in a room too narrow to step it
        self.notches = diagonals >= 4
        self.deadline = 0.0  # when the step in hand must stop
        self.end = 0.0  # when everything must
        self.charge = math.inf  # per unit a face is off by; inf forbids any
        # whether a charged face may be any length, or only longer than the limit
        self.any_length = False
        self.first_charge = 1.0

    # ----------------------------------------------------------------- moves

    def _cuts(self, loop: Loop, wide: bool) -> list[tuple[int, int]]:
        """Where a run may be cut, as (face, units along it): every corner, a few
        places inside each long face, and wherever a corner of the room sits
        against a face (the mouth of a corridor is where a bump can go)."""
        t = math.radians(loop.heading)
        cos, sin = math.cos(t), math.sin(t)
        x, y = 0.0, 0.0
        cuts = []
        for f, (h, length) in enumerate(loop.faces):
            if length > c.MAX_FACE_LENGTH:
                # where a step would leave a side of exactly the limit
                inner = set(range(c.MAX_FACE_LENGTH, length - 4, c.MAX_FACE_LENGTH))
                inner |= {self.smallest[0], self.longest, length // 2}
                inner |= {length - k for k in inner}
            elif length >= 10:
                # where both parts could be built: the shortest part at either
                # end, and the split nearest the middle
                splits = [
                    k
                    for k in range(5, length - 4)
                    if k in self.buildable and length - k in self.buildable
                ]
                inner = set()
                if splits:
                    inner = {splits[0], splits[-1]}
                    if wide or length < 15:
                        inner.add(min(splits, key=lambda k: abs(2 * k - length)))
                elif length >= 15:
                    inner = {5, length - 5}
            else:
                inner = set()
            ux, uy = DIRECTIONS[h]
            if length >= 10:
                wx = ux * cos - uy * sin
                wy = ux * sin + uy * cos
                dx = self.room_corners[:, 0] - (loop.start[0] + x * cos - y * sin)
                dy = self.room_corners[:, 1] - (loop.start[1] + x * sin + y * cos)
                along = dx * wx + dy * wy
                near = np.abs(dx * wy - dy * wx) <= NEAR_FACE
                for value in along[near & (along > 4.5) & (along < length - 4.5)][:4]:
                    inner.update((math.floor(value), math.ceil(value)))
            cuts.append((f, 0))
            cuts.extend((f, k) for k in sorted(inner) if 5 <= k <= length - 5)
            x += length * ux
            y += length * uy
        return cuts

    @staticmethod
    def _between(faces: Faces, first: tuple[int, int], last: tuple[int, int]) -> list:
        """The pieces of the loop walked forward from one cut to another."""
        (f1, o1), (f2, o2) = first, last
        if f1 == f2 and o1 < o2:
            return [(faces[f1][0], o2 - o1)]
        n = len(faces)
        out = [(faces[f1][0], faces[f1][1] - o1)]
        k = (f1 + 1) % n
        while k != f2:
            out.append(faces[k])
            k = (k + 1) % n
        if o2:
            out.append((faces[f2][0], o2))
        return out

    def _value(self, score: float, over: int) -> float:
        return score - self.charge * over if over else score

    def _moves(self, loop: Loop, slides: tuple[int, ...], kick: bool) -> list[tuple]:
        """Every loop one slide away that the stock allows, as (where it starts
        in the old frame, faces, score, units over). Remembered for the loop
        last asked about: raising the charge asks again about the same loop."""
        strict = self.charge == math.inf
        key = (loop.start, loop.faces, slides, kick, strict, self.any_length)
        if key in self._remembered:
            return self._remembered[key]
        if self._remembered_for != (loop.start, loop.faces):
            self._remembered_for = (loop.start, loop.faces)
            self._remembered = {}

        faces = loop.faces
        n = len(faces)
        cuts = self._cuts(loop, wide=kick)
        m = len(cuts)
        starts = corners((0.0, 0.0), faces)
        at = [
            (
                starts[f][0] + offset * DIRECTIONS[faces[f][0]][0],
                starts[f][1] + offset * DIRECTIONS[faces[f][0]][1],
            )
            for f, offset in cuts
        ]
        corner = [offset == 0 for _, offset in cuts]
        # Sliding the run from cut i to cut j by a vector v sweeps out a
        # parallelogram of area cross(at[j] - at[i], v), whatever the run's shape,
        # so a candidate's area needs no fresh walk round its outline.
        area = area_of(starts)
        far = n if n <= ALL_CORNERS_UP_TO else CORNER_REACH
        seen = set()
        found = []
        for i in range(m):
            crossed = 0  # corners passed on the way from cut i to cut j
            for span in range(1, m):
                j = (i + span) % m
                # a cut inside a face pairs only with cuts on that face or the
                # next (a bump, a notch); two corners may be further apart
                if crossed > (far if corner[i] and corner[j] else 1):
                    if crossed > far:
                        break
                    crossed += corner[j]
                    continue
                crossed += corner[j]
                run = self._between(faces, cuts[i], cuts[j])
                rest = self._between(faces, cuts[j], cuts[i])
                before, first = rest[-1][0], run[0][0]
                last, after = run[-1][0], rest[0][0]
                for g in self.directions:
                    back = (g + 4) % 8
                    # A turn of 135 degrees has no connector, so a new face that
                    # would make one is not worth building. But a new face lying
                    # along its other neighbour merges into it and makes no
                    # turn of its own: pulling a pocket back along the wall it
                    # hangs from is exactly that, and must not be skipped.
                    joins_first = (first - g) % 4 == 0
                    joins_before = (g - before) % 4 == 0
                    joins_after = (after - back) % 4 == 0
                    joins_last = (back - last) % 4 == 0
                    if (
                        ((g - before) % 8 in SHARP and not joins_first)
                        or ((first - g) % 8 in SHARP and not joins_before)
                        or ((back - last) % 8 in SHARP and not joins_after)
                        or ((after - back) % 8 in SHARP and not joins_last)
                    ):
                        continue
                    chord_x, chord_y = at[j][0] - at[i][0], at[j][1] - at[i][1]
                    ux, uy = DIRECTIONS[g]
                    sweep = chord_x * uy - chord_y * ux
                    for d in slides:
                        pos, new = _normalise(
                            at[j], rest + [(g, d)] + run + [(back, d)]
                        )
                        size = len(new)
                        if size < 4 or size > self.max_faces or (kick and size <= n):
                            continue
                        key2 = (round(pos[0], 5), round(pos[1], 5), tuple(new))
                        if key2 in seen:
                            continue
                        seen.add(key2)
                        worth = self._worth(new, area - d * sweep)
                        if worth is not None:
                            found.append((pos, key2[2], *worth))
                if not self.notches:
                    continue
                chord_x, chord_y = at[j][0] - at[i][0], at[j][1] - at[i][1]
                for g1 in (1, 3, 5, 7):
                    for g2 in ((g1 + 2) % 8, (g1 - 2) % 8):
                        if (g1 - before) % 8 in SHARP or (first - g2) % 8 in SHARP:
                            continue
                        b1, b2 = (g1 + 4) % 8, (g2 + 4) % 8
                        for t1, t2 in ((b1, b2), (b2, b1)):
                            if (t1 - last) % 8 in SHARP or (after - t2) % 8 in SHARP:
                                continue
                            vx = DIRECTIONS[g1][0] + DIRECTIONS[g2][0]
                            vy = DIRECTIONS[g1][1] + DIRECTIONS[g2][1]
                            sweep = chord_x * vy - chord_y * vx
                            # each end of the slide is two faces, not one: a
                            # small triangle of d * d / 2 on each, in or out
                            corner_area = (
                                _cross(DIRECTIONS[g1], DIRECTIONS[g2])
                                + _cross(DIRECTIONS[t1], DIRECTIONS[t2])
                            ) / 2.0
                            for d in self.smallest:
                                pos, new = _normalise(
                                    at[j],
                                    rest
                                    + [(g1, d), (g2, d)]
                                    + run
                                    + [(t1, d), (t2, d)],
                                )
                                size = len(new)
                                if size < 4 or size > self.max_faces:
                                    continue
                                key2 = (round(pos[0], 5), round(pos[1], 5), tuple(new))
                                if key2 in seen:
                                    continue
                                seen.add(key2)
                                worth = self._worth(
                                    new, area - d * sweep + d * d * corner_area
                                )
                                if worth is not None:
                                    found.append((pos, key2[2], *worth))
        self._remembered[key] = found
        return found

    def _candidates(self, loop: Loop, slides: tuple[int, ...], kick: bool) -> list:
        """The moves worth testing against the room, best first, as
        (value, where it starts in the old frame, faces, score, units over)."""
        current = self._value(loop.score, loop.over)
        found = []
        for pos, faces, score, over in self._moves(loop, slides, kick):
            value = self._value(score, over)
            if kick or value > current + EPS:
                found.append((value, pos, faces, score, over))
        found.sort(key=lambda f: -f[0])
        return found

    def _worth(self, faces, area: float) -> tuple[float, int] | None:
        """The most this loop could score and how far its faces are from
        lengths that can be built, or None if the stock rules it out."""
        rights = diagonals = perimeter = over = 0
        previous = faces[-1][0]
        strict = self.charge == math.inf
        for h, length in faces:
            if length > OVER_LIMIT_MAX:
                return None
            miss = self.miss[length]
            if miss:
                if strict or not (self.any_length or length > c.MAX_FACE_LENGTH):
                    return None
                over += miss
            bend = abs(turn(previous, h))
            if bend == 2:
                rights += 1
            elif bend == 1:
                diagonals += 1
            else:
                return None
            perimeter += length
            previous = h
        if rights > self.rights or diagonals > self.diagonals:
            return None
        if perimeter > self.total_length:
            return None
        if area <= 0:
            return None
        return self.score(area, perimeter), over

    def _fitting(self, loop: Loop, candidates: list[tuple]) -> list[tuple]:
        """The candidates that are simple polygons inside the room, as
        (value, start in the room, faces, score, units over), best first."""
        t = math.radians(loop.heading)
        cos, sin = math.cos(t), math.sin(t)
        sx, sy = loop.start
        by_size: dict[int, list[int]] = {}
        rings = []
        starts = []
        for index, (_, pos, faces, _, _) in enumerate(candidates):
            local = np.array(corners(pos, faces))
            world = np.column_stack(
                [
                    sx + local[:, 0] * cos - local[:, 1] * sin,
                    sy + local[:, 0] * sin + local[:, 1] * cos,
                ]
            )
            rings.append(np.vstack([world, world[:1]]))
            starts.append((float(world[0, 0]), float(world[0, 1])))
            by_size.setdefault(len(local), []).append(index)
        ok = np.zeros(len(candidates), dtype=bool)
        for indices in by_size.values():
            polygons = shapely.polygons(np.stack([rings[i] for i in indices]))
            ok[indices] = shapely.is_valid(polygons) & shapely.contains(
                self.region, polygons
            )
        return [
            (candidates[i][0], starts[i], *candidates[i][2:])
            for i in range(len(candidates))
            if ok[i]
        ]

    # ---------------------------------------------------------------- search

    def _out_of_time(self) -> bool:
        return time.process_time() > self.deadline

    def climb(self, loop: Loop) -> Loop:
        """Take the best slide that raises the loop's value, until there is none."""
        while not self._out_of_time():
            better = None
            loose = self.any_length and self.charge != math.inf
            for slides in self.loose_slides if loose else self.exact_slides:
                better = self._step(loop, slides)
                if better is not None:
                    break
            if better is None:
                return loop
            loop = better
        return loop

    def _step(self, loop: Loop, slides: tuple[int, ...]) -> Loop | None:
        current = self._value(loop.score, loop.over)
        fitting = self._fitting(loop, self._candidates(loop, slides, kick=False))
        for _, start, faces, score, over in fitting[:ASSIGN_TRIES]:
            if over == 0:
                score = self.accept(start, loop.heading, faces)
                if score is None or score <= current + EPS:
                    continue
            elif not self.can_build(tuple(n for _, n in faces if not self.miss[n])):
                continue  # even the sides that could be built, can't all be
            return Loop(start, loop.heading, faces, score, over)
        return None

    def _stretch(self, loop: Loop, any_length: bool) -> Loop | None:
        """Allow sides that can't be built for a charge, then raise the charge
        until every side can be. `any_length` says whether that means any
        side, or only one over the face limit."""
        self.any_length = any_length
        charge = self.first_charge
        for _ in range(CHARGE_STEPS):
            if self._out_of_time():
                break
            self.charge = charge
            loop = self.climb(loop)
            charge *= CHARGE_GROWTH
        if loop.over:
            # out of time part-way: one last climb at the full charge, so the
            # spreading done so far is repaired and not thrown away
            self.charge = self.first_charge * CHARGE_GROWTH**CHARGE_STEPS
            self.deadline = min(self.end, time.process_time() + REPAIR_GRACE)
            loop = self.climb(loop)
        self.charge = math.inf
        self.any_length = False
        return self.climb(loop) if loop.over == 0 else None

    def _kicks(self, loop: Loop) -> list[Loop]:
        """Loops one face-adding slide away, whatever that does to the score."""
        kicked = []
        for _, start, faces, _, _ in self._fitting(
            loop, self._candidates(loop, self.smallest, kick=True)
        ):
            score = self.accept(start, loop.heading, faces)
            if score is not None:
                kicked.append(Loop(start, loop.heading, faces, score))
                if len(kicked) >= KICKS:
                    break
            if self._out_of_time():
                break
        return kicked

    def grow(self, seeds: list[Loop], deadline: float, first_charge: float) -> None:
        """Rounds of stretch-and-repair then kicks, each starting from the best
        loops of the round before, for as long as a round still finds better."""
        self.deadline = self.end = deadline
        self.first_charge = first_charge
        pool = self._distinct([self.climb(seed) for seed in seeds])
        best_before = -math.inf
        while pool and not self._out_of_time():
            # each loop gets an equal slice of the time, so one big open area
            # can't use up what the others were due
            now = time.process_time()
            stretch_end = now + (deadline - now) * STRETCH_SHARE
            # both ways for each: they take different routes, and neither is
            # better everywhere
            chosen = [(loop, mode) for mode in MODES for loop in pool[:STRETCHED]]
            weights = SLICE_WEIGHTS[: len(chosen)]
            for k, (loop, any_length) in enumerate(chosen):
                now = time.process_time()
                share = weights[k] / sum(weights[k:])
                self.deadline = now + max(0.0, stretch_end - now) * share
                stretched = self._stretch(loop, any_length)
                if stretched is not None:
                    pool.append(stretched)
            self.deadline = deadline
            pool = self._distinct(pool)

            beam = pool
            while beam and not self._out_of_time():
                grown = []
                for loop in beam:
                    for kicked in self._kicks(loop):
                        if self._out_of_time():
                            break
                        result = self.climb(kicked)
                        if result.score > loop.score + EPS:
                            grown.append(result)
                beam = self._distinct(grown)
                pool = self._distinct(pool + beam)

            if not pool or pool[0].score <= best_before + EPS:
                break
            best_before = pool[0].score

    @staticmethod
    def _distinct(loops: list[Loop]) -> list[Loop]:
        best: dict[tuple, Loop] = {}
        for loop in sorted(loops, key=lambda item: -item.score):
            key = (round(loop.start[0], 3), round(loop.start[1], 3), loop.faces)
            best.setdefault(key, loop)
        return list(best.values())[:BEAM]
