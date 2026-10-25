"""Player 9 — build the biggest, cheapest enclosure that fits in the room.

Score = area (good) - perimeter (bad) + a bonus for a 2nd gate on another face.
No single trick wins every room, so we run a PORTFOLIO of shape builders and
keep whichever result scores best by the real scorer. Each builder is gated on
true score, so a bad attempt is simply ignored -- adding one can only help.

Builders, in the order build_enclosure runs them (each time-sliced):
  1. grid-fill      - fill the largest block of grid cells (best on spirals /
                      cut-outs); cheap, so it runs first for a strong baseline.
  2. free-form grow - slide a loop's corners outward to swallow space (the
                      strongest all-rounder; gets the biggest time slice).
  3. best-rectangle - exhaustive scan of axis-aligned rectangles (safety net
                      for tight rooms where the growers get stuck).
  4. notched-band   - long band with 45-deg V-notches for thin corridors (trades
                      scarce straight connectors for abundant diagonals).
  5. chain-shape    - ROTATED rectilinear shapes built piece-by-piece, for rooms
                      whose best fit is tilted to their own axis (small rooms).
  6. templates / boundary-hug / maximal-shape / legacy library - more fallbacks.
  7. free-form polish - re-grow from the best-so-far to grab any last space.

Everything is capped by a shared CPU deadline and returns the best valid result
found so far. Some ideas were inspired by other groups' write-ups (group 2's
side allocation, group 10's edge hugging) and a teammate's wall-alignment idea.
"""

import math
import time
from functools import lru_cache
from itertools import product

from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from players.player0 import Player
from src.constants import BASELINE_SCORE, MAX_FACE_LENGTH, MIN_WALL_LENGTH, TOL
from src.enclosure import (
    Construction,
    score_construction,
    validate_construction,
    walk_points,
)
from src.pieces import Connector, ConnectorType, Piece, PieceType

HEADINGS = list(range(0, 180, 15))
GRID_STEPS = 8
MAX_TRY = 400
# stay well under the simulator's 300s CPU limit; the search returns the best
# found so far when the budget runs out, so this is a soft cap on effort.
SEARCH_CPU_SECONDS = 230.0
ALLOCATION_NODE_LIMIT = 20000


class Player9(Player):
    # ======================================================================
    # Entry point: run every builder, keep the highest true-scoring result.
    # ======================================================================
    def build_enclosure(self) -> Construction | None:
        deadline = time.process_time() + SEARCH_CPU_SECONDS
        best = None
        best_score = 0.0  # a no-build (0) beats any negative-scoring loop
        straight = self.inventory.connectors.get(ConnectorType.STRAIGHT, 0)
        right = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        diagonal = self.inventory.connectors.get(ConnectorType.DIAGONAL, 0)

        # ---- grid / rasterization FIRST: it is cheap (sub-second to a few
        # seconds even on big rooms) and is the big winner on non-convex rooms
        # (spirals, crowns, cut-out rooms). running it up front means the
        # slower template / hug / maximal searches never starve it of time, and
        # they then only need to beat an already-strong baseline. ----
        if time.process_time() < deadline:
            grid = self._grid_fill(min(deadline, time.process_time() + 60.0))
            if grid is not None:
                gs = self._score_cons(grid)
                if gs > best_score:
                    best, best_score = grid, gs
                    self.note = f"(a) grid-fill score {gs:.1f}"

        # ---- free-form grow SECOND, with a large guaranteed slice: it is the
        # biggest winner on corridors, cut-outs and odd concave rooms, slides a
        # loop's corners outward (including diagonal steps) to fill space the
        # fixed shapes and the grid leave behind. it seeds from the grid-fill
        # block, so it can only extend that result. given most of the budget so
        # the later template/hug searches can't starve it. competes on true
        # score -> never regresses. ----
        if time.process_time() < deadline:
            ff_deadline = min(deadline, time.process_time() + 110.0)
            ff = self._freeform_grow(ff_deadline)
            if ff is not None:
                fs = self._score_cons(ff)
                if fs > best_score:
                    best, best_score = ff, fs
                    self.note = f"(a) free-form score {fs:.1f}"

        # ---- best fitting rectangle: a cheap exhaustive scan that guarantees
        # we match a simple-rectangle opponent on rooms where the growers get
        # trapped (tight / spiral rooms). competes on true score. ----
        if time.process_time() < deadline:
            rect = self._best_rectangle(min(deadline, time.process_time() + 30.0))
            if rect is not None:
                rs = self._score_cons(rect)
                if rs > best_score:
                    best, best_score = rect, rs
                    self.note = f"(a) best-rectangle score {rs:.1f}"

        # ---- notched band: long full-height band with 45-degree V-notches,
        # for thin/corridor rooms where a plain rectangle is capped by scarce
        # STRAIGHT connectors. Trades straights for abundant diagonals to fit a
        # longer band. Competes on true score. ----
        if time.process_time() < deadline:
            band = self._notched_band(min(deadline, time.process_time() + 30.0))
            if band is not None:
                bs = self._score_cons(band)
                if bs > best_score:
                    best, best_score = band, bs
                    self.note = f"(a) notched-band score {bs:.1f}"

        # ---- chain-shape: ROTATED rectilinear templates (rect/plus/pinwheel)
        # built by piece-chaining, for rooms whose best shape is rotated to the
        # room's own symmetry axis (star, pinwheel rooms). Only worthwhile on
        # small rooms -- on large rooms the grid/freeform dominate and this would
        # just burn time -- so gate on room span. Competes on true score. ----
        if time.process_time() < deadline:
            rminx, rminy, rmaxx, rmaxy = self.room.polygon.bounds
            rspan = max(rmaxx - rminx, rmaxy - rminy)
            if rspan <= 60:
                chain = self._chain_shape(min(deadline, time.process_time() + 45.0))
                if chain is not None:
                    cs = self._score_cons(chain)
                    if cs > best_score:
                        best, best_score = chain, cs
                        self.note = f"(a) chain-shape score {cs:.1f}"

        # ---- piece stock as parallel (types, counts) lists ----
        types: list[Piece] = []
        stock: list[int] = []
        for kind, pool in (
            (PieceType.WALL, self.inventory.walls),
            (PieceType.GATE, self.inventory.gates),
        ):
            for length, count in sorted(pool.items()):
                # a piece must be between the min and max buildable length;
                # inventories sometimes list length-4 stock we simply can't use.
                if count > 0 and MIN_WALL_LENGTH <= length <= MAX_FACE_LENGTH:
                    types.append(Piece(kind, length))
                    stock.append(count)
        if not any(p.is_gate for p in types):
            self.note = "No usable gate (length <= 30) -- no valid enclosure."
            return None

        # ---- enumerate faces: tiled runs of pieces (<= straight+1 pieces) ----
        # faces[length] = list of (usage_tuple, joins, has_gate)
        faces: dict[int, list] = {}
        counts = [0] * len(types)

        def enumerate_faces(first, length, number, has_gate):
            if time.process_time() >= deadline:
                return
            if number:
                faces.setdefault(length, []).append(
                    (tuple(counts), number - 1, has_gate)
                )
            if number >= min(6, straight + 1):
                return
            for i in range(first, len(types)):
                nl = length + types[i].length
                if nl > MAX_FACE_LENGTH or counts[i] >= stock[i]:
                    continue
                counts[i] += 1
                enumerate_faces(i, nl, number + 1, has_gate or types[i].is_gate)
                counts[i] -= 1

        enumerate_faces(0, 0, 0, False)
        if not faces:
            self.note = "No face can be built from the inventory."
            return None
        # prefer fewer joins (fewer straight connectors) and gate-bearing faces
        for options in faces.values():
            options.sort(key=lambda o: (o[1], not o[2]))

        # ---- work out how to build each side from the pieces we have ----
        def allocate(lengths, required_gate_faces):
            nodes = 0

            @lru_cache(maxsize=ALLOCATION_NODE_LIMIT)
            def search(side, remaining, avail_straight, gate_faces):
                nonlocal nodes
                nodes += 1
                if nodes > ALLOCATION_NODE_LIMIT or time.process_time() >= deadline:
                    return None
                if side == len(lengths):
                    return () if gate_faces >= required_gate_faces else None
                if gate_faces + len(lengths) - side < required_gate_faces:
                    return None
                for usage, joins, has_gate in faces.get(lengths[side], ()):
                    if joins > avail_straight:
                        continue
                    if any(u > r for u, r in zip(usage, remaining)):
                        continue
                    rest = search(
                        side + 1,
                        tuple(r - u for u, r in zip(usage, remaining)),
                        avail_straight - joins,
                        min(required_gate_faces, gate_faces + int(has_gate)),
                    )
                    if rest is not None:
                        return (usage,) + rest
                return None

            return search(0, tuple(stock), straight, 0)

        # ---- room geometry + placement anchors ----
        room = self.room.polygon
        buffered = prep(room.buffer(TOL))
        corners = list(room.exterior.coords)[:-1]
        minx, miny, maxx, maxy = room.bounds
        room_diag = math.hypot(maxx - minx, maxy - miny)

        headings = set(range(0, 180, 15))
        for p, q in zip(corners, corners[1:] + corners[:1]):
            ang = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
            headings.add(round(ang % 180, 9))
            headings.add(round((ang + 90) % 180, 9))

        centers = [(room.centroid.x, room.centroid.y)]
        rp = room.representative_point()
        centers.append((rp.x, rp.y))
        for i in range(1, 10):
            for j in range(1, 10):
                centers.append(
                    (minx + (maxx - minx) * i / 10, miny + (maxy - miny) * j / 10)
                )

        def placement(vertices):
            local = Polygon(vertices)
            center = local.centroid
            angles = set(range(0, 360, 15))
            for h in headings:
                for off in range(0, 360, 45):
                    angles.add(round((h + off) % 360, 9))
            for h in sorted(angles):
                if time.process_time() >= deadline:
                    return None
                th = math.radians(h)
                co, si = math.cos(th), math.sin(th)
                offs = [(x * co - y * si, x * si + y * co) for x, y in vertices]
                cx = center.x * co - center.y * si
                cy = center.x * si + center.y * co
                # corner-start placements first (notes: corner starts give room)
                starts = [(px - ox, py - oy) for px, py in corners for ox, oy in offs]
                starts.extend((x - cx, y - cy) for x, y in centers)
                for x, y in starts:
                    if time.process_time() >= deadline:
                        return None
                    poly = Polygon([(x + dx, y + dy) for dx, dy in offs])
                    if buffered.contains(poly):
                        return (x, y), h
            return None

        # ---- generate template families, round-robin, ranked by true score ----
        lengths_avail = sorted(faces)

        def templates():
            gens = []
            if right >= 4:
                gens.append(
                    ("rectangle", (a, b, a, b))
                    for a, b in product(lengths_avail, repeat=2)
                    if a <= b
                )
            if right >= 2 and diagonal >= 4:
                gens.append(
                    ("hexagon", (a, b, c, a, b, c))
                    for a, b, c in product(lengths_avail, repeat=3)
                )
            if diagonal >= 8:
                gens.append(
                    ("octagon", (a, b, c, d, a, b, c, d))
                    for a, b, c, d in product(lengths_avail, repeat=4)
                    if a <= c
                )
            if right >= 6:
                gens.append(
                    ("L-shape", (a + b, c, b, d, a, c + d))
                    for a, b, c, d in product(lengths_avail, repeat=4)
                    if a + b in faces and c + d in faces
                )
            gens = [g for g in gens]
            while gens:
                for g in gens[:]:
                    try:
                        yield next(g)
                    except StopIteration:
                        gens.remove(g)

        available_perimeter = sum(p.length * n for p, n in zip(types, stock))
        rw, rh = maxx - minx, maxy - miny
        room_bb_lo, room_bb_hi = min(rw, rh), max(rw, rh)
        candidates = []
        gen_deadline = time.process_time() + max(
            0.5, (deadline - time.process_time()) * 0.2
        )
        for family, sides in templates():
            if time.process_time() >= gen_deadline:
                break
            if sum(sides) > available_perimeter:
                continue
            area = self._template_area(family, sides)
            if area > room.area + TOL * room.length:
                continue
            # skip shapes too big to ever fit the room (even rotated), so we
            # don't waste time trying to place giant octagons that can't fit.
            verts0 = self._integer_vertices(sides, self._template_turns(family))
            if verts0 is None:
                continue
            xs = [v[0] for v in verts0]
            ys = [v[1] for v in verts0]
            sw, sh = max(xs) - min(xs), max(ys) - min(ys)
            s_lo, s_hi = min(sw, sh), max(sw, sh)
            if s_lo > room_bb_lo + TOL or s_hi > room_bb_hi + TOL:
                continue
            base = BASELINE_SCORE + self.weights.A * area + self.weights.C * sum(sides)
            if base + self.weights.G > 0:
                candidates.append((base + self.weights.G, base, family, sides))
        candidates.sort(reverse=True)

        tried = 0
        for upper, base, family, sides in candidates:
            if time.process_time() >= deadline or upper <= best_score:
                break
            tried += 1
            turns = self._template_turns(family)
            vertices = self._integer_vertices(sides, turns)
            if vertices is None:
                continue
            shape = Polygon(vertices)
            if not shape.is_valid or shape.area <= 0:
                continue
            if any(
                math.hypot(x - xx, y - yy) > room_diag + TOL
                for x, y in vertices
                for xx, yy in vertices
            ):
                continue
            assemblies = None
            if self.weights.G > 0:
                assemblies = allocate(sides, 2)
            if assemblies is None and base > best_score:
                assemblies = allocate(sides, 1)
            if assemblies is None:
                continue
            bonus = (
                sum(
                    any(types[i].is_gate and n for i, n in enumerate(face))
                    for face in assemblies
                )
                >= 2
            )
            if base + (self.weights.G if bonus else 0) <= best_score:
                continue
            found = placement(vertices)
            if found is None:
                continue
            pieces, conns = [], []
            for usage, turn in zip(assemblies, turns):
                face = [p for p, n in zip(types, usage) for _ in range(n)]
                pieces.extend(face)
                conns.append(
                    Connector(
                        ConnectorType.RIGHT
                        if abs(turn) == 90
                        else ConnectorType.DIAGONAL,
                        reflex=turn < 0,
                    )
                )
                conns.extend(Connector(ConnectorType.STRAIGHT) for _ in face[1:])
            start, heading = found
            cons = Construction(start, heading, pieces, conns)
            res = validate_construction(cons, self.room, self.inventory)
            if res.valid:
                sc = score_construction(res, self.weights)
                if sc > best_score:
                    best, best_score = cons, sc
                    self.note = f"(a) {family} score {sc:.1f}"

        # ---- boundary hug: inset the room and trace its actual outline. this
        # follows arbitrary non-convex rooms (the room's own shape is the most
        # area-efficient enclosure), so it is usually the big winner. ----
        hug = self._boundary_hug()
        if hug is not None:
            hs = self._score_cons(hug)
            if hs > best_score:
                best, best_score = hug, hs
                self.note = f"(a) boundary-hug score {hs:.1f}"

        # ---- maximal fitting shape: grow the biggest rectangle / chamfered
        # rectangle / gabled shape that fits inside the room. helps rooms where
        # the template search settles for a shape smaller than what fits. we
        # skip it when we already cover most of the room (it can only add
        # perimeter then) to save time on big rooms the hug already fills. ----
        covered = 0.0
        if best is not None:
            r = validate_construction(best, self.room, self.inventory)
            if r.valid:
                covered = r.area / room.area
        if covered < 0.90 and time.process_time() < deadline:
            mx_deadline = min(deadline, time.process_time() + 120.0)
            mx = self._maximal_shape(mx_deadline)
        else:
            mx = None
        if mx is not None:
            ms = self._score_cons(mx)
            if ms > best_score:
                best, best_score = mx, ms
                self.note = f"(a) maximal-shape score {ms:.1f}"

        # ---- extra family: trace the room outline (helps concave rooms) ----
        traced = self._room_traced()
        if traced is not None:
            ts = self._score_cons(traced)
            if ts > best_score:
                best, best_score = traced, ts
                self.note = f"(a) room-traced score {ts:.1f}"

        # ---- safety net: the proven template library + wall-aligned placement.
        # covers rooms the parameterized search misses (odd concave shapes). it
        # competes on true score, so it can only raise the result, never lower.
        legacy = self._legacy_search(best_score)
        if legacy is not None:
            lcons, lscore = legacy
            if lscore > best_score:
                best, best_score = lcons, lscore
                self.note = f"(b) legacy shape score {lscore:.1f}"

        # ---- final polish: re-run the free-form grower seeded from the best
        # result so far, so it can slide that shape's corners out into any room
        # the fixed-shape searches left unused. only if time remains. ----
        if best is not None and time.process_time() < deadline:
            ff2 = self._freeform_grow(deadline, seed_cons=best)
            if ff2 is not None:
                f2 = self._score_cons(ff2)
                if f2 > best_score:
                    best, best_score = ff2, f2
                    self.note = f"(a) free-form polish score {f2:.1f}"

        if best is None:
            self.note = "No valid positive-scoring enclosure found."
        return best

    # ======================================================================
    # Direct shape builders (scan a shape family, build + score, keep best).
    # ======================================================================
    def _best_rectangle(self, deadline) -> Construction | None:
        """Scan every axis-aligned rectangle that fits and is buildable, keep
        the highest-scoring one. Our safety net for tight rooms where the
        growers get stuck (e.g. a spiral room where only a small box fits)."""
        from shapely.geometry import Polygon as _P
        from shapely.prepared import prep as _prep

        room = self.room.polygon
        pr = _prep(room.buffer(TOL))
        minx, miny, maxx, maxy = (int(math.floor(b)) for b in room.bounds)
        span = max(maxx - minx, maxy - miny)
        step = 1 if span <= 60 else max(1, span // 60)

        best = None
        best_score = float("-inf")
        # rank candidate (w, h) by the score they'd give if placed, best first,
        # so we can stop early once no bigger box can beat the incumbent.
        sizes = []
        for w in range(5, MAX_FACE_LENGTH + 1):
            for h in range(w, MAX_FACE_LENGTH + 1):
                upper = self._score_poly(w * h, 2 * (w + h), True)
                if upper > 0:
                    sizes.append((upper, w, h))
        sizes.sort(reverse=True)

        for upper, w, h in sizes:
            if time.process_time() >= deadline or upper <= best_score:
                break
            for ow, oh in ((w, h), (h, w)):
                placed = False
                for ox in range(minx, maxx - ow + 1, step):
                    for oy in range(miny, maxy - oh + 1, step):
                        rect = _P(
                            [(ox, oy), (ox + ow, oy), (ox + ow, oy + oh), (ox, oy + oh)]
                        )
                        if not pr.contains(rect):
                            continue
                        built = self._trace_polygon(rect)
                        if built is None:
                            continue
                        cons = Construction(*built)
                        sc = self._score_cons(cons)
                        if sc > best_score:
                            best, best_score = cons, sc
                        placed = True
                        break  # one placement per size is enough
                    if placed:
                        break
        return best

    def _notched_band(self, deadline) -> Construction | None:
        """For thin/long rooms: a full-height band along the long axis with
        periodic 45-deg V-notches on both edges. A plain rectangle in a thin
        room runs out of STRAIGHT connectors; each V-notch replaces a straight
        join with two DIAGONAL corners (usually far more plentiful), so a much
        longer band fits -- more area. A notch of leg `g` dips the edge in by
        g/sqrt(2); both legs are buildable diagonal walls."""
        from shapely.geometry import Polygon as _P
        from shapely.prepared import prep as _prep

        sq = math.sqrt(2)
        have_diag = self.inventory.connectors.get(ConnectorType.DIAGONAL, 0)
        have_right = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        if have_diag < 4 or have_right < 4:
            return None

        room = self.room.polygon
        minx, miny, maxx, maxy = room.bounds
        rw, rh = maxx - minx, maxy - miny
        # only worthwhile on clearly elongated rooms
        if min(rw, rh) < 5 or max(rw, rh) / max(min(rw, rh), 1e-9) < 2.0:
            return None

        # a length is usable for a single wall PIECE only if it can be tiled
        # from inventory (a leg must be one buildable wall; segments/caps may be
        # multi-piece runs). cache buildability of integer lengths.
        def buildable_run(L):
            return (
                self._tile_run(
                    L, dict(self.inventory.walls), dict(self.inventory.gates), False
                )
                is not None
            )

        # leg must be a SINGLE diagonal wall whose length exists in stock
        leg_lengths = sorted(
            L
            for L, c in self.inventory.walls.items()
            if c > 0 and 5 <= L <= MAX_FACE_LENGTH
        )
        if not leg_lengths:
            return None

        pr = _prep(room.buffer(TOL))
        best = None
        best_score = float("-inf")

        horizontal = rw >= rh
        axis_len = rw if horizontal else rh
        cross = rh if horizontal else rw

        def to_world(u, v):
            if horizontal:
                return (minx + u, miny + v)
            return (minx + v, miny + u)

        seg_choices = [
            s for s in (6, 10, 12, 15, 18, 20, 24, 25, 30) if buildable_run(s)
        ]
        if not seg_choices:
            return None

        for inset in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            if time.process_time() >= deadline:
                break
            H = cross - 2 * inset
            Hi = int(round(H))
            if Hi < 5 or abs(H - Hi) > 0.05 or not buildable_run(Hi):
                continue
            for g in leg_lengths:
                depth = g / sq
                # Both long edges are notched (mirrored) so the loop closes with
                # only integer-length pieces -- a single flat edge would have an
                # irrational length (notch width = g*sqrt(2)) and be unbuildable.
                # The band may locally NARROW where a top and bottom notch meet;
                # that is fine -- the min-length rule applies to wall PIECES (the
                # diagonal legs are length g >= 5), not to the band width. We
                # only need the two notches not to cross (H - 2*depth > 0); a
                # simple/valid check below enforces the rest.
                both = True
                if H - 2 * depth <= 0.5:
                    continue
                for seg in seg_choices:
                    if time.process_time() >= deadline:
                        break
                    notch_w = 2 * depth
                    period = seg + notch_w
                    n = int((axis_len - 2 * inset - seg) // period)
                    if n < 1:
                        continue
                    # connector budget (measured empirically for the mirrored
                    # both-edge band): DIAGONAL = 4*n, RIGHT = 4 + 2*n.
                    cap_d = have_diag // 4
                    cap_r = (have_right - 4) // 2
                    max_n = min(cap_d, cap_r)
                    if max_n < 1:
                        continue
                    n = min(n, max_n)
                    used = seg + n * period
                    if used < 20:
                        continue
                    start_lo = inset
                    start_hi = axis_len - inset - used
                    if start_hi < start_lo - 1e-9:
                        continue
                    span = max(0.0, start_hi - start_lo)
                    offsets = {start_lo, start_lo + span / 2, start_hi}
                    for s0 in sorted(offsets):
                        verts = self._band_vertices(
                            s0, inset, H, depth, seg, n, to_world, both
                        )
                        if verts is None:
                            continue
                        try:
                            poly = _P(verts)
                        except Exception:
                            continue
                        if not poly.is_valid or poly.area <= 0:
                            continue
                        if not pr.contains(poly):
                            continue
                        built = self._trace_polygon(poly)
                        if built is None:
                            continue
                        cons = Construction(*built)
                        sc = self._score_cons(cons)
                        if sc > best_score:
                            best, best_score = cons, sc
        return best

    @staticmethod
    def _band_vertices(s0, inset, H, depth, seg, n, to_world, both=True):
        """Vertex ring of a notched band in local (u,v) coords mapped through
        `to_world`. Bottom edge (v=v0) L->R with V dips UP by `depth`. If
        `both`, the top edge (v=vt) R->L mirrors with V dips DOWN; otherwise the
        top edge is a single flat run (used when the room is too short to notch
        both edges). Each V leg is a 45-degree wall of geometric length
        depth*sqrt(2)."""
        v0 = inset
        vt = inset + H
        pts = []
        u = s0
        pts.append((u, v0))  # bottom-left cap
        u += seg
        pts.append((u, v0))
        for _ in range(n):
            pts.append((u + depth, v0 + depth))  # up the leg (into band)
            pts.append((u + 2 * depth, v0))  # back down to edge
            u += 2 * depth
            u += seg
            pts.append((u, v0))
        pts.append((u, vt))  # right cap (vertical, RIGHT corner)
        u_right = u
        if both:
            u2 = u_right - seg
            pts.append((u2, vt))
            for _ in range(n):
                pts.append((u2 - depth, vt - depth))  # dip down (into band)
                pts.append((u2 - 2 * depth, vt))
                u2 -= 2 * depth
                u2 -= seg
                pts.append((u2, vt))
            if abs(u2 - s0) > 1e-6:
                return None
        else:
            # flat top edge straight back to the left cap. its length is the
            # full band length; it will be split into <=30 runs by the tiler,
            # which costs STRAIGHT connectors -- acceptable since only one edge
            # is notched.
            pts.append((s0, vt))
        return [to_world(uu, vv) for uu, vv in pts]

    def _chain_shape(self, deadline) -> Construction | None:
        """Builds ROTATED rectilinear shapes (rectangle / plus / pinwheel) the
        way the engine does: a turtle walk of integer-length PIECES at an
        arbitrary start heading. Vertices may land on non-integer points (fine),
        only piece lengths need be integers -- so this reaches shapes our
        vertex-based tracers reject as 'non-integer edge'. Unblocks rooms whose
        best fit is tilted to their own axis (e.g. star).

        Each template is a ring of (edge_length, turn_sign) where +1 = convex
        90, -1 = reflex 270. We scan rotation + center offset, place a gate, and
        keep the best valid build.
        """
        room = self.room.polygon
        minx, miny, maxx, maxy = room.bounds
        cx0, cy0 = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        span = max(maxx - minx, maxy - miny)

        gate_lengths = sorted(
            L
            for L, cnt in self.inventory.gates.items()
            if cnt > 0 and 5 <= L <= MAX_FACE_LENGTH
        )
        if not gate_lengths:
            return None
        gmin = gate_lengths[0]

        # edge builders: split an integer edge length into wall pieces <= 30,
        # optionally seating one gate. returns list[(PieceType, L)] or None.
        def build_edge(total, want_gate):
            return self._tile_run(
                int(round(total)),
                dict(self.inventory.walls),
                dict(self.inventory.gates),
                want_gate,
            )

        # templates: each returns a ring of (length, turn_sign). lengths are
        # parametrised by integer dims. turn_sign +1 convex(90), -1 reflex(270).
        def tmpl_rect(w, h):
            return [(w, +1), (h, +1), (w, +1), (h, +1)]

        def tmpl_plus(arm, th):
            # a symmetric plus: arm length `arm`, bar thickness `th`.
            return [
                (th, +1),
                (arm, +1),
                (th, -1),
                (arm, +1),
                (th, +1),
                (arm, +1),
                (th, -1),
                (arm, +1),
                (th, +1),
                (arm, +1),
                (th, -1),
                (arm, +1),
            ]

        def tmpl_pinwheel_p10(a, b, s, c_):
            # topology of p10's star winner: lens [a,b,s,s,c,s,s,s,c,s,s,b] with
            # turns +,+,-,+,+,-,+,+,-,+,+,- (sum 4). `c_` is scanned freely; the
            # walk-closure check in _ring_to_construction rejects combos that
            # don't close, so we don't need a closed-form for it.
            turns = [+1, +1, -1, +1, +1, -1, +1, +1, -1, +1, +1, -1]
            lens = [a, b, s, s, c_, s, s, s, c_, s, s, b]
            return list(zip(lens, turns))

        templates = []
        maxdim = min(int(span), MAX_FACE_LENGTH)
        for w in range(6, maxdim + 1, 2):
            for h in range(6, maxdim + 1, 2):
                templates.append(tmpl_rect(w, h))
        for arm in range(6, min(maxdim, 18) + 1, 2):
            for th in range(max(5, gmin), min(arm, 16) + 1, 2):
                templates.append(tmpl_plus(arm, th))
        # pinwheel family (unblocks rotated star/pinwheel rooms). c_ scanned.
        for a in range(8, min(maxdim, 20) + 1):
            for b in range(5, min(a, 14) + 1):
                for s in range(5, 9):
                    for c_ in range(5, min(maxdim, 16) + 1):
                        templates.append(tmpl_pinwheel_p10(a, b, s, c_))

        best = None
        best_score = float("-inf")
        # fine rotation: rectilinear shapes repeat every 90, but the room's
        # symmetry axis can be any angle, so scan the full quarter finely.
        rots = [r for r in range(0, 90, 3)]

        # PRE-FILTER: keep only templates whose turtle walk actually CLOSES
        # (position + heading return to start). Closure is rotation/translation
        # invariant, so we test each template once at rot 0 with a cheap walk --
        # this throws out the vast majority of (a,b,s,c) combos before the
        # expensive per-rotation/offset build+validate scan.
        def closes(ring):
            if sum(s for _, s in ring) != 4:
                return False
            x = y = 0.0
            hd = 0.0
            for i, (L, sgn) in enumerate(ring):
                hd += 90.0 if sgn > 0 else -90.0
                x += L * math.cos(math.radians(hd))
                y += L * math.sin(math.radians(hd))
            return math.hypot(x, y) < 1e-6

        closed = [r for r in templates if closes(r)]

        # center offsets: fine grid so we can slide the shape into the room's
        # best position. kept small (per-step 2 units) and bounded.
        step = 2.0
        rng = max(step, span * 0.25)
        n_off = int(rng / step)
        offs = (
            [0.0]
            + [k * step for k in range(1, n_off + 1)]
            + [-k * step for k in range(1, n_off + 1)]
        )

        best = None
        best_score = float("-inf")
        # keep the top-K distinct (ring, rot) candidates by coarse score, so the
        # fine refine isn't locked to a single template that merely won the
        # coarse round on a lucky offset -- the true optimum may belong to a
        # template whose coarse-grid score was slightly lower.
        import heapq

        topk = []  # (score, tiebreak, ring, rot, ox, oy)
        tie = 0
        for ring in closed:
            if time.process_time() >= deadline:
                break
            for rot in rots:
                if time.process_time() >= deadline:
                    break
                for ox in offs:
                    for oy in offs:
                        cons = self._ring_to_construction(
                            ring, rot, cx0 + ox, cy0 + oy, build_edge, gmin
                        )
                        if cons is None:
                            continue
                        sc = self._score_cons(cons)
                        if sc > best_score:
                            best, best_score = cons, sc
                        tie += 1
                        if len(topk) < 8:
                            heapq.heappush(topk, (sc, tie, ring, rot, ox, oy))
                        elif sc > topk[0][0]:
                            heapq.heapreplace(topk, (sc, tie, ring, rot, ox, oy))

        # LOCAL REFINE: slide each top candidate on a fine (0.5-unit) grid to
        # recover sub-`step` area the coarse scan missed (optima often sit on
        # fractional offsets).
        fine = [d * 0.5 for d in range(-6, 7)]
        for _sc, _t, ring, rot, ox0, oy0 in sorted(topk, reverse=True):
            if time.process_time() >= deadline:
                break
            for dx in fine:
                for dy in fine:
                    cons = self._ring_to_construction(
                        ring, rot, cx0 + ox0 + dx, cy0 + oy0 + dy, build_edge, gmin
                    )
                    if cons is None:
                        continue
                    sc = self._score_cons(cons)
                    if sc > best_score:
                        best, best_score = cons, sc
        return best

    def _ring_to_construction(self, ring, rot, cx, cy, build_edge, gmin):
        """Turn a template ring of (edge_length, turn_sign) into a valid
        Construction at rotation `rot`, centered near (cx, cy). Builds each edge
        (splitting >30, seating one gate on the first capable edge), lays out the
        pieces/connectors, recenters the loop on (cx, cy), and returns it only if
        the walk closes and fits. Vertices may be non-integer; only piece lengths
        are integers."""
        # choose the gate edge. prefer an edge whose length EXACTLY equals an
        # available gate length (seats cleanly, needs no extra wall); otherwise
        # fall back to any edge that can hold the smallest gate with a buildable
        # remainder. `build_edge` does the real tiling/inventory check, so we
        # just pick a reasonable candidate order here.
        gate_set = {
            L
            for L, cnt in self.inventory.gates.items()
            if cnt > 0 and 5 <= L <= MAX_FACE_LENGTH
        }
        gate_edge = None
        for i, (L, _) in enumerate(ring):
            if int(round(L)) in gate_set:
                gate_edge = i
                break
        if gate_edge is None:
            for i, (L, _) in enumerate(ring):
                if L >= gmin and (L - gmin == 0 or L - gmin >= MIN_WALL_LENGTH):
                    gate_edge = i
                    break
        if gate_edge is None:
            return None

        pieces: list[Piece] = []
        conns: list[Connector] = []
        # connectors[i] is the turn INTO piece i; connectors[0] is the closing
        # turn (end of last edge back to first). We assemble per-edge runs and
        # set the FIRST piece of each edge to carry the corner turn, the rest
        # STRAIGHT.
        edge_runs = []
        for i, (L, sgn) in enumerate(ring):
            segs = build_edge(L, i == gate_edge)
            if segs is None:
                return None
            edge_runs.append((segs, sgn))

        # corner turn for edge i is placed at its first piece; it reflects the
        # turn FROM the previous edge INTO this edge. For a CCW ring, convex=+90
        # (RIGHT), reflex=-90 (RIGHT flipped => internal 270). The turn at the
        # start of edge i is determined by edge i's own sign.
        for i, (segs, sgn) in enumerate(edge_runs):
            for si, (ptype, L) in enumerate(segs):
                pieces.append(Piece(ptype, L))
                if si == 0:
                    conns.append(Connector(ConnectorType.RIGHT, reflex=(sgn < 0)))
                else:
                    conns.append(Connector(ConnectorType.STRAIGHT, False))
        if len(conns) != len(pieces) or len(pieces) < 5:
            return None

        # walk from origin at heading rot, then translate to center on (cx, cy)
        from src.enclosure import build_geometry as _bg

        trial = Construction((0.0, 0.0), float(rot), pieces, conns)
        try:
            verts, pos_gap, head_gap = _bg(trial)
        except Exception:
            return None
        if pos_gap > 1e-3 or abs(head_gap) > 1e-3:
            return None  # doesn't close
        xs = [p[0] for p in verts]
        ys = [p[1] for p in verts]
        shift_x = cx - (min(xs) + max(xs)) / 2.0
        shift_y = cy - (min(ys) + max(ys)) / 2.0
        cons = Construction((shift_x, shift_y), float(rot), pieces, conns)
        return cons

    # ======================================================================
    # Template search + legacy shape library (safety-net fallbacks).
    # ======================================================================
    def _legacy_search(self, best_score: float):
        """Proven shape library (rectangles, tiled rectangles, octagons) placed
        with the wall-aligned `_place`, then the room-aware L fallback. Returns
        (construction, true_score) for the best that beats `best_score`, or None."""
        cands = self._rectangles() + self._tiled_rectangles() + self._octagons()
        cands.sort(key=lambda c: c[0], reverse=True)
        best = None
        bscore = best_score
        for score, pieces, conns in cands[:MAX_TRY]:
            if score <= bscore:
                break
            cons = self._place(pieces, conns)
            if cons is not None:
                sc = self._score_cons(cons)
                if sc > bscore:
                    best, bscore = cons, sc
                    break
        if best is None:
            fb = self._l_shape_fallback()
            if fb is not None:
                sc = self._score_cons(fb)
                if sc > best_score:
                    best, bscore = fb, sc
        return (best, bscore) if best is not None else None

    @staticmethod
    def _template_turns(family):
        return {
            "rectangle": (90, 90, 90, 90),
            "hexagon": (90, 45, 45, 90, 45, 45),
            "octagon": (45,) * 8,
            "L-shape": (90, 90, 90, -90, 90, 90),
        }[family]

    @staticmethod
    def _template_area(family, sides):
        if family == "rectangle":
            return sides[0] * sides[1]
        if family == "hexagon":
            a, b, c = sides[:3]
            return a * c + (a + c) * b / math.sqrt(2)
        if family == "octagon":
            a, b, c, d = sides[:4]
            return a * c + (a + c) * (b + d) / math.sqrt(2) + b * d
        width, c, b, d, a, _ = sides  # L-shape
        return width * c + a * d

    @staticmethod
    def _integer_vertices(lengths, turns):
        """Check the shape actually closes back to its start, then return its
        corner points. Returns None if the sides don't form a closed loop."""
        if (
            len(lengths) != len(turns)
            or sum(turns) != 360
            or any(
                not isinstance(n, int) or not 5 <= n <= MAX_FACE_LENGTH for n in lengths
            )
        ):
            return None
        ax = ay = dx_ = dy_ = 0
        direction = 0
        verts = []
        dirs = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
        for i, L in enumerate(lengths):
            if i:
                direction = (direction + turns[i] // 45) % 8
            verts.append((ax + dx_ / math.sqrt(2), ay + dy_ / math.sqrt(2)))
            ddx, ddy = dirs[direction]
            if direction % 2:
                dx_ += L * ddx
                dy_ += L * ddy
            else:
                ax += L * ddx
                ay += L * ddy
        if (ax, ay, dx_, dy_) != (0, 0, 0, 0):
            return None
        return verts

    # ----- scoring helpers (every builder is gated through these) -----
    def _score_cons(self, cons: Construction) -> float:
        """True score of a construction, or -inf if it isn't valid."""
        res = validate_construction(cons, self.room, self.inventory)
        if not res.valid:
            return float("-inf")
        return score_construction(res, self.weights)

    def _score_poly(self, area: float, perim: float, two_gates: bool) -> float:
        """Score a shape from its raw area/perimeter (used to rank candidates
        before building them)."""
        g = self.weights.G if two_gates else 0.0
        return 1000.0 + self.weights.A * area + self.weights.C * perim + g

    def _maximal_shape(self, deadline) -> Construction | None:
        """Grow the biggest simple shape that fits inside the room and build it.

        Tries three shape families, picking whatever the corner connectors allow
        and whatever scores best once traced:
          - plain rectangle (4 square corners),
          - chamfered rectangle / octagon (corners cut at 45 degrees), which
            fills a square-ish space better than a rectangle, and
          - gabled rectangle (a 45-degree roof on one side), useful when a flat
            top wastes space against a slanted part of the room.
        For each candidate footprint we scan positions and sizes inside the
        room, keep the ones that fit, trace them into real pieces, and return
        the highest-scoring valid build. Never grows a face past the 30-unit
        limit (longer sides are split by _trace_polygon's tiling)."""
        room = self.room.polygon
        prepared = prep(room.buffer(TOL))
        minx, miny, maxx, maxy = (int(round(b)) for b in room.bounds)

        have_right = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        have_diag = self.inventory.connectors.get(ConnectorType.DIAGONAL, 0)

        best = None
        best_score = float("-inf")

        def consider(coords):
            nonlocal best, best_score
            poly = Polygon(coords)
            if not poly.is_valid or poly.area <= 0:
                return
            if not poly.exterior.is_simple:
                return
            if not prepared.contains(poly):
                return
            built = self._trace_polygon(poly)
            if built is None:
                return
            cons = Construction(built[0], built[1], built[2], built[3])
            sc = self._score_cons(cons)
            if sc > best_score:
                best, best_score = cons, sc

        # step size: coarse on big rooms so we stay inside the time budget
        span = max(maxx - minx, maxy - miny)
        # step size for seeding: coarse on big rooms so we cover the whole room
        span = max(maxx - minx, maxy - miny)
        seed_step = max(5, span // 20)
        grow_step = max(1, span // 60)

        def fits(coords):
            poly = Polygon(coords)
            return (
                poly.is_valid
                and poly.area > 0
                and poly.exterior.is_simple
                and prepared.contains(poly)
            )

        # ---- greedy grow: seed a small box at many points, then push each of
        # its four edges outward until it hits a wall. this scales as O(room)
        # per seed instead of scanning every box, so it works on huge rooms.
        # we checkpoint (keep the best traced+scored build) as we grow, so if
        # time runs out we still return the biggest shape grown so far. ----
        if have_right >= 4:
            for sx in range(minx, maxx, seed_step):
                if time.process_time() >= deadline:
                    break
                for sy in range(miny, maxy, seed_step):
                    # start with a tiny box centered on the seed, if it fits
                    sx0, sy0, sx1, sy1 = sx - 3, sy - 3, sx + 3, sy + 3
                    if not fits([(sx0, sy0), (sx1, sy0), (sx1, sy1), (sx0, sy1)]):
                        continue
                    # greedy growth is order-sensitive (a box that grows wide
                    # first can block itself from growing tall), so try several
                    # priority orders per seed and keep the best of each.
                    orders = (
                        ("L", "R", "D", "U"),  # balanced
                        ("L", "R"),  # width-first, then all
                        ("D", "U"),  # height-first, then all
                    )
                    for order in orders:
                        if time.process_time() >= deadline:
                            break
                        x0, y0, x1, y1 = sx0, sy0, sx1, sy1

                        def grow(dirs, gs):
                            nonlocal x0, y0, x1, y1
                            improved = True
                            while improved:
                                if time.process_time() >= deadline:
                                    return
                                improved = False
                                if "L" in dirs and fits(
                                    [(x0 - gs, y0), (x1, y0), (x1, y1), (x0 - gs, y1)]
                                ):
                                    x0 -= gs
                                    improved = True
                                if "R" in dirs and fits(
                                    [(x0, y0), (x1 + gs, y0), (x1 + gs, y1), (x0, y1)]
                                ):
                                    x1 += gs
                                    improved = True
                                if "D" in dirs and fits(
                                    [(x0, y0 - gs), (x1, y0 - gs), (x1, y1), (x0, y1)]
                                ):
                                    y0 -= gs
                                    improved = True
                                if "U" in dirs and fits(
                                    [(x0, y0), (x1, y0), (x1, y1 + gs), (x0, y1 + gs)]
                                ):
                                    y1 += gs
                                    improved = True

                        # grow the prioritized directions first, then everything,
                        # coarse then fine, to maximize the final rectangle.
                        grow(order, grow_step)
                        grow(("L", "R", "D", "U"), grow_step)
                        grow(("L", "R", "D", "U"), 1)
                        consider([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                        if have_diag >= 8:
                            w, h = x1 - x0, y1 - y0
                            cmax = min(w, h) // 2
                            for c in (cmax, cmax // 2, 5):
                                if 2 <= c <= cmax:
                                    consider(
                                        [
                                            (x0 + c, y0),
                                            (x1 - c, y0),
                                            (x1, y0 + c),
                                            (x1, y1 - c),
                                            (x1 - c, y1),
                                            (x0 + c, y1),
                                            (x0, y1 - c),
                                            (x0, y0 + c),
                                        ]
                                    )

        return best

    def _connector_for_angle(self, interior: float):
        """Map a corner interior angle (deg) to the connector that reproduces it."""
        tol = 2.0
        table = [
            (90.0, Connector(ConnectorType.RIGHT, False)),
            (270.0, Connector(ConnectorType.RIGHT, True)),
            (135.0, Connector(ConnectorType.DIAGONAL, False)),
            (225.0, Connector(ConnectorType.DIAGONAL, True)),
            (180.0, Connector(ConnectorType.STRAIGHT, False)),
        ]
        for ang, conn in table:
            if abs(interior - ang) <= tol:
                return conn
        return None

    def _room_traced(self) -> Construction | None:
        """Follow the room's own boundary. The room outline is the most
        area-efficient enclosure, so for any room whose corners are 45-degree
        multiples we trace it (optionally inset inward so it sits strictly
        inside), tiling each edge from a shared inventory and matching each
        corner's connector. Material is usually plentiful, so we try the FULL
        outline first, then progressively larger insets only if needed."""

        base_poly = self.room.polygon
        # try the exact outline first, then inward buffers (keeps the shape,
        # unlike scaling toward the centroid which distorts a concave room).
        for inset in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
            poly = base_poly if inset == 0.0 else base_poly.buffer(-inset, join_style=2)
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            built = self._trace_polygon(poly)
            if built is None:
                continue
            start, heading, pieces, conns = built
            cons = Construction(start, heading, pieces, conns)
            if validate_construction(cons, self.room, self.inventory).valid:
                return cons
        return None

    # ======================================================================
    # Grid fill: rasterize the room and enclose the largest block of cells.
    # ======================================================================
    def _grid_fill(self, deadline) -> Construction | None:
        """Rasterize the room: lay a grid of w-by-h cells over it, keep the
        cells wholly inside, take the largest 4-connected block, and trace that
        block's outline into pieces.

        The block's outline has only right (90/270) and straight corners -- no
        45-degree pieces. Two things make it buildable from any inventory:

        * cell width/height are chosen from the lengths we actually stock, so
          every east-west side is a whole number of ``w`` and every north-south
          side a whole number of ``h`` -- always tileable;
        * a straight run over the 30-unit face limit is notched by dropping one
          cell along it (adds two corners), and cells that touch only at a
          corner (which would make the outline self-touch) are dropped.

        This is the method for big / non-45-degree rooms (spirals, blobby
        concave rooms) where the outline-tracing families either reject the
        corners or settle for one small shape. Competes on true score, so it
        never regresses the result.
        """
        from collections import deque

        from shapely.geometry import box as _box
        from shapely.ops import unary_union

        room = self.room.polygon
        have_right = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        if have_right < 4:
            return None  # a rectilinear loop needs at least 4 right corners

        # cell dimensions come from the wall/gate lengths we stock, so every
        # grid-aligned side is a multiple of a buildable length. keep a spread
        # from small (fine fit, more perimeter) to large (coarse, cheap).
        stock_lengths = sorted(
            {L for L, n in self.inventory.walls.items() if n > 0 and 5 <= L <= 30}
            | {L for L, n in self.inventory.gates.items() if n > 0 and 5 <= L <= 30}
        )
        if not stock_lengths:
            return None
        # candidate (w, h) pairs: same-length squares first (fewest distinct
        # pieces), then a few mixed pairs so slanted walls step cheaply.
        cell_pairs = [(s, s) for s in stock_lengths]
        small = stock_lengths[:3]
        for w in small:
            for h in stock_lengths:
                if (w, h) not in cell_pairs:
                    cell_pairs.append((w, h))

        # candidate grid rotations: axis-aligned plus the room's own edge
        # directions, so a grid can line up with a slanted wall and step it
        # cheaply instead of leaving a staircase gap. a rotated rectilinear
        # loop is still all 90-degree corners (just at a rotated base heading),
        # which _trace_rectilinear builds directly.
        import math as _m

        room_corners = list(room.exterior.coords)[:-1]
        angles = {0.0}
        for i in range(len(room_corners)):
            ax, ay = room_corners[i]
            bx, by = room_corners[(i + 1) % len(room_corners)]
            ang = _m.degrees(_m.atan2(by - ay, bx - ax)) % 90.0
            angles.add(round(ang, 3))
        # keep a small set (axis-aligned first, then up to 5 distinct wall
        # angles) so we don't blow the time budget.
        rot_angles = [0.0] + sorted(a for a in angles if a > 0.5 and a < 89.5)[:5]

        best = None
        best_score = float("-inf")
        NEIGH = ((1, 0), (-1, 0), (0, 1), (0, -1))

        def largest_component(cells):
            left = set(cells)
            best_block = set()
            while left:
                seed = left.pop()
                block = {seed}
                q = deque([seed])
                while q:
                    x, y = q.popleft()
                    for dx, dy in NEIGH:
                        nb = (x + dx, y + dy)
                        if nb in left:
                            left.remove(nb)
                            block.add(nb)
                            q.append(nb)
                if len(block) > len(best_block):
                    best_block = block
            return best_block

        def connected(cells):
            if not cells:
                return False
            seed = next(iter(cells))
            seen = {seed}
            q = deque([seed])
            while q:
                x, y = q.popleft()
                for dx, dy in NEIGH:
                    nb = (x + dx, y + dy)
                    if nb in cells and nb not in seen:
                        seen.add(nb)
                        q.append(nb)
            return len(seen) == len(cells)

        def pinches(cells, around=None):
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

        def drop_pinches(cells):
            while True:
                p = pinches(cells)
                if not p:
                    return
                cells.discard(p[0])

        def runs(cells):
            rows, cols = [], []
            for side in (-1, 1):
                edge = sorted((y, x) for x, y in cells if (x, y + side) not in cells)
                run = []
                for y, x in edge:
                    if run and run[-1] == (x - 1, y):
                        run.append((x, y))
                    else:
                        if run:
                            rows.append(run)
                        run = [(x, y)]
                if run:
                    rows.append(run)
            for side in (-1, 1):
                edge = sorted((x, y) for x, y in cells if (x + side, y) not in cells)
                run = []
                for x, y in edge:
                    if run and run[-1] == (x, y - 1):
                        run.append((x, y))
                    else:
                        if run:
                            cols.append(run)
                        run = [(x, y)]
                if run:
                    cols.append(run)
            return rows, cols

        def break_long_sides(cells, most_x, most_y):
            # most_x cells == 30 units along a row; notch anything longer by
            # dropping a cell (keeping the block connected and pinch-free).
            while time.process_time() < deadline:
                rows, cols = runs(cells)
                long = [(r, most_x) for r in rows if len(r) > most_x]
                long += [(cc, most_y) for cc in cols if len(cc) > most_y]
                if not long:
                    return True
                for run, most in long:
                    for k in range(most, 0, -1):
                        cell = run[k]
                        cells.discard(cell)
                        x, y = cell
                        near = [
                            (x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                        ]
                        if not pinches(cells, near) and connected(cells):
                            break
                        cells.add(cell)
                    else:
                        return False
            return False

        minx, miny, maxx, maxy = room.bounds
        from shapely import affinity

        for rot in rot_angles:
            if time.process_time() >= deadline:
                break
            # rasterize in a frame where the grid is axis-aligned: rotate the
            # room by -rot, grid it, then the block is rotated back by +rot.
            if rot == 0.0:
                rroom = room
            else:
                rroom = affinity.rotate(room, -rot, origin=(0, 0))
            rprep = prep(rroom.buffer(TOL))
            rminx, rminy, rmaxx, rmaxy = rroom.bounds
            for w, h in cell_pairs:
                if time.process_time() >= deadline:
                    break
                most_x = int(MAX_FACE_LENGTH // w)
                most_y = int(MAX_FACE_LENGTH // h)
                if most_x < 1 or most_y < 1:
                    continue
                for sx in (0.0, 0.5):
                    for sy in (0.0, 0.5):
                        if time.process_time() >= deadline:
                            break
                        ox = rminx - sx * w
                        oy = rminy - sy * h
                        cells = set()
                        a = 0
                        x = ox
                        while x < rmaxx:
                            b = 0
                            y = oy
                            while y < rmaxy:
                                if rprep.contains(_box(x, y, x + w, y + h)):
                                    cells.add((a, b))
                                b += 1
                                y += h
                            a += 1
                            x += w
                        if len(cells) < 4:
                            continue
                        cells = largest_component(cells)
                        drop_pinches(cells)
                        cells = largest_component(cells)
                        if len(cells) < 4:
                            continue
                        if not break_long_sides(cells, most_x, most_y):
                            continue
                        boxes = [
                            _box(
                                ox + a * w,
                                oy + b * h,
                                ox + (a + 1) * w,
                                oy + (b + 1) * h,
                            )
                            for a, b in cells
                        ]
                        block = unary_union(boxes)
                        if block.geom_type != "Polygon" or block.interiors:
                            continue
                        if block.area <= 0 or not block.exterior.is_simple:
                            continue
                        # collinear-merge so each straight run of cells is one
                        # face (up to 30 units) that can seat a gate.
                        block = block.simplify(0.0)
                        if rot == 0.0:
                            if not room.buffer(TOL).contains(block):
                                continue
                            built = self._trace_polygon(block)
                        else:
                            # rotate the block back into the room's frame; it is
                            # a rectilinear loop at base heading `rot`.
                            wblock = affinity.rotate(block, rot, origin=(0, 0))
                            if not room.buffer(TOL).contains(wblock):
                                continue
                            built = self._trace_rectilinear(block, rot)
                        if built is None:
                            continue
                        cons = Construction(built[0], built[1], built[2], built[3])
                        sc = self._score_cons(cons)
                        if sc > best_score:
                            best, best_score = cons, sc
        return best

    # ======================================================================
    # Free-form grower: slide a loop's corners outward to fill the room.
    # ======================================================================
    def _freeform_grow(self, deadline, seed_cons=None) -> Construction | None:
        """Grow a free-form loop that fills the room (ported mechanism from a
        shared class algorithm). Starts from seed loops (the grid-fill block and
        a fitting rectangle), then repeatedly slides runs of corners outward to
        swallow more room, keeping only loops the inventory can build and the
        simulator accepts. Wins on non-convex rooms where a fixed shape or the
        orthogonal grid leaves area on the table (corridors, cut-outs, slanted
        walls). Competes on true score -> never regresses."""
        import numpy as np
        import shapely as _sh

        from players.player9.freeform import Grower, Loop

        room = self.room.polygon
        region = room.buffer(TOL)
        _sh.prepare(region)  # in-place spatial index; stays a plain Geometry
        room_corners = np.array(room.exterior.coords[:-1])

        rights = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        diagonals = self.inventory.connectors.get(ConnectorType.DIAGONAL, 0)

        wall_lengths = sorted(
            L for L, n in self.inventory.walls.items() if n > 0 and 5 <= L <= 30
        )
        gate_lengths = sorted(
            L for L, n in self.inventory.gates.items() if n > 0 and 5 <= L <= 30
        )
        if not gate_lengths or (rights < 4 and diagonals < 4):
            return None
        all_lengths = sorted(set(wall_lengths) | set(gate_lengths))
        if not all_lengths:
            return None
        buildable = set()
        reach = {0: 0}
        for t in range(5, MAX_FACE_LENGTH + 1):
            for p in all_lengths:
                if p <= t and (t - p) in reach:
                    reach[t] = min(reach.get(t, 99), reach[t - p] + 1)
                    buildable.add(t)
                    break
        if not buildable:
            return None
        total_length = sum(L * n for L, n in self.inventory.walls.items()) + sum(
            L * n for L, n in self.inventory.gates.items()
        )

        def score_fn(area, perimeter):
            return (
                BASELINE_SCORE
                + self.weights.A * area
                + self.weights.C * perimeter
                + max(self.weights.G, 0.0)
            )

        def faces_to_cons(start, heading, faces):
            n = len(faces)
            if n < 3:
                return None
            walls = dict(self.inventory.walls)
            gates = dict(self.inventory.gates)
            cap = [
                i
                for i in range(n)
                if any(
                    gl <= faces[i][1]
                    and (faces[i][1] - gl == 0 or faces[i][1] - gl >= 5)
                    for gl in gate_lengths
                )
            ]
            if not cap:
                return None
            gate_faces = {cap[0]}
            if sum(gates.values()) >= 2:
                far = max(cap, key=lambda i: min((i - cap[0]) % n, (cap[0] - i) % n))
                if far != cap[0]:
                    gate_faces.add(far)
            pieces, conns = [], []
            for i in range(n):
                h, length = faces[i]
                ph = faces[(i - 1) % n][0]
                t = (h - ph) % 8
                if t == 2:
                    corner = Connector(ConnectorType.RIGHT, False)
                elif t == 6:
                    corner = Connector(ConnectorType.RIGHT, True)
                elif t == 1:
                    corner = Connector(ConnectorType.DIAGONAL, False)
                elif t == 7:
                    corner = Connector(ConnectorType.DIAGONAL, True)
                else:
                    return None
                segs = self._tile_run(int(round(length)), walls, gates, i in gate_faces)
                if segs is None:
                    return None
                for k, (ptype, L) in enumerate(segs):
                    if L < 5:
                        return None  # safety: never emit a sub-5 piece
                    pieces.append(Piece(ptype, L))
                    conns.append(
                        corner if k == 0 else Connector(ConnectorType.STRAIGHT, False)
                    )
            # start_heading is the WORLD direction of face 0 (heading of index 0
            # plus 45 deg per direction-index step).
            start_heading = heading + 45.0 * faces[0][0]
            return Construction((start[0], start[1]), start_heading, pieces, conns)

        def can_build(lengths):
            walls = dict(self.inventory.walls)
            gates = dict(self.inventory.gates)
            for L in lengths:
                if self._tile_run(int(round(L)), walls, gates, False) is None:
                    return False
            return True

        accepted = {"best": None, "score": float("-inf")}

        def accept(start, heading, faces):
            try:
                cons = faces_to_cons(start, heading, faces)
            except Exception:
                return None
            if cons is None:
                return None
            res = validate_construction(cons, self.room, self.inventory)
            if not res.valid:
                return None
            sc = score_construction(res, self.weights)
            if sc > accepted["score"]:
                accepted["best"], accepted["score"] = cons, sc
            return sc

        seeds = []

        def loop_from_cons(cons):
            pts = walk_points(cons)[:-1]
            n = len(pts)
            faces = []
            for i in range(n):
                dx = pts[(i + 1) % n][0] - pts[i][0]
                dy = pts[(i + 1) % n][1] - pts[i][1]
                length = round(math.hypot(dx, dy))
                if length < 1:
                    continue
                ang = round((math.degrees(math.atan2(dy, dx)) % 360) / 45) % 8
                faces.append((ang, length))
            if len(faces) < 3:
                return None
            merged = []
            for h, length in faces:
                if merged and merged[-1][0] == h:
                    merged[-1] = (h, merged[-1][1] + length)
                else:
                    merged.append((h, length))
            sc = self._score_cons(cons)
            return Loop(pts[0], 0.0, tuple(merged), sc if sc > 0 else 0.0)

        grid = self._grid_fill(min(deadline, time.process_time() + 20.0))
        if grid is not None:
            lp = loop_from_cons(grid)
            if lp is not None:
                seeds.append(lp)
        # refine an externally supplied best construction too (e.g. the best
        # the template / hug / maximal searches found): the grower can polish
        # it by sliding corners outward into unused room.
        if seed_cons is not None:
            lp = loop_from_cons(seed_cons)
            if lp is not None:
                seeds.append(lp)
        minx, miny, maxx, maxy = room.bounds
        from shapely.geometry import Polygon as _P

        min_gate = min(gate_lengths)
        # small-to-medium rectangles at several positions so corridors and
        # separate pockets each get a seed to grow from. size each so a side
        # can hold the smallest gate.
        base_side = max(5, min(MAX_FACE_LENGTH, min_gate + 5))
        positions = [(room.centroid.x, room.centroid.y)]
        rp = room.representative_point()
        positions.append((rp.x, rp.y))
        for fx in (0.25, 0.5, 0.75):
            for fy in (0.25, 0.5, 0.75):
                positions.append((minx + (maxx - minx) * fx, miny + (maxy - miny) * fy))
        for cx, cy in positions:
            for side in (base_side, min(MAX_FACE_LENGTH, base_side * 2)):
                x0, y0 = cx - side / 2, cy - side / 2
                rect = _P(
                    [(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)]
                )
                if room.buffer(TOL).contains(rect):
                    seeds.append(
                        Loop(
                            (x0, y0),
                            0.0,
                            ((0, side), (2, side), (4, side), (6, side)),
                            0.0,
                        )
                    )
                    break  # one fitting square per position is enough
        # de-duplicate seeds by (start, faces)
        uniq = {}
        for s in seeds:
            uniq[(round(s.start[0], 3), round(s.start[1], 3), s.faces)] = s
        seeds = list(uniq.values())
        if not seeds:
            return None

        # accept each seed up front: the grower only validates grown loops, so
        # without this a perfectly good seed (e.g. the grid-fill block) could be
        # lost if no improving move happens to be accepted.
        for s in seeds:
            accept(s.start, s.heading, s.faces)

        grower = Grower(
            region=region,
            room_corners=room_corners,
            buildable=buildable,
            rights=rights,
            diagonals=diagonals,
            total_length=total_length,
            score=score_fn,
            accept=accept,
            can_build=can_build,
        )
        try:
            grower.grow(seeds, deadline, first_charge=max(self.weights.A, 0.1))
        except Exception:
            return accepted["best"]
        return accepted["best"]

    # ======================================================================
    # Geometry -> pieces: turn a polygon/outline into buildable walls+gates
    # (tracing, tiling edges, boundary-hug, long-face splitting).
    # ======================================================================
    def _trace_rectilinear(self, block, rot_deg):
        """Trace an axis-aligned cell block whose loop will sit at base heading
        `rot_deg` in the room. Every corner is 90/270 (a RIGHT connector) and
        every edge is an integer piece length; the absolute heading is rotated
        but that does not change the connectors. Returns
        (start, heading, pieces, connectors) or None."""
        pts = list(block.exterior.coords)[:-1]
        n = len(pts)
        if n < 3:
            return None
        signed = sum(
            pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
            for i in range(n)
        )
        if signed < 0:
            pts = pts[::-1]
        # in the axis-aligned frame, headings are 0/90/180/270; corner turns are
        # +-90. edges are axis lengths (integer). build corner connectors + edges.
        hd = [
            math.degrees(
                math.atan2(
                    pts[(i + 1) % n][1] - pts[i][1], pts[(i + 1) % n][0] - pts[i][0]
                )
            )
            % 360
            for i in range(n)
        ]
        corner_conns = []
        budget = {
            ConnectorType.RIGHT: self.inventory.connectors.get(ConnectorType.RIGHT, 0),
            ConnectorType.STRAIGHT: self.inventory.connectors.get(
                ConnectorType.STRAIGHT, 0
            ),
        }
        for i in range(n):
            turn = (hd[i] - hd[(i - 1) % n] + 180.0) % 360.0 - 180.0
            internal = round((180.0 - turn) % 360.0)
            if abs(internal - 90) < 2:
                conn = Connector(ConnectorType.RIGHT, False)
            elif abs(internal - 270) < 2:
                conn = Connector(ConnectorType.RIGHT, True)
            elif abs(internal - 180) < 2:
                conn = Connector(ConnectorType.STRAIGHT, False)
            else:
                return None  # not rectilinear
            corner_conns.append(conn)
            budget[conn.connector_type] -= 1
            if budget[conn.connector_type] < 0:
                return None
        edges = []
        for i in range(n):
            dx = pts[(i + 1) % n][0] - pts[i][0]
            dy = pts[(i + 1) % n][1] - pts[i][1]
            edges.append(math.hypot(dx, dy))
        if any(abs(e - round(e)) > 0.05 for e in edges):
            return None
        straight_budget = budget[ConnectorType.STRAIGHT]
        built = self._tile_outline_global(edges, corner_conns, straight_budget)
        if built is None:
            return None
        pieces, conns = built
        # rotate the start point into the room frame and offset the heading.
        th = math.radians(rot_deg)
        c_, s_ = math.cos(th), math.sin(th)
        sx = pts[0][0] * c_ - pts[0][1] * s_
        sy = pts[0][0] * s_ + pts[0][1] * c_
        return (sx, sy), hd[0] + rot_deg, pieces, conns

    def _trace_polygon(self, poly) -> tuple | None:
        """Build pieces+connectors that trace `poly`'s boundary exactly. Returns
        (start, heading, pieces, connectors) or None if it can't be tiled from
        inventory within the connector budgets."""
        pts = list(poly.exterior.coords)[:-1]
        n = len(pts)
        if n < 3:
            return None
        # CCW winding so turns sum to +360
        signed = sum(
            pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
            for i in range(n)
        )
        if signed < 0:
            pts = pts[::-1]

        hd = [
            math.degrees(
                math.atan2(
                    pts[(i + 1) % n][1] - pts[i][1], pts[(i + 1) % n][0] - pts[i][0]
                )
            )
            for i in range(n)
        ]
        # corner connector = turn INTO edge i (at vertex i)
        corner_conns = []
        corner_budget = {
            ConnectorType.RIGHT: self.inventory.connectors.get(ConnectorType.RIGHT, 0),
            ConnectorType.DIAGONAL: self.inventory.connectors.get(
                ConnectorType.DIAGONAL, 0
            ),
            ConnectorType.STRAIGHT: self.inventory.connectors.get(
                ConnectorType.STRAIGHT, 0
            ),
        }
        for i in range(n):
            turn = (hd[i] - hd[(i - 1) % n] + 180.0) % 360.0 - 180.0
            internal = round((180.0 - turn) % 360.0)
            conn = self._connector_for_angle(internal)
            if conn is None:
                return None  # non-45 corner
            corner_conns.append(conn)
            corner_budget[conn.connector_type] -= 1
            if corner_budget[conn.connector_type] < 0:
                return None  # not enough corner connectors of this type

        # each edge's buildable PIECE length == its geometric length. The
        # engine advances a piece of length L by exactly L units along its
        # heading (src/enclosure.py line ~102: dx = length*cos(theta)), so a
        # diagonal (45-deg) wall of length L spans L geometric units too -- NOT
        # L*sqrt(2). (Earlier code divided diagonals by sqrt(2), which built
        # shapes smaller than intended.)
        edges = []
        for i in range(n):
            dx = pts[(i + 1) % n][0] - pts[i][0]
            dy = pts[(i + 1) % n][1] - pts[i][1]
            glen = math.hypot(dx, dy)
            edges.append(glen)
        # reject if any edge isn't close to an integer buildable length
        if any(abs(e - round(e)) > 0.05 for e in edges):
            return None
        # remaining STRAIGHT connectors available to JOIN pieces within edges
        straight_budget = corner_budget[ConnectorType.STRAIGHT]

        built = self._tile_outline_global(edges, corner_conns, straight_budget)
        if built is None:
            return None
        pieces, conns = built
        return (pts[0][0], pts[0][1]), hd[0], pieces, conns

    def _boundary_hug(self) -> Construction | None:
        """Follow the room's own outline, just inside the walls. The room shape
        holds the most area, so hugging it beats any fixed shape on odd rooms.
        We pull the walls in by a few amounts (so the fence sits inside), break
        up any wall run longer than 30 with a small inward notch, then build and
        place it. We keep the best size that actually fits our pieces."""
        import time as _t

        pts0 = list(self.room.polygon.exterior.coords)[:-1]
        n0 = len(pts0)
        if n0 < 3:
            return None
        if (
            sum(
                pts0[i][0] * pts0[(i + 1) % n0][1] - pts0[(i + 1) % n0][0] * pts0[i][1]
                for i in range(n0)
            )
            < 0
        ):
            pts0 = pts0[::-1]

        best = None
        best_score = 0.0
        deadline = _t.process_time() + 30.0
        sq = math.sqrt(2)
        for d in (1, 2, 3, 4, 5, 6, 7, 8, 2.5, 3.5, 4.5, 1.5):  # inset depths
            if _t.process_time() >= deadline:
                break
            verts = self._inset_integer(pts0, d, d / sq)
            if verts is None:
                continue
            verts = self._split_long_faces(verts, d)
            if verts is None or len(verts) < 3:
                continue
            try:
                hp = Polygon(verts)
            except Exception:
                continue
            if not hp.is_valid or hp.area <= 0:
                continue
            if not self.room.polygon.buffer(1e-6).contains(hp):
                continue
            built = self._trace_polygon(hp)
            if built is None:
                continue
            start, heading, pieces, conns = built
            cons = Construction(start, heading, pieces, conns)
            res = validate_construction(cons, self.room, self.inventory)
            if res.valid:
                sc = score_construction(res, self.weights)
                if sc > best_score:
                    best, best_score = cons, sc
        return best

    @staticmethod
    def _inset_integer(pts, d_axis, d_diag):
        """Pull every wall inward along its own normal (straight walls by
        d_axis, diagonal walls by d_diag) and reconnect the corners. The amounts
        are chosen so the new corners still land on buildable points."""
        n = len(pts)
        lines = []
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy)
            if L < 1e-9:
                return None
            nx, ny = -dy / L, dx / L  # inward normal for CCW
            h = round(math.degrees(math.atan2(dy, dx)) % 90)
            off = d_diag if abs(h - 45) < 1 else d_axis
            lines.append(((ax + nx * off, ay + ny * off), (dx, dy)))

        def inter(p1, d1, p2, d2):
            x1, y1 = p1
            x2, y2 = p2
            dx1, dy1 = d1
            dx2, dy2 = d2
            den = dx1 * dy2 - dy1 * dx2
            if abs(den) < 1e-9:
                return None
            t = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / den
            return (x1 + dx1 * t, y1 + dy1 * t)

        verts = []
        for i in range(n):
            v = inter(
                lines[(i - 1) % n][0], lines[(i - 1) % n][1], lines[i][0], lines[i][1]
            )
            if v is None:
                return None
            verts.append(v)
        return verts

    def _split_long_faces(self, verts, inset):
        """Break any straight run longer than 30 units into legal sub-faces by
        inserting a small rectangular inward notch, so no single face exceeds
        the 30-unit limit. The notch steps DEPTH units inward, runs forward a
        little, then back out -- all legs >= the 5-unit min piece length and at
        90-degree (right) corners, so every piece stays buildable.

        (An outward bump would add area, but it is fragile: a bump that fits
        locally can still make the full traced polygon self-touch or produce an
        unbuildable edge, which collapses the whole hug. Inward is always safe,
        so it is what we use.)"""
        n = len(verts)
        if n < 3:
            return verts
        DEPTH = 5.0
        out = []
        for i in range(n):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % n]
            out.append((ax, ay))
            L = math.hypot(bx - ax, by - ay)
            if L <= 30 + 1e-6:
                continue
            ux, uy = (bx - ax) / L, (by - ay) / L
            nx, ny = -uy, ux  # inward normal (interior on left, CCW)
            segs = int(math.ceil(L / 28.0))
            step = L / segs
            if step < 2 * DEPTH + 5:
                continue  # too short to notch cleanly; leave as-is
            for k in range(1, segs):
                cx, cy = ax + ux * step * k, ay + uy * step * k
                a1 = (round(cx + nx * DEPTH), round(cy + ny * DEPTH))
                a2 = (
                    round(cx + ux * DEPTH + nx * DEPTH),
                    round(cy + uy * DEPTH + ny * DEPTH),
                )
                out.append((round(cx), round(cy)))  # arrive on the edge
                out.append(a1)  # 90-deg turn inward
                out.append(a2)  # run forward inside
                out.append((round(cx + ux * DEPTH), round(cy + uy * DEPTH)))
        return out

    def _tile_outline_global(self, edge_lengths, corner_conns, straight_budget):
        """Tile all edges from one shared inventory, hardest edge first, each
        edge tiled with the FEWEST pieces (to conserve straight connectors).
        Keeps a running straight-connector budget. connectors[i] is the turn
        INTO piece i, so each edge's first piece carries its corner connector and
        the rest carry STRAIGHT. Places two gates on different faces if it can."""
        walls = dict(self.inventory.walls)
        gates = dict(self.inventory.gates)
        n = len(edge_lengths)
        targets = [int(round(e)) for e in edge_lengths]
        if any(t < 5 for t in targets):
            return None
        # a gate edge must be long enough to actually hold one of the gates we
        # own (and leave either 0 or >=5 units for the remaining wall run).
        gate_lengths = sorted(L for L, cnt in gates.items() if cnt > 0 and 5 <= L <= 30)

        def edge_fits_gate(t):
            return any(gl <= t and (t - gl == 0 or t - gl >= 5) for gl in gate_lengths)

        gate_edges: set[int] = set()
        if gate_lengths:
            # first gate: prefer the earliest edge that can hold a gate so the
            # loop is guaranteed at least one gate (required for validity).
            for i in range(n):
                if edge_fits_gate(targets[i]):
                    gate_edges.add(i)
                    break
            # second gate on a DIFFERENT face, for the G bonus -- only if we
            # have >=2 gates and another gate-capable edge exists. pick one far
            # from the first so they land on different faces.
            if sum(gates.values()) >= 2 and len(gate_edges) == 1:
                first = next(iter(gate_edges))
                cand = [
                    i
                    for i in range(n)
                    if i not in gate_edges and edge_fits_gate(targets[i])
                ]
                if cand:
                    gate_edges.add(max(cand, key=lambda i: abs(i - first)))

        # tile gate edges first so they claim gates before the shared wall/gate
        # pool is spent, then the rest hardest (longest) first.
        order = sorted(range(n), key=lambda i: (i not in gate_edges, -targets[i]))
        edge_runs = [None] * n
        used_straight = 0
        for ei in order:
            segs = self._tile_run(targets[ei], walls, gates, ei in gate_edges)
            if segs is None:
                return None
            used_straight += len(segs) - 1  # joins within this edge
            if used_straight > straight_budget:
                return None
            edge_runs[ei] = segs

        pieces: list[Piece] = []
        conns: list[Connector] = []
        for ei in range(n):
            for si, (ptype, L) in enumerate(edge_runs[ei]):
                if L < 5:
                    return None  # never emit a sub-5 piece (would crash Piece)
                pieces.append(Piece(ptype, L))
                conns.append(
                    corner_conns[ei]
                    if si == 0
                    else Connector(ConnectorType.STRAIGHT, False)
                )
        if len(conns) != len(pieces):
            return None
        return pieces, conns

    def _tile_run(self, target, walls, gates, want_gate):
        """Tile a run of total length `target` using the FEWEST pieces possible,
        consuming from the walls/gates dicts (mutates them only on success).
        Fewest pieces == fewest STRAIGHT connectors spent joining them, which is
        the real scarce resource when tracing a whole room. Places one gate first
        if `want_gate`. Returns list of (PieceType, length) or None."""
        out = []
        remaining = target
        if want_gate:
            # prefer the gate that leaves the easiest-to-tile remainder; longest
            # first to keep piece count low.
            gchoice = next(
                (
                    L
                    for L in sorted(gates, reverse=True)
                    if gates[L] > 0
                    and 5 <= L <= remaining
                    and (remaining - L == 0 or remaining - L >= 5)
                ),
                None,
            )
            if gchoice is None:
                return None
            gates[gchoice] -= 1
            out.append((PieceType.GATE, gchoice))
            remaining -= gchoice

        # fill the remainder with the FEWEST pieces, drawing from walls AND any
        # spare gates (a gate can stand in as a wall segment). track the source
        # so inventory is consumed correctly.
        pool = {}  # length -> [wall_count, gate_count]
        for L, c in walls.items():
            if c > 0:
                pool.setdefault(L, [0, 0])[0] += c
        for L, c in gates.items():
            if c > 0:
                pool.setdefault(L, [0, 0])[1] += c
        avail = sorted(pool, reverse=True)
        counts = {L: pool[L][0] + pool[L][1] for L in avail}
        best = [None]

        def solve(rem, acc):
            if best[0] is not None and len(acc) >= len(best[0]):
                return
            if rem == 0:
                best[0] = list(acc)
                return
            for L in avail:
                if counts[L] > 0 and L <= rem and (rem - L == 0 or rem - L >= 5):
                    counts[L] -= 1
                    acc.append(L)
                    solve(rem - L, acc)
                    acc.pop()
                    counts[L] += 1

        solve(remaining, [])
        if best[0] is None:
            return None
        for L in best[0]:
            # prefer spending a wall; use a gate only if no wall of that length
            if walls.get(L, 0) > 0:
                walls[L] -= 1
                out.append((PieceType.WALL, L))
            else:
                gates[L] -= 1
                out.append((PieceType.GATE, L))
        return out

    def _l_shape_fallback(self) -> Construction | None:
        """Try L-shaped enclosures oriented to the room's concave corners. Only
        called when no primary shape placed, so it can only improve a 0. An L
        has sides [X, Y, nx, ny, X-nx, Y-ny] with a reflex (270) corner at
        vertex 4; it fills an L / notched room better than a rectangle."""
        gl = sorted(
            L for L, n in self.inventory.gates.items() if n > 0 and 5 <= L <= 30
        )
        if not gl:
            return None

        anchors = self._concave_anchors()
        if not anchors:
            return None

        # a bounded set of L outer-box / notch sizes, built from inventory via
        # tiling. coarse steps keep this fast.
        best: tuple[float, Construction] | None = None
        for X in range(10, 31, 3):
            for Y in range(10, 31, 3):
                for nx in range(5, X - 4, 3):
                    for ny in range(5, Y - 4, 3):
                        sides = [X, Y, nx, ny, X - nx, Y - ny]
                        if any(s < 5 or s > 30 for s in sides):
                            continue
                        for two in (True, False):
                            built = self._l_pieces(sides, two)
                            if built is None:
                                continue
                            pieces, conns = built
                            area = X * Y - nx * ny
                            score = self._score_poly(area, sum(sides), two)
                            if score <= 0:
                                continue
                            cons = self._place_at_anchors(pieces, conns, anchors)
                            if cons is not None and (best is None or score > best[0]):
                                best = (score, cons)
        return best[1] if best else None

    def _concave_anchors(self) -> list[tuple[float, float]]:
        """Points near the room's reflex (concave) corners -- where an L's
        notch wants to sit -- plus the centroid as a fallback."""
        coords = list(self.room.polygon.exterior.coords)[:-1]
        n = len(coords)
        pts = []
        for i in range(n):
            ax, ay = coords[i]
            bx, by = coords[(i + 1) % n]
            cx, cy = coords[(i - 1) % n]
            # cross product of incoming->outgoing; sign flags a reflex corner
            cross = (ax - cx) * (by - ay) - (ay - cy) * (bx - ax)
            if cross < 0:  # reflex for a CCW polygon
                pts.append((ax, ay))
        cen = self.room.polygon.centroid
        pts.append((cen.x, cen.y))
        return pts

    def _place_at_anchors(self, pieces, conns, anchors) -> Construction | None:
        """Place a shape near each concave anchor over several headings."""
        for ax, ay in anchors:
            for hd in range(0, 360, 30):
                cons = self._at(pieces, conns, (ax, ay), hd)
                if validate_construction(cons, self.room, self.inventory).valid:
                    return cons
        return None

    # ======================================================================
    # Shape library + placement: fixed shapes (rect/tiled/octagon/L) and the
    # helpers that tile their pieces and place them in the room.
    # ======================================================================
    def _rectangles(self) -> list[tuple[float, list[Piece], list[Connector]]]:
        wl = sorted(
            L for L, n in self.inventory.walls.items() if n > 0 and 5 <= L <= 30
        )
        gl = sorted(
            L for L, n in self.inventory.gates.items() if n > 0 and 5 <= L <= 30
        )
        out = []
        seen = set()
        for w in gl:
            for h in wl:
                for two in (True, False):
                    key = (w, h, two)
                    if key in seen:
                        continue
                    seen.add(key)
                    pieces = self._rect_pieces(w, h, two)
                    if pieces is None:
                        continue
                    score = self._score_poly(w * h, 2 * (w + h), two)
                    conns = [Connector(ConnectorType.RIGHT)] * 4
                    out.append((score, pieces, conns))
        return out

    def _rect_pieces(self, w, h, two) -> list[Piece] | None:
        scratch = self.inventory.copy()
        layout = [(True, w), (False, h), (two, w), (False, h)]
        pieces = []
        for gate, L in layout:
            p = Piece(PieceType.GATE if gate else PieceType.WALL, L)
            if not scratch.take_piece(p):
                return None
            pieces.append(p)
        return pieces

    def _tiled_rectangles(self) -> list[tuple[float, list[Piece], list[Connector]]]:
        out = []
        # candidate side lengths: 5..30, both dimensions. keep it bounded so we
        # stay within the cpu budget. only emit when inventory can tile all four
        # sides and supply a gate.
        for w in range(5, 31):
            for h in range(w, 31):  # w <= h avoids duplicate orientations
                for two in (True, False):
                    built = self._tiled_rect_pieces(w, h, two)
                    if built is None:
                        continue
                    pieces, conns = built
                    score = self._score_poly(w * h, 2 * (w + h), two)
                    out.append((score, pieces, conns))
        return out

    def _tiled_rect_pieces(
        self, w: int, h: int, two: bool
    ) -> tuple[list[Piece], list[Connector]] | None:
        """A w x h rectangle whose four sides are tiled from inventory. A real
        corner (right connector) starts each side; straight connectors join the
        pieces within a side. Gate on side 0's first piece, and (if `two`) on
        the opposite side's first piece for the G bonus."""
        walls = dict(self.inventory.walls)
        gates = dict(self.inventory.gates)
        sides = [w, h, w, h]
        gate_sides = {0, 2} if two else {0}

        pieces: list[Piece] = []
        conns: list[Connector] = []
        for si, side_len in enumerate(sides):
            want_gate = si in gate_sides
            # if this side carries a gate, spend one gate on its first segment,
            # then tile the rest from walls.
            seg_lengths: list[int] = []
            first_is_gate = False
            if want_gate:
                # pick the largest gate <= side_len that leaves a tileable rest
                gchoice = None
                for gl in sorted(
                    (L for L, n in gates.items() if n > 0 and 5 <= L <= 30),
                    reverse=True,
                ):
                    if gl <= side_len and (side_len - gl == 0 or side_len - gl >= 5):
                        gchoice = gl
                        break
                if gchoice is None:
                    return None
                gates[gchoice] -= 1
                seg_lengths.append(gchoice)
                first_is_gate = True
                rest = side_len - gchoice
                if rest > 0:
                    tail = self._tile(rest, walls)
                    if tail is None:
                        return None
                    seg_lengths.extend(tail)
            else:
                tiled = self._tile(side_len, walls)
                if tiled is None:
                    return None
                seg_lengths = tiled

            for k, L in enumerate(seg_lengths):
                if k == 0:
                    conns.append(Connector(ConnectorType.RIGHT))  # the real corner
                    is_gate = first_is_gate
                else:
                    conns.append(Connector(ConnectorType.STRAIGHT))  # within the side
                    is_gate = False
                ptype = PieceType.GATE if is_gate else PieceType.WALL
                pieces.append(Piece(ptype, L))

        return pieces, conns

    def _octagons(self) -> list[tuple[float, list[Piece], list[Connector]]]:
        gl = sorted(
            L for L, n in self.inventory.gates.items() if n > 0 and 5 <= L <= 30
        )
        if not gl:
            return []
        out = []
        for s in sorted(
            {L for L, n in self.inventory.walls.items() if n > 0 and 5 <= L <= 30}
            | set(gl)
        ):
            # need 1 or 2 gates of length s + remaining walls of length s
            for two in (True, False):
                pieces = self._oct_pieces(s, two)
                if pieces is None:
                    continue
                # regular octagon with side s: area = 2(1+sqrt2) s^2, perim = 8s
                area = 2 * (1 + math.sqrt(2)) * s * s
                perim = 8 * s
                score = self._score_poly(area, perim, two)
                conns = [Connector(ConnectorType.DIAGONAL)] * 8
                out.append((score, pieces, conns))
        return out

    def _place(self, pieces, conns) -> Construction | None:
        # (Yuan) room-aware pass: snap an enclosure corner flush onto a room-edge
        # endpoint with its outgoing edge parallel to that wall. Wins on concave /
        # cornered / angled rooms where centering can't find a valid spot.
        pts = walk_points(Construction((0.0, 0.0), 0, pieces, conns))
        corners = [i for i, conn in enumerate(conns) if not conn.is_straight()]
        if corners:
            room_pts = list(self.room.polygon.exterior.coords)[:-1]
            # Sample <=16 room edges and <=8 enclosure corners: <= 256 tries.
            corners = corners[:: max(1, math.ceil(len(corners) / 8))]
            step = max(1, math.ceil(len(room_pts) / 16))
            for j in range(0, len(room_pts), step):
                a, b = room_pts[j], room_pts[(j + 1) % len(room_pts)]
                if a == b:
                    continue
                # Both directions handle either winding of the room boundary.
                for (ax, ay), (bx, by) in ((a, b), (b, a)):
                    room_angle = math.atan2(by - ay, bx - ax)
                    for i in corners:
                        x, y = pts[i]
                        nx, ny = pts[i + 1]
                        angle = room_angle - math.atan2(ny - y, nx - x)
                        c, s = math.cos(angle), math.sin(angle)
                        # Rotate the outgoing edge parallel to the room edge, then
                        # translate this enclosure vertex onto its endpoint.
                        start = (ax - (c * x - s * y), ay - (s * x + c * y))
                        cons = Construction(start, math.degrees(angle), pieces, conns)
                        if validate_construction(cons, self.room, self.inventory).valid:
                            return cons

        # fallback: centroid then grid of anchors, several headings
        cen = self.room.polygon.centroid
        for hd in HEADINGS:
            cons = self._at(pieces, conns, (cen.x, cen.y), hd)
            if validate_construction(cons, self.room, self.inventory).valid:
                return cons
        minx, miny, maxx, maxy = self.room.polygon.bounds
        xs = [minx + (maxx - minx) * i / (GRID_STEPS - 1) for i in range(GRID_STEPS)]
        ys = [miny + (maxy - miny) * i / (GRID_STEPS - 1) for i in range(GRID_STEPS)]
        for x in xs:
            for y in ys:
                if not self.room.polygon.contains(Point(x, y)):
                    continue
                for hd in HEADINGS:
                    cons = self._at(pieces, conns, (x, y), hd)
                    if validate_construction(cons, self.room, self.inventory).valid:
                        return cons
        return None

    def _at(self, pieces, conns, center, heading_deg) -> Construction:
        # place so the shape's centroid lands near `center`: walk from origin,
        # find centroid, then offset start. cheap + good enough for search.
        pts = [(0.0, 0.0)]
        hd = heading_deg
        # headings per piece from the turn sequence
        headings = [heading_deg]
        for i in range(1, len(pieces)):
            hd = hd + conns[i].turn_angle()
            headings.append(hd)
        for i, p in enumerate(pieces):
            th = math.radians(headings[i])
            x, y = pts[-1]
            pts.append((x + p.length * math.cos(th), y + p.length * math.sin(th)))
        cx = sum(x for x, _ in pts[:-1]) / len(pieces)
        cy = sum(y for _, y in pts[:-1]) / len(pieces)
        start = (center[0] - cx, center[1] - cy)
        return Construction(start, heading_deg, pieces, conns)

    def _l_pieces(
        self, sides: list[int], two: bool
    ) -> tuple[list[Piece], list[Connector]] | None:
        """Tile the six L sides from inventory. Reflex (270) corner at vertex 4,
        convex (90) elsewhere; straights within a side. Gate on side 0, second
        gate on side 2 (a different face) when `two`."""
        walls = dict(self.inventory.walls)
        gates = dict(self.inventory.gates)
        gate_sides = {0, 2} if two else {0}
        pieces: list[Piece] = []
        conns: list[Connector] = []
        for si, side_len in enumerate(sides):
            corner = Connector(ConnectorType.RIGHT, reflex=(si == 4))
            seg_lengths: list[int] = []
            first_is_gate = False
            if si in gate_sides:
                gchoice = None
                for gl in sorted(
                    (L for L, n in gates.items() if n > 0 and 5 <= L <= 30),
                    reverse=True,
                ):
                    if gl <= side_len and (side_len - gl == 0 or side_len - gl >= 5):
                        gchoice = gl
                        break
                if gchoice is None:
                    return None
                gates[gchoice] -= 1
                seg_lengths.append(gchoice)
                first_is_gate = True
                rest = side_len - gchoice
                if rest > 0:
                    tail = self._tile(rest, walls)
                    if tail is None:
                        return None
                    seg_lengths.extend(tail)
            else:
                tiled = self._tile(side_len, walls)
                if tiled is None:
                    return None
                seg_lengths = tiled
            for k, L in enumerate(seg_lengths):
                if k == 0:
                    conns.append(corner)
                    is_gate = first_is_gate
                else:
                    conns.append(Connector(ConnectorType.STRAIGHT))
                    is_gate = False
                ptype = PieceType.GATE if is_gate else PieceType.WALL
                pieces.append(Piece(ptype, L))
        return pieces, conns

    @staticmethod
    def _tile(target: int, avail: dict[int, int]) -> list[int] | None:
        """Break `target` into a run of piece lengths available in `avail`
        (greedy largest-first), leaving no stub shorter than 5. Only lengths in
        [5, 30] are usable. Mutates `avail`. Returns the lengths, or None."""
        sizes = sorted(
            (L for L, n in avail.items() if n > 0 and 5 <= L <= 30), reverse=True
        )
        out: list[int] = []
        rem = target
        while rem > 0:
            for L in sizes:
                if L <= rem and avail.get(L, 0) > 0 and (rem - L == 0 or rem - L >= 5):
                    avail[L] -= 1
                    out.append(L)
                    rem -= L
                    break
            else:
                return None
        return out

    def _oct_pieces(self, s, two) -> list[Piece] | None:
        scratch = self.inventory.copy()
        pieces = []
        for i in range(8):
            # gates on side 0 and side 4 (opposite -> different face) when two
            gate = (i == 0) or (two and i == 4)
            p = Piece(PieceType.GATE if gate else PieceType.WALL, s)
            if not scratch.take_piece(p):
                return None
            pieces.append(p)
        return pieces
