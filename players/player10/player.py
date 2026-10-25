import heapq
import math
import random
import signal
import sys
import time
from collections import Counter
from itertools import islice

import src.constants as c
from players.player0 import Player
from players.player10.assign import Assigner, Assignment
from players.player10.beam import free_shapes
from players.player10.cells import CellSearch, cell_loops
from players.player10.freeform import (
    MAX_FACES,
    Faces,
    Grower,
    Loop,
    area_of,
    corners,
)
from players.player10.placement import RoomSearch
from players.player10.polish import Polisher
from players.player10.raster import raster_loops
from players.player10.shapes import (
    CORNERS_NEEDED,
    HEADING_STEP_DEG,
    HEXAGON,
    L_SHAPE,
    MEASURES,
    OCTAGON,
    RECTANGLE,
    SYMMETRY_DEG,
    U_SHAPE,
    Shape,
    contains_smaller,
    free_shape,
    headed_faces,
    inner_radius,
    inner_rectangles,
    make_shape,
)
from src.enclosure import Construction, score_construction, validate_construction
from src.limits import CpuLimit
from src.pieces import Connector, ConnectorType, Piece, PieceType

# CPU seconds the search may use. The simulator allows 300; nothing can interrupt
# a player on Windows, so it stops itself with room to spare for the last step it
# is in the middle of. (The submission check runs with 60 s, so a long run shows
# there as a timeout warning, not a failure.)
TIME_BUDGET = 270.0
# how far into the budget each stage of the search may run; growing gets the rest
RECTANGLE_SHARE = 0.10
U_SHARE = 0.42  # the other families, sharing their time fairly, end here
SEGMENTED_SHARE = 0.50
# Filling the room with grid cells (raster.py): the cell sides tried, and the most
# time it may take. Only sides whose every multiple up to the face limit can be
# built are used, paired so that one-cell steps use the most plentiful pieces.
RASTER_SIZES = (5, 6, 7, 8, 9, 10, 12, 15)
RASTER_PAIRS = 24
RASTER_SHARE = 0.1  # of the whole budget, at most
# choosing grid cells under the stock's budget (cells.py), its share of the budget
CELLS_SHARE = 0.06
CELL_STEPS = (5,)
CELLS_SEED = 4444  # the same scenario gives the same search every run
# only in rooms big enough, or with corners enough, for a block of cells to
# beat what growing finds: in a small room it only takes growing's time
CELLS_MIN_AREA = 1000.0
CELLS_MIN_RIGHTS = 100
RASTER_MAX_FACES = 900  # piece assignment walks the faces one level deep each
# Large, highly articulated rooms with abundant stock benefit from comparing all
# useful grid sizes and room-derived orientations.  This remains conditional so
# ordinary scenarios retain the short raster pass.
DENSE_RASTER_SECONDS = 80.0
DENSE_RASTER_MIN_VERTICES = 32
DENSE_RASTER_MIN_RIGHTS = 128
SMALL_STOCK_MAX_PIECES = 20
SMALL_STOCK_MAX_LENGTH = 180
SMALL_STOCK_WIDTHS = (500, 2000, 8000, 24000)
# Loops walked along the walls (beam.py): few win outright, but the shapes they find
# are starting points for growing that the shape families would never give.
BEAM_SECONDS = 8.0
# Growing runs in two parts. The first ends here; then, where the best so far
# leaves a large gap, a deeper walk along the walls gets its turn (small stocks,
# or rooms barely filled), and growing resumes from the best loop with what is left.
FIRST_GROW_SHARE = 0.55
GROW_SHARE = 0.96  # growing the best shapes found so far (freeform.py)
# polishing the best rectilinear shape (polish.py) gets the rest
# growing asks for pieces for thousands of loops, so each ask gets less effort
GROW_ASSIGN_NODES = 3000
MAX_SEEDS = 6
# the best rectangle is also grown from other places it fits, this many at most
EXTRA_SPOTS = 2
# rectangles tried, biggest first, for a seed lined up with each wall direction
WALL_SEED_SIZES = 40
GRID_SEEDS = 2  # grid outlines grown, besides the shapes
RASTER_FIT_CORNERS = True  # drop cells until a grid outline fits the corner stock
SPOTS_TRIED = 24
BOUNDARY_WIDTHS = (160, 640, 1600)
BOUNDARY_MAX_FACES = 40
BOUNDARY_STARTS = 16
SEGMENTED_NOTCH_LENGTHS = 6
SEGMENTED_TRIPLES = 400
FALLBACK_BUDGET = 3.0
FALLBACK_SHAPES = 40
RECENT_FAILURES = 400
SHRINK_STEPS = 4
POLISH_TRIES = 300  # candidates tried per round before giving up

Candidate = tuple[float, str, tuple[int, ...]]


def segmented_rectangle(a: int, middle: int, b: int, d: int, h: int) -> Shape:
    """A long rectangle with mirrored diagonal bays in its horizontal edges."""
    headed = (
        (0, a),
        (1, d),
        (0, middle),
        (7, d),
        (0, b),
        (2, h),
        (4, b),
        (5, d),
        (4, middle),
        (3, d),
        (4, a),
        (6, h),
    )
    draft = free_shape(headed)
    horizontal = a + middle + b
    area = (horizontal + math.sqrt(2.0) * d) * h - math.sqrt(2.0) * middle * d - d * d
    perimeter = 2.0 * (horizontal + h + 2 * d)
    return Shape(
        draft.family,
        draft.params,
        draft.faces,
        draft.turns,
        area,
        perimeter,
    )


def cpu_limit() -> float:
    """The CPU seconds the simulator gives this run: the limit of its running
    timer where one can be read (Unix), else --cpu-limit from the command line,
    else the simulator's default."""
    try:
        handler = (
            signal.getsignal(signal.SIGPROF) if hasattr(signal, "SIGPROF") else None
        )
        owner = getattr(handler, "__self__", None)
        if isinstance(owner, CpuLimit) and math.isfinite(owner.seconds):
            return float(owner.seconds)
    except (OSError, ValueError):
        pass
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        flag, _, value = arg.partition("=")
        if flag == "--cpu-limit":
            try:
                seconds = float(value if value else args[i + 1])
            except (ValueError, IndexError):
                continue
            if math.isfinite(seconds) and seconds > 0:
                return seconds
    return float(c.CPU_TIME_LIMIT_SECONDS)


class Player10(Player):
    """Group 10: the best-scoring rectangle, L, U, hexagon or octagon the inventory
    can build, placed against the room's own walls, then grown face by face into
    a free-form loop that fills the room."""

    def build_enclosure(self) -> Construction | None:
        self._best: Construction | None = None
        self._best_score = 0.0  # a valid enclosure can score below 0; None is 0
        self._best_shape: Shape | None = None
        self._best_start: tuple[float, float] = (0.0, 0.0)
        self._best_heading = 0.0
        try:
            self._build()
        except Exception as e:  # noqa: BLE001 -- a crash would end the whole run
            self.note = f"search stopped by {type(e).__name__}: {e}"
        return self._best

    def _build(self) -> None:
        started = time.process_time()
        # the simulator's limit may be lower than 300 (it is set per run); leave a
        # tenth of it, and at least 20 s, for the step in hand when time runs out
        limit = cpu_limit()
        budget = min(TIME_BUDGET, limit - max(0.1 * limit, 20.0), limit * 0.9)
        budget = max(budget, 1.0)
        self._budget = budget
        weights = self.weights
        self._assigner = Assigner(self.inventory, want_second_gate=weights.G > 0)
        if self._assigner.gates == 0:
            self.note = "no usable gate in the inventory -- not attempting a build"
            return
        self._search_room = RoomSearch(self.room)
        self._failures: dict[str, list[tuple[int, ...]]] = {}
        self._fitting_rectangles: list[tuple[int, int]] = []
        self._seeds: list[Loop] = []
        self._grid_seeds: list[Loop] = []
        self._rectangle: tuple | None = None
        bonus = weights.G if len(self._assigner.targets) == 2 else 0.0
        self._bonus = bonus
        dense_raster = self._dense_raster_worthwhile()
        self._grow_max_faces = RASTER_MAX_FACES if dense_raster else MAX_FACES
        stock = self.inventory.connectors
        families = [
            family
            for family, needs in CORNERS_NEEDED.items()
            if all(stock.get(kind, 0) >= n for kind, n in needs.items())
        ]

        # cheapest families first, so there is always something to fall back on
        # and a score for the dearer searches to beat. Octagons are slow to rule
        # out; U-shapes are many, and only pay off where nothing else fits.
        if RECTANGLE in families:
            self._deadline = started + budget * RECTANGLE_SHARE
            self._search(self._candidates(RECTANGLE, bonus))
        # the other families share their time fairly: a family whose big shapes
        # keep failing to fit can't starve one whose best shape is next in line
        others = [f for f in (L_SHAPE, OCTAGON, HEXAGON, U_SHAPE) if f in families]
        if others:
            self._deadline = started + budget * U_SHARE
            self._search_fair({f: self._candidates(f, bonus) for f in others})

        self._deadline = started + budget * SEGMENTED_SHARE
        self._segmented_rectangles(bonus)

        self._raster(
            min(
                started + budget * GROW_SHARE,
                time.process_time() + budget * RASTER_SHARE,
            )
        )

        self._cells(
            min(
                started + budget * GROW_SHARE,
                time.process_time() + budget * CELLS_SHARE,
            )
        )

        if dense_raster:
            self._raster(
                min(
                    started + budget * 0.58,
                    time.process_time() + DENSE_RASTER_SECONDS,
                ),
                headings=self._search_room.base_headings(90),
                pair_limit=None,
            )

        # loops walked along the walls: they only have to beat the best shape so
        # far, and every one that is valid also seeds growing
        self._deadline = time.process_time() + BEAM_SECONDS
        self._freeform(bonus)

        self._grow(started + budget * FIRST_GROW_SHARE)

        # a deeper walk along the walls where the best so far leaves a large gap
        extra_end = started + budget * GROW_SHARE
        if not dense_raster and self._small_stock_search_worthwhile():
            self._small_stock_boundary(
                extra_end, widths=SMALL_STOCK_WIDTHS, diverse_beam=True
            )
        elif not dense_raster and self._should_explore():
            self._extended_boundary(extra_end)

        # the rest of the time grows again, from the best loop found so far
        self._grow(started + budget * GROW_SHARE, from_best=True)

        self._deadline = started + budget
        self._polish(bonus)

        if self._best is None and not self.note:
            self.note = "nothing the inventory can build fits and scores above 0"

    # ------------------------------------------------------------------ grow

    def _grow(self, deadline: float, from_best: bool = False) -> None:
        """Grow the shapes found so far into free-form loops (see freeform.py).
        With `from_best`, the best loop found so far leads the seeds."""
        if not self._seeds or time.process_time() >= deadline:
            return
        stock = self.inventory.connectors
        grower = Grower(
            region=self._search_room.region,
            room_corners=self._search_room.corners,
            buildable=set(self._assigner.lengths),
            rights=stock.get(ConnectorType.RIGHT, 0),
            diagonals=stock.get(ConnectorType.DIAGONAL, 0),
            total_length=self._assigner.total_length,
            score=lambda area, perimeter: self._score(area, perimeter) + self._bonus,
            accept=self._accept_loop,
            can_build=lambda lengths: self._assigner.assign(lengths) is not None,
            max_faces=self._grow_max_faces,
        )
        self._assigner.max_nodes = GROW_ASSIGN_NODES
        seeds = sorted(self._seeds, key=lambda loop: -loop.score)[:MAX_SEEDS]
        # grid outlines join the shapes, rather than pushing them out: they score
        # well at the start, but the shapes often grow further
        seeds += sorted(self._grid_seeds, key=lambda loop: -loop.score)[:GRID_SEEDS]
        seeds += self._other_spots()
        seeds += self._along_walls()
        if from_best and self._best_shape is not None:
            best = Loop(
                self._best_start,
                self._best_heading,
                headed_faces(self._best_shape),
                self._best_score,
            )
            seeds.insert(0, best)
        # the charge for a side over the limit starts below what a unit of
        # growth earns, so spreading out comes first and repairs second
        grower.grow(seeds, deadline, first_charge=max(self.weights.A, 0.1))

    def _along_walls(self) -> list[Loop]:
        """A rectangle lined up with each of the room's main wall directions.
        Growing keeps the angle of the shape it starts from, so without these it
        would only ever follow the walls the first shape happened to line up with."""
        if self._rectangle is None:
            return []
        best, _, _ = self._rectangle
        used = {round(loop.heading % 90, 2) for loop in self._seeds}
        sizes = sorted(
            {
                (a, b)
                for a in self._assigner.lengths
                for b in self._assigner.lengths
                if b <= a
            },
            key=lambda p: -p[0] * p[1],
        )
        sizes = [p for p in sizes if p[0] * p[1] <= best.params[0] * best.params[1]]
        seeds = []
        for heading in self._search_room.base_headings(90):
            if round(heading % 90, 2) in used:
                continue
            for a, b in sizes[:WALL_SEED_SIZES]:
                shape = make_shape(RECTANGLE, (a, b))
                assignment = self._assigner.assign(shape.faces)
                if assignment is None:
                    continue
                spots = self._search_room.place(
                    shape.vertices(), 90, 180, limit=1, only=[heading, heading + 90]
                )
                if not spots:
                    continue
                start, angle = spots[0]
                construction = self._construction(shape, assignment, start, angle)
                result = validate_construction(construction, self.room, self.inventory)
                if result.valid:
                    score = score_construction(result, self.weights)
                    seeds.append(Loop(start, angle, headed_faces(shape), score))
                    break
        return seeds

    def _should_explore(self) -> bool:
        """Whether a valid result still leaves a credible large score gap.

        The isoperimetric bound is intentionally optimistic.  It is used only to
        decide whether more search is worthwhile; final selection still uses the
        official score. Requiring unused perimeter and enough corner connectors
        avoids spending two extra minutes on stock-limited rooms.
        """
        if self._best_shape is None or self.weights.A <= 0:
            return False
        area = self._best_shape.area
        perimeter = self._best_shape.perimeter
        if self._assigner.total_length <= 0:
            return False
        corners_available = sum(
            self.inventory.connectors.get(kind, 0)
            for kind in (ConnectorType.RIGHT, ConnectorType.DIAGONAL)
        )
        # A wide search needs enough corners to express a materially more complex
        # outline. With fewer than 32, the 40-face global walk cannot realize its
        # intended complexity and empirical hard cases are connector-limited.
        if corners_available < 32 or perimeter >= 0.85 * self._assigner.total_length:
            return False
        area_bound = min(
            self._search_room.area,
            self._assigner.total_length**2 / (4.0 * math.pi),
        )
        if area >= 0.40 * area_bound:
            return False
        optimistic = (
            c.BASELINE_SCORE
            + self.weights.A * area_bound
            + self.weights.C * perimeter
            + self._bonus
        )
        required_gain = max(100.0, 0.10 * max(self._best_score, 1.0))
        return optimistic >= self._best_score + required_gain

    def _dense_raster_worthwhile(self) -> bool:
        """Whether a broader raster sweep can express the room and be supplied.

        A large boundary needs hundreds of corners and at least roughly its own
        perimeter in material.  Testing these structural conditions avoids a
        scenario-name special case and prevents the extended pass from slowing
        ordinary inventories.
        """
        vertices = len(self.room.get_boundary_points())
        rights = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        return (
            vertices >= DENSE_RASTER_MIN_VERTICES
            and rights >= DENSE_RASTER_MIN_RIGHTS
            and self._assigner.total_length >= self.room.polygon.length
        )

    def _small_stock_search_worthwhile(self) -> bool:
        """Whether exact stock pruning makes a deeper geometric beam worthwhile."""
        if self._best_shape is None or self.weights.A <= 0:
            return False
        piece_count = sum(self._assigner.stock.values())
        corner_count = sum(
            self.inventory.connectors.get(kind, 0)
            for kind in (ConnectorType.RIGHT, ConnectorType.DIAGONAL)
        )
        if (
            piece_count > SMALL_STOCK_MAX_PIECES
            or self._assigner.total_length > SMALL_STOCK_MAX_LENGTH
            or len(self.room.get_boundary_points()) < 12
            or corner_count < 8
            or self._best_shape.perimeter < 0.75 * self._assigner.total_length
        ):
            return False
        area_bound = min(
            self._search_room.area,
            self._assigner.total_length**2 / (4.0 * math.pi),
        )
        return self._best_shape.area < 0.80 * area_bound

    def _small_stock_boundary(
        self,
        deadline: float,
        *,
        widths=SMALL_STOCK_WIDTHS,
        diverse_starts=False,
        diverse_beam=False,
    ) -> None:
        """Search complex small inventories after pruning infeasible face prefixes."""
        corners_available = sum(
            self.inventory.connectors.get(kind, 0)
            for kind in (ConnectorType.RIGHT, ConnectorType.DIAGONAL)
        )
        max_faces = min(SMALL_STOCK_MAX_PIECES, corners_available)
        for shape, start, heading in free_shapes(
            self._search_room,
            self._assigner,
            self.inventory,
            self.weights,
            deadline,
            widths=widths,
            max_faces=max_faces,
            diverse_starts=diverse_starts,
            diverse_beam=diverse_beam,
        ):
            if (
                self._score(shape.area, shape.perimeter) + self._bonus
                <= self._best_score
            ):
                continue
            assignment = self._assigner.assign(shape.faces)
            if assignment is not None:
                self._try_fixed(shape, assignment, start, heading)

    def _extended_boundary(self, deadline: float) -> None:
        """Search longer room-following loops from the most useful wall anchors."""
        for shape, start, heading in free_shapes(
            self._search_room,
            self._assigner,
            self.inventory,
            self.weights,
            deadline,
            widths=BOUNDARY_WIDTHS,
            max_faces=BOUNDARY_MAX_FACES,
            max_starts=BOUNDARY_STARTS,
        ):
            if (
                self._score(shape.area, shape.perimeter) + self._bonus
                <= self._best_score
            ):
                continue
            assignment = self._assigner.assign(shape.faces)
            if assignment is not None:
                self._try_fixed(shape, assignment, start, heading)

    def _other_spots(self) -> list[Loop]:
        """The best rectangle again, at other places it fits: a room with
        several open areas gets a seed growing in each."""
        if self._rectangle is None:
            return []
        shape, first, score = self._rectangle
        a, b = shape.params

        def centre(start, heading):
            t = math.radians(heading)
            return (
                start[0] + (a * math.cos(t) - b * math.sin(t)) / 2.0,
                start[1] + (a * math.sin(t) + b * math.cos(t)) / 2.0,
            )

        taken = [centre(*first)]
        spots = []
        for start, heading in self._placements(shape, limit=SPOTS_TRIED):
            here = centre(start, heading)
            if all(math.dist(here, other) >= max(a, b) for other in taken):
                taken.append(here)
                spots.append(Loop(start, heading, headed_faces(shape), score))
                if len(spots) >= EXTRA_SPOTS:
                    break
        return spots

    def _accept_loop(
        self, start: tuple[float, float], heading: float, faces: Faces
    ) -> float | None:
        """The true score of a loop, if the inventory can build it and the
        simulator accepts it; the best one seen is kept."""
        shape = free_shape(faces)
        assignment = self._assigner.assign(shape.faces)
        if assignment is None:
            return None
        # a loop's faces are headed from the loop's own heading, and its first
        # face need not point along it
        first = heading + 45.0 * faces[0][0]
        construction = self._construction(shape, assignment, start, first)
        result = validate_construction(construction, self.room, self.inventory)
        if not result.valid:
            return None
        score = score_construction(result, self.weights)
        # ``free_shape`` carries topology but deliberately leaves area at zero.
        # Retain the validator's exact geometry so later budget decisions compare
        # the real incumbent against the room rather than treating every grown
        # result as an empty polygon.
        shape = Shape(
            shape.family,
            shape.params,
            shape.faces,
            shape.turns,
            result.area,
            result.perimeter,
        )
        self._keep(construction, score, shape, start, first)
        return score

    # ------------------------------------------------------------ candidates

    def _candidates(self, family: str, bonus: float) -> list[Candidate]:
        """Every shape of a family worth trying, with the most it could score."""
        lengths = self._assigner.lengths
        usable = set(lengths)
        if family == RECTANGLE:
            params = [(a, b) for a in lengths for b in lengths if b <= a]
        elif family == OCTAGON:
            # p is the longest side: the other choices are the same octagon turned
            params = [
                (p, r, q, t)
                for p in lengths
                for r in lengths
                if r <= p
                for q in lengths
                if q <= p
                for t in lengths
                if t <= p
            ]
        elif family == HEXAGON:
            params = [(a, d1, d2) for a in lengths for d1 in lengths for d2 in lengths]
        elif family == U_SHAPE:
            params = [
                (left, gap, right, height, leg)
                for left in lengths
                for gap in lengths
                for right in lengths
                if left + gap + right in usable
                for height in lengths
                for leg in lengths
                if height - leg >= c.MIN_WALL_LENGTH
            ]
        else:
            params = [
                (a, b, cut, d)
                for a in lengths
                for cut in lengths
                if a - cut in usable
                for b in lengths
                for d in lengths
                if b + d in usable
            ]
        measure = MEASURES[family]
        out = []
        for p in params:
            area, perimeter = measure(*p)
            if area > self._search_room.area + c.TOL:
                continue
            score = self._score(area, perimeter) + bonus
            if score > self._best_score:
                out.append((score, family, p))
        return out

    def _score(self, area: float, perimeter: float) -> float:
        return c.BASELINE_SCORE + self.weights.A * area + self.weights.C * perimeter

    def _segmented_rectangles(self, bonus: float) -> None:
        """Long rectangles whose mirrored diagonal bays split overlong edges.

        With notch length ``d``, the two diagonal faces consume only
        ``d / sqrt(2)`` of room depth.  This permits a nearly full-width enclosure
        in a thin room while every individual linear face remains at most 30.
        """
        low_x, low_y, high_x, high_y = self.room.polygon.bounds
        width, height = high_x - low_x, high_y - low_y
        if min(width, height) <= 0 or max(width, height) / min(width, height) < 2.5:
            return
        if self._search_room.area < 0.75 * width * height:
            return
        stock = self.inventory.connectors
        if (
            stock.get(ConnectorType.RIGHT, 0) < 4
            or stock.get(ConnectorType.DIAGONAL, 0) < 8
        ):
            return
        lengths = self._assigner.lengths
        triples = sorted(
            (
                (a + middle + b, a, middle, b)
                for a in lengths
                for middle in lengths
                for b in lengths
                if a >= b
            ),
            reverse=True,
        )[:SEGMENTED_TRIPLES]
        ranked = []
        for d in lengths[:SEGMENTED_NOTCH_LENGTHS]:
            for h in lengths:
                if h <= math.sqrt(2.0) * d + c.TOL:
                    continue
                for horizontal, a, middle, b in triples:
                    potential_area = (horizontal + math.sqrt(2.0) * d) * h - d * d
                    if potential_area > self._search_room.area + c.TOL:
                        continue
                    area = (
                        (horizontal + math.sqrt(2.0) * d) * h
                        - math.sqrt(2.0) * middle * d
                        - d * d
                    )
                    perimeter = 2.0 * (horizontal + h + 2 * d)
                    # Rank a seed by the surrounding long rectangle: the grower
                    # can push the recessed middle faces back outward.
                    potential = self._score(potential_area, perimeter) + bonus
                    if potential > self._best_score:
                        exact = self._score(area, perimeter) + bonus
                        ranked.append((potential, exact, a, middle, b, d, h))
        ranked.sort(reverse=True)

        for upper, _, a, middle, b, d, h in ranked:
            if time.process_time() >= self._deadline or upper <= self._best_score:
                return
            shape = segmented_rectangle(a, middle, b, d, h)
            assignment = self._assigner.assign(shape.faces)
            if assignment is None:
                continue
            for start, heading in self._search_room.place(
                shape.vertices(), 45, 180, limit=3
            ):
                if self._try_fixed(shape, assignment, start, heading):
                    break

    # ---------------------------------------------------------------- search

    def _search(self, candidates: list[Candidate]) -> None:
        """Try candidates from the highest score down; the first that can be built
        and placed is the best of them."""
        # (-score, tie-break, family, params, score is exact)
        heap = [(-s, i, f, p, False) for i, (s, f, p) in enumerate(candidates)]
        heapq.heapify(heap)
        counter = len(heap)
        timed_out = False

        while heap:
            if time.process_time() > self._deadline:
                timed_out = True
                break
            negative, _, family, params, exact = heapq.heappop(heap)
            if -negative <= self._best_score:
                return
            if self._cannot_fit(family, params):
                continue
            shape = make_shape(family, params)
            assignment = self._assigner.assign(shape.faces)
            if assignment is None:
                continue
            if not exact:
                score = self._score(shape.area, shape.perimeter)
                if assignment.gate_faces >= 2:
                    score += self.weights.G
                if score < -negative - 1e-9:
                    # no second gate after all: it is worth less than it was
                    # ranked at, so let it wait its turn
                    counter += 1
                    heapq.heappush(heap, (-score, counter, family, params, True))
                    continue
            if self._try(shape, assignment):
                return
            self._note_failure(family, params)

        if timed_out and self._best is None:
            # ran out of time on big shapes: settle for the smallest ones
            deadline = time.process_time() + FALLBACK_BUDGET
            for _, _, family, params, _ in heapq.nlargest(FALLBACK_SHAPES, heap):
                if time.process_time() > deadline:
                    break
                shape = make_shape(family, params)
                assignment = self._assigner.assign(shape.faces)
                if assignment is not None:
                    self._try(shape, assignment)

    def _search_fair(self, by_family: dict[str, list[Candidate]]) -> None:
        """Like `_search`, for several families at once: each family's shapes
        are tried from its highest score down, and the next turn always goes to
        the family that has had the least time so far. A family is done when its
        best shape is placed or nothing of it could beat the best so far."""
        heaps = {}
        for family, candidates in by_family.items():
            heap = [(-s, i, f, p, False) for i, (s, f, p) in enumerate(candidates)]
            if heap:
                heapq.heapify(heap)
                heaps[family] = heap
        spent = dict.fromkeys(heaps, 0.0)
        counter = sum(len(h) for h in heaps.values())
        while heaps and time.process_time() <= self._deadline:
            family = min(heaps, key=lambda f: spent[f])
            heap = heaps[family]
            began = time.process_time()
            done = self._search_step(heap, counter)
            counter += 1
            spent[family] += time.process_time() - began
            if done or not heap:
                del heaps[family]

    def _search_step(self, heap: list, counter: int) -> bool:
        """One candidate off the heap. True when the family is finished: its
        best shape is placed, or its next could not beat the best so far."""
        negative, _, family, params, exact = heapq.heappop(heap)
        if -negative <= self._best_score:
            return True
        if self._cannot_fit(family, params):
            return False
        shape = make_shape(family, params)
        assignment = self._assigner.assign(shape.faces)
        if assignment is None:
            return False
        if not exact:
            score = self._score(shape.area, shape.perimeter)
            if assignment.gate_faces >= 2:
                score += self.weights.G
            if score < -negative - 1e-9:
                heapq.heappush(heap, (-score, counter, family, params, True))
                return False
        if self._try(shape, assignment):
            return True
        self._note_failure(family, params)
        return False

    def _cannot_fit(self, family: str, params: tuple[int, ...]) -> bool:
        """Cheap reasons a shape can't be placed, tried before the real search."""
        if inner_radius(family, params) > self._search_room.inradius:
            return True
        recent = self._failures.get(family, [])[-RECENT_FAILURES:]
        if any(contains_smaller(family, params, small) for small in recent):
            return True
        return not all(
            self._rectangle_fits(max(r), min(r))
            for r in inner_rectangles(family, params)
        )

    def _rectangle_fits(self, a: int, b: int) -> bool:
        """Whether an a x b rectangle (a >= b) fits anywhere in the room, whether
        or not the inventory can build it. Remembered, and inferred from other
        rectangles where possible."""
        if any(fa >= a and fb >= b for fa, fb in self._fitting_rectangles):
            return True
        failures = self._failures.setdefault(RECTANGLE, [])
        if any(a >= fa and b >= fb for fa, fb in failures):
            return False
        if inner_radius(RECTANGLE, (a, b)) > self._search_room.inradius:
            return False
        fits = self._placements(make_shape(RECTANGLE, (a, b)), limit=1) != []
        (self._fitting_rectangles if fits else failures).append((a, b))
        return fits

    def _try(self, shape: Shape, assignment: Assignment) -> bool:
        """Place the shape; keep it if it is valid and beats the best so far."""
        for start, heading in self._placements(shape):
            if self._try_fixed(shape, assignment, start, heading):
                return True
        return False

    def _try_fixed(
        self,
        shape: Shape,
        assignment: Assignment,
        start: tuple[float, float],
        heading: float,
    ) -> bool:
        """Build at a known start and heading; True if the simulator accepts it.
        Every accepted shape is also kept as a seed for growing."""
        construction = self._construction(shape, assignment, start, heading)
        # the simulator's own judge: only what it accepts is ever returned
        result = validate_construction(construction, self.room, self.inventory)
        if not result.valid:
            return False
        score = score_construction(result, self.weights)
        self._seeds.append(Loop(start, heading, headed_faces(shape), score))
        if shape.family == RECTANGLE:
            self._rectangle = (shape, (start, heading), score)
        self._keep(construction, score, shape, start, heading)
        return True

    def _keep(self, construction, score, shape, start, heading) -> None:
        """Remember the best construction, and the shape it came from for the
        polish pass."""
        if score > self._best_score:
            self._best, self._best_score = construction, score
            self._best_shape, self._best_start, self._best_heading = (
                shape,
                start,
                heading,
            )
            right = sum(abs(turn) == 90 for turn in shape.turns)
            diagonal = sum(abs(turn) == 45 for turn in shape.turns)
            self.note = (
                f"{shape.describe()}: score {score:.1f}, area {shape.area:.1f}, "
                f"perimeter {shape.perimeter:.1f}, {len(construction.pieces)} pieces, "
                f"corners {right}R/{diagonal}D"
            )

    def _raster(
        self,
        deadline: float,
        *,
        headings: list[float] | None = None,
        pair_limit: int | None = RASTER_PAIRS,
    ) -> None:
        """Outlines of the room filled with grid cells (see raster.py)."""
        lengths = set(self._assigner.lengths)
        top = int(c.MAX_FACE_LENGTH)
        sides = [
            s
            for s in RASTER_SIZES
            if all(k * s in lengths for k in range(1, top // s + 1))
        ]
        rights = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        if not sides or rights < 4:
            return
        # a one-cell step needs a single piece of exactly that length
        single = {
            s: self.inventory.walls.get(s, 0) + self.inventory.gates.get(s, 0)
            for s in sides
        }
        pairs = sorted(
            ((w, h) for w in sides for h in sides),
            key=lambda p: (-min(single[p[0]], single[p[1]]), p[0] * p[1]),
        )
        if pair_limit is not None:
            pairs = pairs[:pair_limit]
        if headings is None:
            headings = self._search_room.base_headings(90)[:2]
            main = self._search_room.main_direction()
            if all(abs((main - h + 45) % 90 - 45) > 0.5 for h in headings):
                headings = headings + [main]
        for start, heading, faces in raster_loops(
            self._search_room.region,
            self.room.polygon,
            headings,
            pairs,
            deadline,
            corner_budget=min(rights, RASTER_MAX_FACES) if RASTER_FIT_CORNERS else None,
        ):
            perimeter = sum(n for _, n in faces)
            if (
                len(faces) > min(rights, RASTER_MAX_FACES)
                or perimeter > self._assigner.total_length
            ):
                continue
            area = area_of(corners((0.0, 0.0), faces))
            if self._score(area, perimeter) + self._bonus <= self._best_score:
                continue
            # quick count: sides only one piece can make, against that piece's stock
            need = Counter(n for _, n in faces if n in single)
            if any(count > single[n] for n, count in need.items()):
                continue
            score = self._accept_loop(start, heading, tuple(faces))
            # a grid outline small enough to grow is a starting point for growing:
            # it fills the room, and growing can add the diagonals it lacks
            if score is not None and len(faces) <= self._grow_max_faces:
                self._grid_seeds.append(Loop(start, heading, tuple(faces), score))

    def _cells(self, deadline: float) -> None:
        """Outlines of grid cells chosen under the stock's budget (see cells.py):
        where wall or corners run short, the best block of cells is often not
        the whole room trimmed down."""
        rights = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        lengths = set(self._assigner.lengths)
        steps = [s for s in CELL_STEPS if s in lengths] or [min(lengths)]
        if rights < 4:
            return
        if self.room.polygon.area < CELLS_MIN_AREA and rights < CELLS_MIN_RIGHTS:
            return
        search = CellSearch(
            lengths,
            min(rights, RASTER_MAX_FACES),
            self._assigner.total_length,
            self.weights.A,
            self.weights.C,
            lambda sides: self._assigner.assign(sides) is not None,
            random.Random(CELLS_SEED),
        )
        for start, heading, faces in cell_loops(
            self._search_room.region,
            self.room.polygon,
            self._search_room.base_headings(90),
            steps,
            search,
            deadline,
        ):
            perimeter = sum(n for _, n in faces)
            area = area_of(corners((0.0, 0.0), faces))
            if self._score(area, perimeter) + self._bonus <= self._best_score:
                continue
            score = self._accept_loop(start, heading, faces)
            if score is not None and len(faces) <= self._grow_max_faces:
                self._grid_seeds.append(Loop(start, heading, faces, score))

    def _freeform(self, bonus: float) -> None:
        """Free-form polygons from the beam search, best-ranked first."""
        for shape, start, heading in free_shapes(
            self._search_room,
            self._assigner,
            self.inventory,
            self.weights,
            self._deadline,
        ):
            # the most it could score; skip anything already beaten
            if self._score(shape.area, shape.perimeter) + bonus <= self._best_score:
                continue
            assignment = self._assigner.assign(shape.faces)
            if assignment is not None:
                self._try_fixed(shape, assignment, start, heading)

    def _polish(self, bonus: float) -> None:
        """Grow and trim the best shape by adding or cutting rectangles, keeping
        each change the simulator accepts that scores higher, until none does."""
        if self._best_shape is None:
            return
        polisher = Polisher(
            self._search_room.region,
            self._assigner,
            self.inventory,
            lambda area, perimeter: self._score(area, perimeter) + bonus,
        )
        while time.process_time() < self._deadline:
            before = self._best_score
            for shape, start, heading in islice(
                polisher.neighbours(
                    self._best_shape,
                    self._best_start,
                    self._best_heading,
                    before,
                    self._deadline,
                ),
                POLISH_TRIES,
            ):
                assignment = self._assigner.assign(shape.faces)
                if assignment is not None:
                    self._try_fixed(shape, assignment, start, heading)
                if self._best_score > before:
                    break
            if self._best_score <= before:
                return

    def _placements(self, shape: Shape, limit: int = 3) -> list:
        return self._search_room.place(
            shape.vertices(),
            HEADING_STEP_DEG[shape.family],
            SYMMETRY_DEG[shape.family],
            limit,
        )

    def _note_failure(self, family: str, params: tuple[int, ...]) -> None:
        """Remember that a shape fits nowhere. Shapes are tried biggest first, so a
        failure on its own rules out little; a much smaller version that also
        fails rules out everything that could hold it."""
        failures = self._failures.setdefault(family, [])
        smallest = params
        if family in (RECTANGLE, OCTAGON, HEXAGON):
            fitting, failing = 0.0, 1.0
            for _ in range(SHRINK_STEPS):
                scale = (fitting + failing) / 2.0
                shrunk = tuple(max(c.MIN_WALL_LENGTH, round(scale * p)) for p in params)
                if shrunk == smallest:
                    break
                if family == RECTANGLE:
                    fits = self._rectangle_fits(max(shrunk), min(shrunk))
                else:
                    fits = self._placements(make_shape(family, shrunk), 1) != []
                if fits:
                    fitting = scale
                else:
                    failing, smallest = scale, shrunk
        if smallest not in failures:
            failures.append(smallest)

    @staticmethod
    def _construction(
        shape: Shape,
        assignment: Assignment,
        start: tuple[float, float],
        heading: float,
    ) -> Construction:
        pieces, connectors = [], []
        for face, keys in enumerate(assignment.faces):
            for i, (is_gate, length) in enumerate(keys):
                kind = PieceType.GATE if is_gate else PieceType.WALL
                pieces.append(Piece(kind, length))
                connectors.append(
                    shape.corner(face) if i == 0 else Connector(ConnectorType.STRAIGHT)
                )
        return Construction(start, heading, pieces, connectors)
