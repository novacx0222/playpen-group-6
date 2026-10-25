"""Beam search over partial enclosures from several gate poses.

Each open chain is ranked by the score of the best validated enclosure it can
reach with one or two more straight runs (recorded as soon as it is found), by
an estimate of its eventual score, and by diversity. An optimistic score bound
prunes chains that cannot beat the best enclosure found so far. Walls and spare
gates are both used to build faces. Time left after the first pass goes to
wider passes that keep every candidate face.

Interior gate poses come from rectangles the inventory can build, placed at
points spread along the room by walking distance, so long winding rooms are
covered and the gate starts on a corner of something that fits.
"""

import heapq
import math
import time
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from shapely import contains_xy
from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep

from players.player0 import Player
from src.constants import (
    ANGLE_TOL,
    BASELINE_SCORE,
    MAX_FACE_LENGTH,
    MIN_WALL_LENGTH,
    TOL,
)
from src.enclosure import (
    Construction,
    compute_faces,
    score_construction,
    validate_construction,
)
from src.pieces import Connector, ConnectorType, Piece, PieceType

# Beam slots by validated completion score and by estimated score; the rest go
# to diverse states, and any unused capacity is refilled from the others.
COMPLETION_SHARE = 0.5
ESTIMATE_SHARE = 0.3
DIVERSITY_CELL = 5.0
MAX_POSES = 16
BEAM_WIDTH = 256
# Leftover time runs these multiples of the beam width, keeping every face.
WIDER_PASSES = (4, 16)
INTERIOR_AXES = 2
AXIS_CLUSTER_DEGREES = 10.0  # wall directions this close count as one axis
GRID_POINTS = 400  # interior sample points for walking distances
TEMPLATE_SPOTS = 8  # interior points spread along the room for templates
MAX_FACE_PIECES = 6
MAX_ALLOCATIONS = 16
MAX_VALIDATIONS = 4  # official closure validations per chain

# Heading change (degrees) -> the connector that makes it.
TURN_CONNECTORS = {
    round(joint.turn_angle()): joint
    for joint in (
        Connector(ConnectorType.STRAIGHT),
        Connector(ConnectorType.RIGHT),
        Connector(ConnectorType.RIGHT, True),
        Connector(ConnectorType.DIAGONAL),
        Connector(ConnectorType.DIAGONAL, True),
    )
}
STRAIGHT = Connector(ConnectorType.STRAIGHT)


def _share(deadline, remaining):
    """An equal slice of the time left before deadline for one of remaining runs."""
    now = time.process_time()
    return now + max(0.0, deadline - now) / remaining


def _expired(deadline):
    return time.process_time() >= deadline


def _unit(heading):
    rad = math.radians(heading)
    return math.cos(rad), math.sin(rad)


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _turn(heading_from, heading_to):
    """The connector turning heading_from into heading_to, or None."""
    change = (heading_to - heading_from + 180) % 360 - 180
    for turn, joint in TURN_CONNECTORS.items():
        if abs(change - turn) <= ANGLE_TOL:
            return joint
    return None


def _pool(stock, is_gate):
    return stock.gates if is_gate else stock.walls


@dataclass
class Completion:
    """A validated enclosure closing a chain with one or two more straight runs."""

    score: float
    construction: Construction


class BeamFallback(Player):
    """Beam search over partial wall chains from several starting gate poses."""

    def __init__(self, room, inventory, weights, *, deadline):
        super().__init__(room, inventory, weights)
        self.deadline = deadline
        corners = room.get_boundary_points()
        edge_start, edge_end = max(
            zip(corners, corners[1:] + corners[:1]),
            key=lambda edge: math.dist(*edge),
        )
        self.edge_start = tuple(edge_start)
        self.start_heading = math.degrees(
            math.atan2(edge_end[1] - edge_start[1], edge_end[0] - edge_start[0])
        )
        self.best_score = 0.0
        # Piece kinds (length, is_gate) a face can use: longest first, and a
        # wall before a gate of the same length, so allocations prefer walls
        # and keep spare gates for the second-gate bonus.
        self._kinds = sorted(
            {
                (length, is_gate)
                for is_gate in (False, True)
                for length, count in _pool(inventory, is_gate).items()
                if count > 0 and MIN_WALL_LENGTH <= length <= MAX_FACE_LENGTH
            },
            key=lambda kind: (-kind[0], kind[1]),
        )
        self._room_area = room.polygon.buffer(TOL)
        self._room = prep(self._room_area)  # fast repeated containment tests
        self._grid = None
        self._allocation_cache = {}
        self._width = BEAM_WIDTH
        self._keep_all = False

    def build_enclosure(self) -> Construction | None:
        """Search each pose with an equal share of the time, then widen."""
        deadline = self.deadline
        # Returning no enclosure scores zero, so that is the score to beat.
        self.best_score = 0.0
        self._best = None
        if _expired(deadline):
            return None
        poses = self._candidate_poses()
        passes = [(BEAM_WIDTH, False)] + [(BEAM_WIDTH * k, True) for k in WIDER_PASSES]
        for width, keep_all in passes:
            if _expired(deadline):
                break
            self._width, self._keep_all = width, keep_all
            for left, (start, heading) in zip(range(len(poses), 0, -1), poses):
                self._allocation_cache.clear()
                self._search_pose(start, heading, _share(deadline, left))

        return self._best

    # ------------------------------------------------------------------ poses

    def _candidate_poses(self):
        """Up to MAX_POSES gate poses: room corners, corners of enclosure
        templates placed inside the room, and edge midpoints, taken in turn."""
        vertices = [tuple(point) for point in self.room.get_boundary_points()]
        edges = sorted(
            (
                (
                    math.dist(a, b),
                    math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])),
                    a,
                    b,
                )
                for a, b in zip(vertices, vertices[1:] + vertices[:1])
            ),
            key=lambda edge: -edge[0],
        )
        corners, middles = [], []
        for _, heading, a, b in edges:
            middle = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            corners += [(a, heading), (b, heading + 180)]
            middles += [(middle, heading), (middle, heading + 180)]
        axes = self._axes(edges)
        templates = [
            pose
            for spot in self._spread_spots()
            for pose in self._templates(spot, axes)
        ]

        gates = [
            length
            for length, count in self.inventory.gates.items()
            if count > 0 and MIN_WALL_LENGTH <= length <= MAX_FACE_LENGTH
        ]
        shortest_gate = min(gates, default=None)
        poses, seen = [], set()
        groups = [corners, templates, middles]
        for index in range(max(map(len, groups))):
            for group in groups:
                if index >= len(group) or len(poses) >= MAX_POSES:
                    continue
                start, heading = group[index]
                heading = (heading + 180) % 360 - 180
                key = (round(start[0], 6), round(start[1], 6), round(heading, 6))
                if key in seen:
                    continue
                seen.add(key)
                # Skip poses where not even the shortest gate fits.
                ux, uy = _unit(heading)
                if shortest_gate is not None and not self._room.covers(
                    LineString(
                        [
                            start,
                            (
                                start[0] + shortest_gate * ux,
                                start[1] + shortest_gate * uy,
                            ),
                        ]
                    )
                ):
                    continue
                poses.append((tuple(start), heading))
        return poses or [(self.edge_start, self.start_heading)]

    def _axes(self, edges):
        """The room's main wall directions (mod 90), near-equal ones merged and
        ranked by total edge length."""
        clusters = []  # [axis of the longest edge in the cluster, total length]
        for length, heading, _, _ in edges:
            axis = heading % 90
            for cluster in clusters:
                gap = abs(axis - cluster[0]) % 90
                if min(gap, 90 - gap) <= AXIS_CLUSTER_DEGREES:
                    cluster[1] += length
                    break
            else:
                clusters.append([axis, length])
        clusters.sort(key=lambda cluster: -cluster[1])
        return [axis for axis, _ in clusters[:INTERIOR_AXES]]

    def _build_grid(self):
        """Interior sample points, joined to neighbours they can see directly."""
        polygon = self.room.polygon
        minx, miny, maxx, maxy = polygon.bounds
        step = max(1.0, math.sqrt(polygon.area / GRID_POINTS))
        inner = polygon.buffer(-MIN_WALL_LENGTH / 2)
        region = prep(polygon if inner.is_empty else inner)
        nodes = {}
        for i in range(int((maxx - minx) / step) + 1):
            for j in range(int((maxy - miny) / step) + 1):
                point = (minx + (i + 0.5) * step, miny + (j + 0.5) * step)
                if region.contains(Point(point)):
                    nodes[(i, j)] = point
        links = {key: [] for key in nodes}
        for (i, j), point in nodes.items():
            for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
                other = (i + di, j + dj)
                if other in nodes and self._room.covers(
                    LineString([point, nodes[other]])
                ):
                    weight = math.dist(point, nodes[other])
                    links[(i, j)].append((other, weight))
                    links[other].append(((i, j), weight))
        self._grid = (minx, miny, step, nodes, links)

    def _walk(self, sources):
        """Walking distance through the room from sources {node: distance}."""
        links = self._grid[4]
        distance = dict(sources)
        queue = [(d, node) for node, d in sources.items()]
        heapq.heapify(queue)
        while queue:
            d, node = heapq.heappop(queue)
            if d > distance[node]:
                continue
            for other, weight in links[node]:
                if d + weight < distance.get(other, math.inf):
                    distance[other] = d + weight
                    heapq.heappush(queue, (d + weight, other))
        return distance

    def _spread_spots(self):
        """Interior points spread out by walking distance through the room,
        each moved to the roomiest spot near it."""
        if self._grid is None:
            self._build_grid()
        _, _, step, nodes, _ = self._grid
        if not nodes:
            point = self.room.polygon.representative_point()
            return [(point.x, point.y)]
        boundary = self.room.polygon.boundary
        clearance = {key: boundary.distance(Point(p)) for key, p in nodes.items()}
        chosen = [max(nodes, key=clearance.get)]
        nearest = self._walk({chosen[0]: 0.0})
        spots = [nodes[chosen[0]]]
        while len(spots) < TEMPLATE_SPOTS:
            far = max(nodes, key=lambda key: nearest.get(key, math.inf))
            if nearest.get(far, math.inf) < 2 * MIN_WALL_LENGTH:
                break
            around = self._walk({far: 0.0})
            local = [key for key, d in around.items() if d <= 3 * step]
            spots.append(nodes[max(local, key=clearance.get)])
            for key, d in around.items():
                nearest[key] = min(nearest.get(key, math.inf), d)
            nearest[far] = 0.0
        return spots

    def _templates(self, spot, axes):
        """For each axis, a pose on the corner of the best-scoring rectangle
        the inventory can build that fits around spot (or the spot itself)."""
        weights = self.weights
        counts = self._counts(self.inventory)
        straights = self.inventory.connectors.get(ConnectorType.STRAIGHT, 0)
        gates = [
            length
            for length, count in self.inventory.gates.items()
            if count > 0 and MIN_WALL_LENGTH <= length <= MAX_FACE_LENGTH
        ]
        if not gates or self.inventory.connectors.get(ConnectorType.RIGHT, 0) < 4:
            return [(spot, axis) for axis in axes]
        sides = [
            length
            for length in range(MIN_WALL_LENGTH, MAX_FACE_LENGTH + 1)
            if self._allocations(counts, length)
        ]

        def gate_fits(side):
            # A gate plus walls (or the gate alone) can fill this side.
            return any(
                gate == side
                or (
                    side - gate >= MIN_WALL_LENGTH
                    and self._allocations(counts, side - gate)
                )
                for gate in gates
            )

        pairs = []
        for w in sides:
            for h in sides:
                score = BASELINE_SCORE + weights.A * w * h + 2 * weights.C * (w + h)
                if score <= 0 or not (gate_fits(w) or gate_fits(h)):
                    continue
                remaining, pieces = counts, 0
                for side in (w, h, w, h):
                    options = self._allocations(remaining, side)
                    if not options:
                        break
                    remaining = tuple(c - n for c, n in zip(remaining, options[0]))
                    pieces += sum(options[0])
                else:
                    if pieces - 4 <= straights:
                        pairs.append((score, w, h))
        pairs.sort(reverse=True)

        poses = []
        for axis in axes:
            u = _unit(axis)
            v = (-u[1], u[0])
            # Unit cells within one face length of spot, on a grid aligned to
            # the axis through spot rounded to whole units (rooms are drawn on
            # whole units). fits[i, j]: cell [i, i + 1) x [j, j + 1) is inside.
            radius = MAX_FACE_LENGTH
            origin = (round(spot[0]), round(spot[1]))
            offsets = np.arange(-radius, radius) + 0.5
            along, across = np.meshgrid(offsets, offsets, indexing="ij")
            inside = contains_xy(
                self.room.polygon,
                origin[0] + along * u[0] + across * v[0],
                origin[1] + along * u[1] + across * v[1],
            )
            table = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.int32)
            table[1:, 1:] = inside.cumsum(0).cumsum(1)
            pose = (spot, axis)
            for _, w, h in pairs:
                filled = (
                    table[w:, h:] - table[:-w, h:] - table[w:, :-h] + table[:-w, :-h]
                )
                # Only rectangles that contain spot's cell.
                lo_i, lo_j = max(0, radius - w + 1), max(0, radius - h + 1)
                window = filled[lo_i : radius + 1, lo_j : radius + 1] == w * h
                if not window.any():
                    continue
                cells = np.argwhere(window) + (lo_i, lo_j)
                centre = np.array([radius - w / 2, radius - h / 2])
                cells = cells[np.argsort(np.abs(cells - centre).sum(axis=1))]
                for i, j in cells[:3]:
                    a, b = int(i) - radius, int(j) - radius
                    c0 = (
                        origin[0] + a * u[0] + b * v[0],
                        origin[1] + a * u[1] + b * v[1],
                    )
                    c1 = (c0[0] + w * u[0], c0[1] + w * u[1])
                    c2 = (c1[0] + h * v[0], c1[1] + h * v[1])
                    c3 = (c0[0] + h * v[0], c0[1] + h * v[1])
                    if self._room.covers(Polygon([c0, c1, c2, c3])):
                        # Start the gate along a side it fits on, turning left.
                        pose = (c0, axis) if gate_fits(w) else (c1, axis + 90)
                        break
                if pose[0] is not spot:
                    break
            poses.append(pose)
        return poses

    # ----------------------------------------------------------------- search

    def _search_pose(self, start, start_heading, deadline):
        """Beam search from one gate pose, recording completions as they appear."""
        stock0 = self.inventory.copy()
        frontier = []
        ux, uy = _unit(start_heading)
        for gate_length, count in sorted(stock0.gates.items()):
            if (
                count <= 0
                or gate_length < MIN_WALL_LENGTH
                or gate_length > MAX_FACE_LENGTH
            ):
                continue
            end = (start[0] + gate_length * ux, start[1] + gate_length * uy)
            if not self._room.covers(LineString([start, end])):
                continue
            stock = stock0.copy()
            stock.gates[gate_length] -= 1
            state = {
                "start": start,
                "start_heading": start_heading,
                "points": [start, end],
                "heading": start_heading,
                "face": gate_length,
                "first_face": gate_length,
                "pieces": [Piece(PieceType.GATE, gate_length)],
                "joints": [],
                "stock": stock,
                "perimeter": gate_length,
                "area2": _cross(start, end),
            }
            if self._rank(state, deadline):
                frontier.append(state)

        while frontier and not _expired(deadline):
            children = []
            for state in frontier:
                if _expired(deadline):
                    return
                stock = state["stock"]
                if stock.count_pieces() == 0 or stock.count_connectors() < 2:
                    continue
                children.extend(self._face_extensions(state, deadline))
            frontier = self._select(children)

    def _bonus_possible(self, state):
        """Whether a second gate on another face could still be placed."""
        return any(n > 0 for n in state["stock"].gates.values()) or any(
            piece.is_gate for piece in state["pieces"][1:]
        )

    def _rank(self, state, deadline):
        """Record a completion and estimate; prune chains that cannot win."""
        weights = self.weights
        end, start = state["points"][-1], state["start"]
        distance = math.dist(end, start)
        bound = (
            BASELINE_SCORE
            + max(0.0, weights.A) * self._room_area.area
            + weights.C * (state["perimeter"] + distance)
            + (weights.G if self._bonus_possible(state) else 0.0)
        )
        if bound < self.best_score - 1e-6:
            return False
        area = abs(state["area2"] + _cross(end, start)) / 2
        state["estimated_score"] = (
            BASELINE_SCORE
            + weights.A * area
            + weights.C * (state["perimeter"] + distance)
        )
        state["completion_score"] = None
        if not _expired(deadline):
            completion = self._complete(state)
            if completion is not None:
                state["completion_score"] = completion.score
                if completion.score > self.best_score:
                    self._best = completion.construction
                    self.best_score = completion.score
        return True

    def _closing_plans(self, state):
        """Geometric one- and two-run closures: (joints, run lengths, corners)."""
        start, start_heading = state["start"], state["start_heading"]
        end = state["points"][-1]
        gap = (start[0] - end[0], start[1] - end[1])
        if math.hypot(*gap) <= TOL:
            return
        for departure_turn, departure in TURN_CONNECTORS.items():
            first_heading = state["heading"] + departure_turn
            u = _unit(first_heading)
            # One run straight back to the start.
            length = gap[0] * u[0] + gap[1] * u[1]
            if length >= MIN_WALL_LENGTH - TOL:
                length = round(length)
                miss = math.hypot(gap[0] - length * u[0], gap[1] - length * u[1])
                closing = _turn(first_heading, start_heading)
                if miss <= TOL and closing is not None:
                    yield (departure, closing), (length,), ()
            # Two runs: solve length1 * u + length2 * v = gap.
            for middle_turn, middle in TURN_CONNECTORS.items():
                if middle_turn == 0:
                    continue
                second_heading = first_heading + middle_turn
                v = _unit(second_heading)
                denominator = _cross(u, v)
                if abs(denominator) < 1e-9:
                    continue
                length1 = _cross(gap, v) / denominator
                length2 = _cross(u, gap) / denominator
                if min(length1, length2) < MIN_WALL_LENGTH - TOL:
                    continue
                length1, length2 = round(length1), round(length2)
                miss = math.hypot(
                    gap[0] - length1 * u[0] - length2 * v[0],
                    gap[1] - length1 * u[1] - length2 * v[1],
                )
                closing = _turn(second_heading, start_heading)
                if miss > TOL or closing is None:
                    continue
                corner = (end[0] + length1 * u[0], end[1] + length1 * u[1])
                yield (departure, middle, closing), (length1, length2), (corner,)

    def _faces_fit(self, state, joints, lengths):
        """Whether straight joins keep every merged face within the limit."""
        departs_straight = joints[0].is_straight()
        closes_straight = joints[-1].is_straight()
        if max(lengths) > MAX_FACE_LENGTH:
            return False
        if len(lengths) == 1 and departs_straight and closes_straight:
            # The current face, the run and the gate's face are one face.
            merged = state["face"] + lengths[0] + state["first_face"]
            return merged <= MAX_FACE_LENGTH
        if departs_straight and state["face"] + lengths[0] > MAX_FACE_LENGTH:
            return False
        return not (
            closes_straight and lengths[-1] + state["first_face"] > MAX_FACE_LENGTH
        )

    def _complete(self, state):
        """Best validated enclosure closing the chain in one or two runs, or None."""
        weights = self.weights
        stock = state["stock"]
        end, start = state["points"][-1], state["start"]
        scored = []
        for joints, lengths, corners in self._closing_plans(state):
            if not self._faces_fit(state, joints, lengths):
                continue
            needed = {}
            for joint in joints:
                needed[joint.connector_type] = needed.get(joint.connector_type, 0) + 1
            if any(stock.connectors.get(kind, 0) < n for kind, n in needed.items()):
                continue
            ring = [end, *corners, start]
            area2 = state["area2"] + sum(_cross(a, b) for a, b in pairwise(ring))
            score = (
                BASELINE_SCORE
                + weights.A * abs(area2) / 2
                + weights.C * (state["perimeter"] + sum(lengths))
            )
            scored.append((score, joints, lengths, corners, needed))
        scored.sort(key=lambda plan: -plan[0])

        bonus = weights.G if self._bonus_possible(state) else 0.0
        counts = self._counts(stock)
        best = None
        validations = 0
        for score, joints, lengths, corners, needed in scored:
            # Plans are in score order before the bonus: stop once no later
            # plan can beat the best validated one even with the bonus.
            if best is not None and score + bonus <= best.score:
                break
            if validations >= MAX_VALIDATIONS:
                break
            straights = stock.connectors.get(ConnectorType.STRAIGHT, 0) - needed.get(
                ConnectorType.STRAIGHT, 0
            )
            runs = self._allocate_runs(counts, lengths, straights)
            if runs is None:
                continue
            if not self._room.covers(LineString([end, *corners, start])):
                continue
            if not LineString([*state["points"], *corners, start]).is_simple:
                continue
            pieces = list(state["pieces"])
            connectors = list(state["joints"])
            for joint, allocation in zip(joints, runs):
                added = self._pieces(allocation)
                pieces += added
                connectors += [joint] + [STRAIGHT] * (len(added) - 1)
            candidate = self._add_second_gate(
                Construction(
                    start, state["start_heading"], pieces, [joints[-1], *connectors]
                )
            )
            validations += 1
            result = validate_construction(candidate, self.room, self.inventory)
            if not result.valid:
                continue
            actual = score_construction(result, weights)
            if best is None or actual > best.score:
                best = Completion(actual, candidate)
        return best

    def _counts(self, stock):
        return tuple(_pool(stock, gate).get(length, 0) for length, gate in self._kinds)

    def _pieces(self, allocation):
        return [
            Piece(PieceType.GATE if gate else PieceType.WALL, length)
            for (length, gate), count in zip(self._kinds, allocation)
            for _ in range(count)
        ]

    def _allocations(self, counts, total):
        """Allocate pieces to total; rank retained choices by piece count, then gates."""
        key = (counts, total)
        cached = self._allocation_cache.get(key)
        if cached is not None:
            return cached
        kinds = self._kinds
        # Bit r of reachable[i] is set when some allocation of kinds[i:] sums
        # to exactly r, so the search below only follows branches that can
        # still succeed.
        reachable = [0] * (len(kinds) + 1)
        reachable[len(kinds)] = 1
        for index in range(len(kinds) - 1, -1, -1):
            length = kinds[index][0]
            bits = 0
            for quantity in range(min(counts[index], total // length) + 1):
                bits |= reachable[index + 1] << (quantity * length)
            reachable[index] = bits & ((1 << (total + 1)) - 1)
        found = []

        def fill(index, remaining, chosen):
            if remaining == 0:
                found.append(tuple(chosen) + (0,) * (len(kinds) - len(chosen)))
                return
            if len(found) >= MAX_ALLOCATIONS:
                return
            length = kinds[index][0]
            for quantity in range(min(counts[index], remaining // length), -1, -1):
                if reachable[index + 1] >> (remaining - quantity * length) & 1:
                    fill(index + 1, remaining - quantity * length, [*chosen, quantity])

        if not reachable[0] >> total & 1:
            self._allocation_cache[key] = found
            return found
        fill(0, total, [])
        found.sort(
            key=lambda allocation: (
                sum(allocation),
                sum(n for n, (_, gate) in zip(allocation, kinds) if gate),
            )
        )
        self._allocation_cache[key] = found
        return found

    def _allocate_runs(self, counts, lengths, straights):
        """Joint piece allocations for each run that fit the stock and straights."""
        if len(lengths) == 1:
            for allocation in self._allocations(counts, lengths[0]):
                if sum(allocation) - 1 <= straights:
                    return (allocation,)
            return None
        best = None
        for first in self._allocations(counts, lengths[0]):
            rest = tuple(c - n for c, n in zip(counts, first))
            for second in self._allocations(rest, lengths[1]):
                pieces = sum(first) + sum(second)
                if pieces - 2 <= straights and (best is None or pieces < best[0]):
                    best = (pieces, first, second)
                break  # The first retained allocation uses the fewest pieces.
        return None if best is None else best[1:]

    def _face_extensions(self, state, deadline):
        """Complete straight faces from bounded knapsack choices, ranked."""
        stock = state["stock"]
        at_gate = len(state["pieces"]) == 1
        # Walls first, so each (pieces, length) slot lists a wall-only option
        # before one that spends a spare gate.
        kinds = sorted(self._kinds, key=lambda kind: (kind[1], kind[0]))
        for turn, first_joint in TURN_CONNECTORS.items():
            if turn == 0 and not at_gate:
                continue
            if _expired(deadline):
                return
            if stock.connectors.get(first_joint.connector_type, 0) <= 0:
                continue
            heading = state["heading"] + turn
            initial_face = state["face"] if turn == 0 else 0
            face_limit = MAX_FACE_LENGTH - initial_face
            after_joint = stock.copy()
            after_joint.connectors[first_joint.connector_type] -= 1
            max_pieces = min(
                MAX_FACE_PIECES,
                after_joint.count_pieces(),
                # n pieces need n - 1 straights plus one connector to close.
                after_joint.count_connectors(),
                after_joint.connectors.get(ConnectorType.STRAIGHT, 0) + 1,
                int(face_limit // MIN_WALL_LENGTH),
            )
            if max_pieces <= 0:
                continue

            # dp[(piece_count, total_length)] holds up to two different piece
            # multisets. Piece order within a straight face does not affect
            # its endpoint, so multisets avoid permutation repeats.
            zero = (0,) * len(kinds)
            dp = {(0, 0): [zero]}
            for index, (length, gate) in enumerate(kinds):
                updated = {key: list(options) for key, options in dp.items()}
                available = max(0, _pool(stock, gate).get(length, 0))
                for (piece_count, total), allocations in dp.items():
                    for quantity in range(
                        1,
                        min(
                            available,
                            max_pieces - piece_count,
                            (face_limit - total) // length,
                        )
                        + 1,
                    ):
                        key = (piece_count + quantity, total + quantity * length)
                        options = updated.setdefault(key, [])
                        for allocation in allocations:
                            candidate = list(allocation)
                            candidate[index] += quantity
                            candidate = tuple(candidate)
                            if candidate not in options and len(options) < 2:
                                options.append(candidate)
                dp = updated

            px, py = state["points"][-1]
            ux, uy = _unit(heading)
            candidates = []
            for (piece_count, total), allocations in dp.items():
                if piece_count == 0:
                    continue
                end = (px + total * ux, py + total * uy)
                if math.dist(end, state["start"]) <= TOL:
                    continue  # Closing runs are found by the parent's completion.
                if not self._room.covers(LineString([(px, py), end])):
                    continue
                if not LineString([*state["points"], end]).is_simple:
                    continue
                for allocation in allocations:
                    if _expired(deadline):
                        return
                    next_stock = after_joint.copy()
                    added = []
                    for (length, gate), quantity in zip(kinds, allocation):
                        _pool(next_stock, gate)[length] -= quantity
                        added += [
                            Piece(PieceType.GATE if gate else PieceType.WALL, length)
                        ] * quantity
                    if piece_count > 1:
                        next_stock.connectors[ConnectorType.STRAIGHT] -= piece_count - 1
                    points = list(state["points"])
                    for piece in added:
                        x, y = points[-1]
                        points.append((x + piece.length * ux, y + piece.length * uy))
                    child = {
                        "start": state["start"],
                        "start_heading": state["start_heading"],
                        "points": points,
                        "heading": heading,
                        "face": initial_face + total,
                        "first_face": (
                            initial_face + total
                            if at_gate and turn == 0
                            else state["first_face"]
                        ),
                        "pieces": [*state["pieces"], *added],
                        "joints": [
                            *state["joints"],
                            first_joint,
                            *([STRAIGHT] * (piece_count - 1)),
                        ],
                        "stock": next_stock,
                        "perimeter": state["perimeter"] + total,
                        "area2": state["area2"] + _cross((px, py), end),
                    }
                    if self._rank(child, deadline):
                        candidates.append((child, total))
            if not candidates:
                continue
            if self._keep_all:
                yield from (child for child, _ in candidates)
                continue

            # Keep the longest face, the best estimate, a shorter alternative
            # and the best validated completion for this turn.
            def best(options, key):
                return max(options, key=lambda option: key(option[0]))[0]

            # Ties (same geometry, different pieces) go to fewer pieces, which
            # leaves more pieces and connectors for the rest of the enclosure.
            def estimate(child):
                return child["estimated_score"], -len(child["pieces"])

            longest = max(total for _, total in candidates)
            selected = [
                best([c for c in candidates if c[1] == longest], estimate),
                best(candidates, estimate),
            ]
            shorter = [c for c in candidates if c[1] < longest]
            if shorter:
                target = longest * 0.7
                closest = min(abs(total - target) for _, total in shorter)
                selected.append(
                    best(
                        [c for c in shorter if abs(c[1] - target) == closest], estimate
                    )
                )
            completed = [c for c in candidates if c[0]["completion_score"] is not None]
            if completed:
                selected.append(
                    best(
                        completed,
                        lambda child: (
                            child["completion_score"],
                            -len(child["pieces"]),
                        ),
                    )
                )
            seen = set()
            for child in selected:
                if id(child) not in seen:
                    seen.add(id(child))
                    yield child

    def _select(self, children):
        """Next beam: best validated completions, best estimates, then diversity."""
        width = self._width
        chosen, seen = [], set()

        def take(order, quota):
            for child in order:
                if quota <= 0 or len(chosen) >= width:
                    return
                if id(child) not in seen:
                    seen.add(id(child))
                    chosen.append(child)
                    quota -= 1

        by_completion = sorted(
            (child for child in children if child["completion_score"] is not None),
            key=lambda child: -child["completion_score"],
        )
        by_estimate = sorted(children, key=lambda child: -child["estimated_score"])
        take(by_completion, int(width * COMPLETION_SHARE))
        take(by_estimate, int(width * ESTIMATE_SHARE))
        cells = {}
        for child in by_estimate:
            x, y = child["points"][-1]
            key = (
                round(x / DIVERSITY_CELL),
                round(y / DIVERSITY_CELL),
                round(child["heading"]) % 360,
                child["stock"].count_pieces(),
            )
            cells.setdefault(key, child)
        take(list(cells.values()), width)
        take(by_completion, width)
        take(by_estimate, width)
        return chosen

    def _add_second_gate(self, construction):
        """Replace a wall with a same-length gate on another face when possible."""
        if self.weights.G <= 0:
            return construction
        faces = compute_faces(construction)
        if sum(face.has_gate for face in faces) >= 2:
            return construction
        used_gates = {}
        for piece in construction.pieces:
            if piece.is_gate:
                used_gates[piece.length] = used_gates.get(piece.length, 0) + 1

        for face in faces:
            if face.has_gate:
                continue
            for piece_index in face.piece_indices:
                piece = construction.pieces[piece_index]
                if piece.is_gate:
                    continue
                already_used = used_gates.get(piece.length, 0)
                if self.inventory.gates.get(piece.length, 0) <= already_used:
                    continue
                pieces = list(construction.pieces)
                pieces[piece_index] = Piece(PieceType.GATE, piece.length)
                candidate = Construction(
                    construction.start,
                    construction.start_heading,
                    pieces,
                    list(construction.connectors),
                )
                result = validate_construction(candidate, self.room, self.inventory)
                if result.valid and result.second_gate_bonus:
                    return candidate
        return construction
