"""Room-guided wall following with inventory-aware inward backtracking."""

import math
import time
from functools import lru_cache
from itertools import pairwise

from shapely.geometry import LineString, Point
from shapely.geometry.polygon import orient

from players.player0 import Player
from src.constants import MAX_FACE_LENGTH, TOL
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType


class Player3(Player):
    """Start at roomy corners, follow the boundary, and backtrack to close."""

    SEARCH_SECONDS = 55.0  # Leave headroom below the submission checker's 60s limit.
    MAX_NODES = 60000
    NODES_PER_START = 3000

    def _starts(self):
        """Rank corners by usable local floor area; also try points inside them.

        Edge samples let an enclosure start beyond a sloping/narrow corner.
        All headings follow a CCW room edge, with the room on the left.
        """
        polygon = orient(self.room.polygon, sign=1.0)
        vertices = list(polygon.exterior.coords)
        lengths = [
            k
            for pool in (self.inventory.walls, self.inventory.gates)
            for k, count in pool.items()
            if count > 0
        ]
        radius = 2 * min(lengths)
        starts = []
        for a, b in pairwise(vertices):
            dx, dy = b[0] - a[0], b[1] - a[1]
            distance = math.hypot(dx, dy)
            if distance == 0:
                continue
            heading = math.degrees(math.atan2(dy, dx))
            space = polygon.intersection(Point(a).buffer(radius)).area
            for fraction in (0.0, 0.25, 0.5, 0.75):
                for inset in (0.0, min(lengths) / 2):
                    start = (
                        a[0] + fraction * dx - inset * dy / distance,
                        a[1] + fraction * dy + inset * dx / distance,
                    )
                    if polygon.covers(Point(start)):
                        starts.append((space, fraction, inset, start, heading))
        starts.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [(item[3], item[4]) for item in starts]

    @staticmethod
    def _ray_capacity(room, origin, heading, limit):
        """Length of the first continuous room interval along this heading."""
        if limit <= TOL:
            return 0.0
        theta = math.radians(heading)
        ux, uy = math.cos(theta), math.sin(theta)
        ray = LineString([origin, (origin[0] + limit * ux, origin[1] + limit * uy)])
        clipped = room.intersection(ray)
        parts = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
        capacity = 0.0
        for part in parts:
            if part.is_empty or part.geom_type != "LineString":
                continue
            distances = [
                (x - origin[0]) * ux + (y - origin[1]) * uy for x, y in part.coords
            ]
            if min(distances) <= TOL:
                capacity = max(capacity, max(distances))
        return capacity

    @staticmethod
    @lru_cache(maxsize=4096)
    def _balance_capacity(counts, slots):
        """Necessary return budgets and attainable signed sums of unused lengths.

        A piece can be unused, positive, or negative in any single direction
        family. This relaxation never rejects a genuinely attainable return.
        """
        available = sorted(
            (length for length, count in counts for _ in range(min(count, slots))),
            reverse=True,
        )
        total = sum(available)
        bits = 1 << total
        for length in available:
            bits |= (bits << length) | (bits >> length)
        return sum(available[:slots]), bits, total

    def build_enclosure(self) -> Construction | None:
        if (
            not self.inventory.has_gate()
            or self.inventory.count_pieces() < 3
            or self.inventory.count_connectors() < 3
        ):
            self.note = "Not enough pieces, connectors, or gates to close a loop"
            return None

        stock = self.inventory.copy()
        room = self.room.polygon.buffer(TOL)
        boundary = self.room.polygon.boundary
        lengths = sorted(set(stock.walls) | set(stock.gates), reverse=True)
        turns = [
            (0, Connector(ConnectorType.STRAIGHT)),
            (45, Connector(ConnectorType.DIAGONAL)),
            (90, Connector(ConnectorType.RIGHT)),
            (-45, Connector(ConnectorType.DIAGONAL, reflex=True)),
            (-90, Connector(ConnectorType.RIGHT, reflex=True)),
        ]
        turn_lookup = dict(turns)
        max_depth = min(stock.count_pieces(), stock.count_connectors(), 40)
        deadline = getattr(
            self, "SEARCH_DEADLINE", time.process_time() + self.SEARCH_SECONDS
        )
        nodes = 0
        best, best_score = None, 0.0

        starts = self._starts()
        # Try long boundary-following chains first; short loops are a fallback.
        depth_limits = sorted(
            {min(4, max_depth), min(8, max_depth), max_depth}, reverse=True
        )
        attempts = (
            (start, heading, depth)
            for depth in depth_limits
            for start, heading in starts
        )
        for start, initial_heading, depth_limit in attempts:
            if nodes >= self.MAX_NODES or time.process_time() >= deadline:
                break
            seed_limit = min(nodes + self.NODES_PER_START, self.MAX_NODES)
            points, pieces, joints, segments = [start], [], [], []

            def search(
                heading,
                total_turn,
                face_length,
                balances=(0, 0, 0, 0),
                *,
                seed_limit=seed_limit,
                depth_limit=depth_limit,
                start=start,
                initial_heading=initial_heading,
                points=points,
                pieces=pieces,
                joints=joints,
                segments=segments,
            ):
                nonlocal nodes, best, best_score
                nodes += 1
                if nodes > seed_limit or time.process_time() >= deadline:
                    return False
                if pieces and math.dist(points[-1], start) <= TOL:
                    closing_turn = 360 - total_turn
                    connector = turn_lookup.get(closing_turn)
                    if (
                        len(pieces) < 3
                        or connector is None
                        or stock.connectors.get(connector.connector_type, 0) < 1
                    ):
                        return False
                    candidate = Construction(
                        start,
                        initial_heading,
                        pieces.copy(),
                        [connector] + joints.copy(),
                    )
                    result = validate_construction(candidate, self.room, self.inventory)
                    if result.valid:
                        score = score_construction(result, self.weights)
                        if score > best_score:
                            best, best_score = candidate, score
                        return True
                    return False
                if len(pieces) >= depth_limit:
                    return False
                # One remaining connector must always be reserved for the start.
                if sum(stock.connectors.values()) < (2 if pieces else 3):
                    return False
                slots = min(
                    depth_limit - len(pieces), sum(stock.connectors.values()) - 1
                )
                counts = tuple(
                    (length, stock.walls.get(length, 0) + stock.gates.get(length, 0))
                    for length in lengths
                )
                budget, reachable, offset = self._balance_capacity(counts, slots)
                # Axial and diagonal lengths balance independently: integer
                # lengths cannot cancel irrational diagonal projections.
                if sum(abs(value) for value in balances) > budget:
                    return False
                if any(
                    abs(value) > offset or not (reachable >> (offset + value)) & 1
                    for value in balances
                ):
                    return False
                if not any(piece.is_gate for piece in pieces) and not stock.has_gate():
                    return False
                turn_budget = 90 * stock.connectors.get(
                    ConnectorType.RIGHT, 0
                ) + 45 * stock.connectors.get(ConnectorType.DIAGONAL, 0)
                if abs(360 - total_turn) > turn_budget:
                    return False

                options = []
                for turn, connector in turns if pieces else [(0, None)]:
                    if (
                        connector
                        and stock.connectors.get(connector.connector_type, 0) < 1
                    ):
                        continue
                    if not -180 <= total_turn + turn <= 450:
                        continue
                    next_heading = heading + turn
                    theta = math.radians(next_heading)
                    face_budget = (
                        MAX_FACE_LENGTH - face_length if turn == 0 else MAX_FACE_LENGTH
                    )
                    capacity = self._ray_capacity(
                        room, points[-1], next_heading, face_budget
                    )
                    fitting = [
                        length
                        for length in lengths
                        if length <= capacity + TOL
                        and stock.walls.get(length, 0) + stock.gates.get(length, 0) > 0
                    ]
                    if not fitting:
                        continue  # Try a right/diagonal turn instead of each oversized wall.
                    for length in fitting:
                        # A gate always wins a tie against an available equal-length wall.
                        kind = (
                            PieceType.GATE
                            if stock.gates.get(length, 0)
                            else PieceType.WALL
                        )
                        pool = stock.gates if kind == PieceType.GATE else stock.walls
                        if pool.get(length, 0) < 1:
                            continue
                        next_face = face_length + length if turn == 0 else length
                        if next_face > MAX_FACE_LENGTH + TOL:
                            continue
                        end = (
                            points[-1][0] + length * math.cos(theta),
                            points[-1][1] + length * math.sin(theta),
                        )
                        edge = LineString([points[-1], end])
                        if not room.covers(edge):
                            continue
                        closes = math.dist(end, start) <= TOL
                        if closes:
                            end = start
                            edge = LineString([points[-1], end])
                        # Nonadjacent walls may only meet at the final closing vertex.
                        crossed = False
                        for i, old in enumerate(segments):
                            intersection = edge.intersection(old)
                            if intersection.is_empty:
                                continue
                            allowed = (
                                points[-1]
                                if i == len(segments) - 1
                                else (start if closes and i == 0 else None)
                            )
                            if (
                                allowed is None
                                or intersection.geom_type != "Point"
                                or intersection.distance(Point(allowed)) > TOL
                            ):
                                crossed = True
                                break
                        if crossed:
                            continue
                        # Follow the boundary first, then bend toward the interior.
                        proximity = boundary.distance(
                            edge.interpolate(0.5, normalized=True)
                        )
                        direction = round((next_heading - initial_heading) / 45) % 8
                        family = (0, 2, 1, 3)[direction % 4]
                        signed_length = length * (1 if direction < 4 else -1)
                        return_detour = total_turn + turn >= 180 and (
                            balances[family] * signed_length >= 0
                            or length > abs(balances[family])
                        )
                        options.append(
                            (
                                (
                                    not closes,
                                    return_detour,
                                    turn < 0,
                                    abs(turn) == 45 and balances[2:] == (0, 0),
                                    abs(turn),
                                    -length,
                                    proximity,
                                ),
                                Piece(kind, length),
                                connector,
                                end,
                                edge,
                                next_heading,
                                total_turn + turn,
                                next_face,
                            )
                        )
                options.sort(key=lambda item: item[0])
                for (
                    _,
                    piece,
                    connector,
                    end,
                    edge,
                    next_heading,
                    next_turn,
                    next_face,
                ) in options:
                    pool = stock.gates if piece.is_gate else stock.walls
                    pool[piece.length] -= 1
                    if connector:
                        stock.connectors[connector.connector_type] -= 1
                        joints.append(connector)
                    pieces.append(piece)
                    points.append(end)
                    segments.append(edge)
                    next_balances = list(balances)
                    direction = round((next_heading - initial_heading) / 45) % 8
                    family = (0, 2, 1, 3)[direction % 4]
                    next_balances[family] += piece.length * (1 if direction < 4 else -1)
                    found = search(
                        next_heading, next_turn, next_face, tuple(next_balances)
                    )
                    segments.pop()
                    points.pop()
                    pieces.pop()
                    if connector:
                        joints.pop()
                        stock.connectors[connector.connector_type] += 1
                    pool[piece.length] += 1
                    if found:
                        return True
                    if nodes >= seed_limit or time.process_time() >= deadline:
                        break
                return False

            search(initial_heading, 0, 0)
        self.note = (
            f"Boundary-following backtracking: closed {len(best.pieces)} pieces"
            if best
            else "No positive-scoring loop found within the backtracking budget"
        )
        return best
