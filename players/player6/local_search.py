"""Bounded local 45-degree moves that preserve integer face lengths and closure.

A move translates a short boundary chain by a diagonal vector and connects its
ends with equal, opposite diagonal faces. Their irrational displacements cancel
exactly. Existing pieces and gates stay assigned to their original faces; only
the two new faces consume spare pieces. Every accepted move is simulator-valid
and strictly improves the real score. This is a local heuristic, not arbitrary
corner chamfering or a global reassignment of materials.
"""

import math
import time
from collections import Counter
from dataclasses import dataclass

from shapely import Polygon, affinity
from shapely.prepared import prep

from players.player6.diagonal_search import DIRS, SQ, DiagonalSearch
from players.player6.inscribed_polygon import SearchStopped, check_stop
from players.player6.inscribed_polygon_stock import assign_inventory, clean_stock
from src.constants import BASELINE_SCORE, TOL
from src.enclosure import score_construction, validate_construction, walk_points
from src.pieces import ConnectorType, Piece, PieceType


@dataclass(frozen=True)
class BoundaryFace:
    length: int
    heading: int  # multiples of 45 degrees in the seed's coordinate frame
    pieces: tuple[Piece, ...]


@dataclass
class LocalSearchStats:
    attempted: int = 0
    feasible: int = 0
    accepted: int = 0
    cpu_seconds: float = 0.0


def turn_between(first, second):
    return (second - first + 4) % 8 - 4


def exact_vertices(faces):
    """Float witnesses from exact Z + Z*sqrt(2)/2 coordinates; require closure."""
    position = (0, 0, 0, 0)
    vertices = []
    for face in faces:
        xm, xn, ym, yn = position
        vertices.append((xm + xn * SQ, ym + yn * SQ))
        direction = DIRS[face.heading]
        position = tuple(a + face.length * b for a, b in zip(position, direction))
    return vertices if position == (0, 0, 0, 0) else None


class Local45Search:
    MAX_WINDOW = 4

    def __init__(
        self,
        room,
        inventory,
        weights,
        *,
        time_budget=2.0,
        stop=None,
        on_improvement=None,
    ):
        self.room = room
        self.inventory = inventory
        self.weights = weights
        self.deadline = time.process_time() + max(0.0, time_budget)
        self.external_stop = stop
        self.on_improvement = on_improvement
        self.stats = LocalSearchStats()

    def stopped(self):
        return time.process_time() >= self.deadline or (
            self.external_stop is not None and self.external_stop()
        )

    def place(self, polygon, origin, prepared_room):
        """Try the current placement and a few bounded translations, without rotation."""
        if prepared_room.covers(polygon):
            return origin
        qx0, qy0, qx1, qy1 = self.room.polygon.bounds
        px0, py0, px1, py1 = polygon.bounds
        xlo, xhi, ylo, yhi = qx0 - px0, qx1 - px1, qy0 - py0, qy1 - py1
        if xlo > xhi or ylo > yhi:
            return None
        offsets = [
            (min(max(0, xlo), xhi), min(max(0, ylo), yhi)),
            ((xlo + xhi) / 2, (ylo + yhi) / 2),
        ]
        offsets.extend((x, y) for x in (xlo, xhi) for y in (ylo, yhi))
        for dx, dy in dict.fromkeys(offsets):
            check_stop(self.stopped)
            if prepared_room.covers(affinity.translate(polygon, xoff=dx, yoff=dy)):
                return origin[0] + dx, origin[1] + dy
        return None

    @staticmethod
    def boundary(construction, result):
        points = walk_points(construction)
        first = result.faces[0].piece_indices[0]
        origin = points[first]
        dx = points[first + 1][0] - origin[0]
        dy = points[first + 1][1] - origin[1]
        angle = math.degrees(math.atan2(dy, dx))
        heading = 0
        faces = []
        for index, face in enumerate(result.faces):
            if index:
                connector = construction.connectors[face.piece_indices[0]]
                heading = (heading + round(connector.turn_angle() / 45)) % 8
            faces.append(
                BoundaryFace(
                    round(face.length),
                    heading,
                    tuple(construction.pieces[i] for i in face.piece_indices),
                )
            )
        return faces, origin, angle

    @staticmethod
    def spare_inventory(construction, inventory):
        used = Counter((piece.is_gate, piece.length) for piece in construction.pieces)
        walls = {
            length: count - used[(False, length)]
            for length, count in inventory.walls.items()
        }
        gates = {
            length: count - used[(True, length)]
            for length, count in inventory.gates.items()
        }
        straights = inventory.connectors.get(ConnectorType.STRAIGHT, 0) - sum(
            connector.is_straight() for connector in construction.connectors
        )
        return clean_stock(walls), clean_stock(gates), straights

    @staticmethod
    def to_construction(faces, origin, angle):
        turns = [
            turn_between(face.heading, faces[(i + 1) % len(faces)].heading)
            for i, face in enumerate(faces)
        ]
        per_face = [
            (
                tuple((0 if p.is_gate else 1, p.length) for p in face.pieces),
                sum(p.is_gate for p in face.pieces),
                len(face.pieces) - 1,
            )
            for face in faces
        ]
        return DiagonalSearch.to_construction(
            tuple((face.length, turn) for face, turn in zip(faces, turns)),
            per_face,
            (None, (angle + faces[0].heading * 45, *origin)),
        )

    def improve(self, seed):
        started = time.process_time()
        result = validate_construction(seed, self.room, self.inventory)
        if not result.valid:
            self.stats.cpu_seconds = time.process_time() - started
            return None
        best = seed
        best_score = score_construction(result, self.weights)
        prepared_room = prep(self.room.polygon.buffer(TOL))
        try:
            while True:
                check_stop(self.stopped)
                candidate = self._improve_once(best, result, best_score, prepared_room)
                if candidate is None:
                    break
                best, result, best_score = candidate
                self.stats.accepted += 1
                if self.on_improvement is not None:
                    self.on_improvement(best)
        except SearchStopped:
            pass
        self.stats.cpu_seconds = time.process_time() - started
        return best

    def _improve_once(self, seed, result, best_score, prepared_room):
        faces, origin, angle = self.boundary(seed, result)
        vertices = exact_vertices(faces)
        if vertices is None or len(faces) < 4:
            return None
        walls, gates, straights = self.spare_inventory(seed, self.inventory)
        if sum(walls.values()) + sum(gates.values()) < 2:
            return None
        radians = math.radians(angle)
        cosine, sine = math.cos(radians), math.sin(radians)
        plans = {}
        # Short additions often fit where long ones cross the room boundary.
        for length in range(5, 31):
            for index, vertex in enumerate(vertices):
                check_stop(self.stopped)
                ordered = faces[index:] + faces[:index]
                entry_turn = turn_between(ordered[-1].heading, ordered[0].heading)
                if abs(entry_turn) != 2:
                    continue
                diagonal = (ordered[-1].heading + entry_turn // 2) % 8
                reverse = (diagonal + 4) % 8
                for window in range(2, min(self.MAX_WINDOW, len(faces) - 2) + 1):
                    check_stop(self.stopped)
                    outline = [BoundaryFace(length, diagonal, ())]
                    outline += ordered[:window]
                    outline += [BoundaryFace(length, reverse, ())]
                    outline += ordered[window:]
                    turns = [
                        turn_between(
                            face.heading, outline[(i + 1) % len(outline)].heading
                        )
                        for i, face in enumerate(outline)
                    ]
                    if any(abs(turn) not in (1, 2) for turn in turns):
                        continue
                    if sum(
                        abs(turn) == 2 for turn in turns
                    ) > self.inventory.connectors.get(ConnectorType.RIGHT, 0) or sum(
                        abs(turn) == 1 for turn in turns
                    ) > self.inventory.connectors.get(ConnectorType.DIAGONAL, 0):
                        continue
                    self.stats.attempted += 1
                    points = exact_vertices(outline)
                    if points is None:
                        continue
                    start = (
                        origin[0] + vertex[0] * cosine - vertex[1] * sine,
                        origin[1] + vertex[0] * sine + vertex[1] * cosine,
                    )
                    polygon = Polygon(
                        [
                            (
                                start[0] + x * cosine - y * sine,
                                start[1] + x * sine + y * cosine,
                            )
                            for x, y in points
                        ]
                    )
                    if not polygon.is_valid or polygon.area <= 0:
                        continue
                    bonus = self.weights.G if result.second_gate_bonus or gates else 0
                    upper = (
                        BASELINE_SCORE
                        + self.weights.A * polygon.area
                        + self.weights.C * polygon.length
                        + bonus
                    )
                    if upper <= best_score + 1e-7:
                        continue
                    placed_origin = self.place(polygon, start, prepared_room)
                    if placed_origin is None:
                        continue
                    if length not in plans:
                        require_gate = bool(
                            gates
                            and not result.second_gate_bonus
                            and self.weights.G > 0
                        )
                        plan = assign_inventory(
                            [length, length],
                            walls,
                            gates,
                            {"straight": straights},
                            stop=self.stopped,
                            min_gate_faces=int(require_gate),
                            prefer_two=False,
                        )
                        if plan is None and require_gate:
                            plan = assign_inventory(
                                [length, length],
                                walls,
                                gates,
                                {"straight": straights},
                                stop=self.stopped,
                                min_gate_faces=0,
                                prefer_two=False,
                            )
                        plans[length] = plan
                    plan = plans[length]
                    if plan is None:
                        continue
                    bridge_pieces = []
                    for pieces, count in plan[length]:
                        bridge_pieces.extend(
                            [
                                tuple(
                                    Piece(
                                        PieceType.GATE if kind == 0 else PieceType.WALL,
                                        size,
                                    )
                                    for kind, size in pieces
                                )
                            ]
                            * count
                        )
                    outline[0] = BoundaryFace(length, diagonal, bridge_pieces[0])
                    outline[window + 1] = BoundaryFace(
                        length, reverse, bridge_pieces[1]
                    )
                    construction = self.to_construction(outline, placed_origin, angle)
                    checked = validate_construction(
                        construction, self.room, self.inventory
                    )
                    if not checked.valid:
                        continue
                    self.stats.feasible += 1
                    score = score_construction(checked, self.weights)
                    if score > best_score + 1e-7:
                        return construction, checked, score
        return None
