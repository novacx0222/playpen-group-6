"""Group 1: local search over a closed loop of straight faces.

At each spot in the room, place the highest-scoring rectangle, hexagon or
octagon the stock can build that fits there. Taking the spots best first,
keep making the change that raises the score most until no change does, or
CHECK_LIMIT shapes have been scored. Then push the loop flush against a room
wall and climb again, for as long as that ends higher. Return the best loop.
"""

import functools
import itertools
import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
import shapely
from shapely.geometry import box

import src.constants as c
from players.player0 import Player
from src.enclosure import (
    Construction,
    build_geometry,
    score_construction,
    validate_construction,
)
from src.inventory import Inventory
from src.pieces import Connector, ConnectorType, Piece, PieceType
from src.room import Room

# the search stops after scoring this many shapes, so a run does the same work
# on any computer
CHECK_LIMIT = 200_000
# a backup only: CPU time per room, 10 s under the simulator's 300 s limit; set
# PLAYER1_CPU_SECONDS in the environment to override
CPU_SECONDS = float(os.environ.get("PLAYER1_CPU_SECONDS", "290"))
MAX_STARTS = 40  # placements kept, more than the search ever reaches
FIT_RECTANGLES = 12  # rectangle sizes tried, highest estimated score first
FIT_HEXAGONS = 8  # hexagon sizes tried, highest estimated score first
FIT_OCTAGONS = 8  # octagon sizes tried, highest estimated score first
FIT_SHARE = 0.3  # of the time, at most, spent fitting shapes to spots
FLUSH_REACH = 6  # how far, at most, a loop is pushed to meet a wall
LENGTHS = range(c.MIN_WALL_LENGTH, c.MAX_FACE_LENGTH + 1)

CONNECTOR_FOR_TURN = {  # turn in degrees -> connector
    90: Connector(ConnectorType.RIGHT),
    -90: Connector(ConnectorType.RIGHT, reflex=True),
    45: Connector(ConnectorType.DIAGONAL),
    -45: Connector(ConnectorType.DIAGONAL, reflex=True),
}
STRAIGHT = Connector(ConnectorType.STRAIGHT)

# ----------------------------------------------------------------- the loop


@dataclass(frozen=True)
class Face:
    direction: int  # degrees, multiple of 45, measured from the loop's rotation
    pieces: tuple[Piece, ...]  # in order along the face

    @property
    def length(self) -> int:
        return sum(piece.length for piece in self.pieces)

    @property
    def gate_count(self) -> int:
        return sum(1 for piece in self.pieces if piece.is_gate)


@dataclass(frozen=True)
class Loop:
    start: tuple[float, float]  # where face 0 begins
    rotation: float  # degrees; every face direction is measured from this
    faces: tuple[Face, ...]

    def with_faces(self, faces) -> "Loop":
        return replace(self, faces=tuple(faces))

    def moved_to(self, start: tuple[float, float]) -> "Loop":
        return replace(self, start=start)

    def corner_turns(self) -> list[int]:
        """The turn at the start of each face (from the face before it)."""
        return [
            turn_between(self.faces[i - 1].direction, face.direction)
            for i, face in enumerate(self.faces)
        ]

    def displacement_of(self, face: Face) -> tuple[float, float]:
        """How far `face` moves along x and y, in room coordinates."""
        angle = math.radians(self.rotation + face.direction)
        return face.length * math.cos(angle), face.length * math.sin(angle)


def turn_between(direction_from: int, direction_to: int) -> int:
    """The turn, in (-180, 180], from one direction to the other."""
    difference = (direction_to - direction_from) % 360
    return difference - 360 if difference > 180 else difference


def opposite(direction: int) -> int:
    return (direction + 180) % 360


def join_aligned_faces(loop: Loop) -> Loop | None:
    """Merge any two adjacent faces that point the same way into one face.
    None if the merged face would be too long, or if the pair is the last
    and first face (merging them would move the start)."""
    faces = [loop.faces[0]]
    for face in loop.faces[1:]:
        if face.direction == faces[-1].direction:
            faces[-1] = Face(face.direction, faces[-1].pieces + face.pieces)
        else:
            faces.append(face)
    if faces[-1].direction == faces[0].direction:
        return None
    if any(face.length > c.MAX_FACE_LENGTH for face in faces):
        return None
    return loop.with_faces(faces)


def to_construction(loop: Loop) -> Construction:
    pieces, connectors = [], []
    for face, turn in zip(loop.faces, loop.corner_turns()):
        pieces += face.pieces
        connectors += [CONNECTOR_FOR_TURN[turn]] + [STRAIGHT] * (len(face.pieces) - 1)
    first_direction = loop.rotation + loop.faces[0].direction
    return Construction(loop.start, first_direction, pieces, connectors)


def corners(loop: Loop) -> list[tuple[float, float]]:
    return build_geometry(to_construction(loop))[0]


class Scored(NamedTuple):
    loop: Loop
    score: float


ANYWHERE = Room(box(-1e5, -1e5, 1e5, 1e5))


def validate_anywhere(loop: Loop, inventory: Inventory):
    """The simulator's checks of `loop`, all but fitting in the room."""
    return validate_construction(to_construction(loop), ANYWHERE, inventory)


# ------------------------------------------------- choosing pieces for a face


class FaceSpec(NamedTuple):
    direction: int
    length: int
    gate_count: int = 0


def spare(inventory: Inventory, faces, *skip: int) -> Inventory:
    """A copy of `inventory` without the pieces used by `faces`, except the
    faces at the indices in `skip`."""
    left = inventory.copy()
    for i, face in enumerate(faces):
        if i not in skip:
            for piece in face.pieces:
                left.take_piece(piece)
    return left


def pick_pieces(
    length: int, inventory: Inventory, gate_count: int = 0
) -> tuple[Piece, ...] | None:
    """Pieces in stock adding up to exactly `length`, at least `gate_count` of
    them gates, or None. Largest pieces first, so few STRAIGHTs are spent;
    walls before any extra gate. An empty tuple for length 0. Cached: the
    same question comes up for many candidates in every step."""
    walls, gates = tuple(inventory.walls.items()), tuple(inventory.gates.items())
    return _pick_pieces(length, gate_count, walls, gates)


def _take_one(stock, length: int):
    return tuple((n, count - (n == length)) for n, count in stock)


@functools.cache
def _pick_pieces(length: int, gate_count: int, walls, gates):
    if length == 0:
        return () if gate_count == 0 else None
    if not c.MIN_WALL_LENGTH <= length <= c.MAX_FACE_LENGTH:
        return None
    gate_options = [(n, True) for n, count in gates if count > 0]
    wall_options = [(n, False) for n, count in sorted(walls, reverse=True) if count > 0]
    if gate_count:
        options = gate_options + wall_options
    else:
        options = wall_options + gate_options
    for n, is_gate in options:
        if n > length:
            continue
        if is_gate:
            rest = _pick_pieces(
                length - n, max(gate_count - 1, 0), walls, _take_one(gates, n)
            )
        else:
            rest = _pick_pieces(length - n, gate_count, _take_one(walls, n), gates)
        if rest is not None:
            return (Piece(PieceType.GATE if is_gate else PieceType.WALL, n),) + rest
    return None


def build_faces(specs: list[FaceSpec], inventory: Inventory) -> list[Face] | None:
    """Faces for `specs`, each taking its pieces from what the previous ones
    left; zero lengths are skipped. None if any face can't be made."""
    faces = []
    for direction, length, gate_count in specs:
        pieces = pick_pieces(length, inventory, gate_count)
        if pieces is None:
            return None
        if pieces:
            faces.append(Face(direction % 360, pieces))
            inventory = spare(inventory, faces[-1:])
    return faces


# -------------------------------------------------------------------- moves

# Each move yields loops that are closed by construction: a face's length only
# changes together with a face pointing the same or the opposite way, and the
# other moves reorder or insert faces without changing where the loop ends.


def parallel_faces(faces) -> Iterator[tuple[int, int, int]]:
    """(i, j, sign) for each pair of faces pointing the same way (sign -1)
    or opposite ways (sign 1)."""
    for i, j in itertools.combinations(range(len(faces)), 2):
        if faces[j].direction == faces[i].direction:
            yield i, j, -1
        elif faces[j].direction == opposite(faces[i].direction):
            yield i, j, 1


def resize_face(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """Change one face's length together with a parallel face, at the
    face's end or at its start."""
    faces = loop.faces
    for i, j, sign in parallel_faces(faces):
        face, other = faces[i], faces[j]
        left = spare(inventory, faces, i, j)
        for length in range(c.MIN_WALL_LENGTH, c.MAX_FACE_LENGTH + 1):
            change = length - face.length
            other_length = other.length + sign * change
            if change == 0 or other_length < c.MIN_WALL_LENGTH:
                continue
            specs = [
                FaceSpec(face.direction, length, face.gate_count),
                FaceSpec(other.direction, other_length, other.gate_count),
            ]
            pair = build_faces(specs, left)
            if pair is None:
                continue
            new_faces = list(faces)
            new_faces[i], new_faces[j] = pair
            resized = loop.with_faces(new_faces)
            yield resized
            dx, dy = loop.displacement_of(new_faces[i])
            back = change / new_faces[i].length
            yield resized.moved_to(
                (loop.start[0] - dx * back, loop.start[1] - dy * back)
            )


def detour_specs(
    face: Face, width: int, before: int, depth: int, side: int
) -> list[FaceSpec]:
    after = face.length - width - before
    direction = face.direction
    return [
        FaceSpec(direction, before),
        FaceSpec(direction + 90 * side, depth),
        FaceSpec(direction, width, face.gate_count),
        FaceSpec(direction - 90 * side, depth),
        FaceSpec(direction, after),
    ]


def extend_with_a_turn(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """Replace `width` units of a face by a rectangular detour of `depth`,
    outwards (a bump) or inwards (a notch), at the face's start, middle or
    end. The face's gate, if any, moves onto the detour.
    Example:
        before:   ────────────────────      one face, length = before + width + after

        after:    ───┐           ┌─────     five faces:
                     │           │            before     (same direction)
                     └───────────┘            depth      (turned 90°)
                                              width      (same direction, carries the gate if the face had one)
                                              depth      (turned back 90°)
                                              after      (same direction)
    """
    faces = loop.faces
    depths = sorted(inventory.walls)
    for i, face in enumerate(faces):
        left = spare(inventory, faces, i)
        for width in range(c.MIN_WALL_LENGTH, face.length + 1):
            rest = face.length - width
            befores = {0, rest // 2, rest}
            for before, depth, side in itertools.product(befores, depths, (-1, 1)):
                detour = build_faces(
                    detour_specs(face, width, before, depth, side), left
                )
                if detour:
                    yield loop.with_faces(faces[:i] + tuple(detour) + faces[i + 1 :])


def add_diagonal_pair(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """Cut two matching 90° corners with opposite diagonals, which cancel."""
    faces, turns = loop.faces, loop.corner_turns()
    left = spare(inventory, faces)
    right_angles = [i for i, turn in enumerate(turns) if abs(turn) == 90]
    for first, second in itertools.combinations(right_angles, 2):
        into_first, into_second = faces[first - 1], faces[second - 1]
        if turns[first] != turns[second]:
            continue
        if into_second.direction != opposite(into_first.direction):
            continue
        half_turn = turns[first] // 2
        for length in sorted(inventory.walls):
            specs = [
                FaceSpec(into_first.direction + half_turn, length),
                FaceSpec(into_second.direction + half_turn, length),
            ]
            diagonals = build_faces(specs, left)
            if diagonals:
                new_faces = list(faces)
                new_faces.insert(second, diagonals[1])  # later index first
                new_faces.insert(first, diagonals[0])
                yield loop.with_faces(new_faces)


def add_gate_to_face(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """Rebuild a face with a gate added or taken away."""
    faces = loop.faces
    for i, face in enumerate(faces):
        left = spare(inventory, faces, i)
        pieces = pick_pieces(face.length, left, 0 if face.gate_count else 1)
        if pieces:
            new_faces = list(faces)
            new_faces[i] = Face(face.direction, pieces)
            if any(new_face.gate_count for new_face in new_faces):
                yield loop.with_faces(new_faces)


MOVES = (
    resize_face,
    extend_with_a_turn,
    add_diagonal_pair,
    add_gate_to_face,
)


def neighboring_loops(loop: Loop, inventory: Inventory) -> Iterator[Loop]:
    """Loops the moves make from `loop` that have a connector for every
    corner. A detour flush with a corner can turn 135 or 180 degrees."""
    for move in MOVES:
        for candidate in move(loop, inventory):
            candidate = join_aligned_faces(candidate)
            if candidate is None:
                continue
            if all(turn in CONNECTOR_FOR_TURN for turn in candidate.corner_turns()):
                yield candidate


# -------------------------------------------------------- initial solutions


def shape_sides(a: int, b: int, d: int, e: int) -> list[tuple[int, int]]:
    """[(direction, length)] of an a by b rectangle with its corners cut off
    at 45 degrees: two opposite corners by sides of length d, the other two
    by sides of length e. With e = 0 a hexagon, which needs 2 right-angle
    connectors and 4 diagonal ones; with d = e = 0 the rectangle.

                a
            ╱‾‾‾‾‾‾‾╲
         d ╱         ╲ e
          │           │
        b │           │ b
          │           │
         e ╲         ╱ d
            ╲_______╱
                a
    """
    return [
        (0, a),
        (45, d),
        (90, b),
        (135, e),
        (180, a),
        (225, d),
        (270, b),
        (315, e),
    ]


def shape_sizes(inventory: Inventory) -> list[tuple[int, int, int, int]]:
    """(a, b, d, e) for every rectangle, hexagon and octagon (d = e) the
    connectors allow, each up to rotation once."""
    right = inventory.connectors.get(ConnectorType.RIGHT, 0)
    diagonal = inventory.connectors.get(ConnectorType.DIAGONAL, 0)
    cuts = [(0, 0)] if right >= 4 else []
    if right >= 2 and diagonal >= 4:
        cuts += [(d, 0) for d in LENGTHS]
    if diagonal >= 8:
        cuts += [(d, d) for d in LENGTHS]
    return [
        (a, b, d, e)
        for a, b in itertools.product(LENGTHS, LENGTHS)
        for d, e in cuts
        if d != e or a >= b
    ]


def corners_cut(size: tuple[int, int, int, int]) -> int:
    """0 for a rectangle, 1 for a hexagon, 2 for an octagon."""
    return (size[2] > 0) + (size[3] > 0)


def estimated_score(a: int, b: int, d: int, e: int, weights) -> float:
    cut_d, cut_e = d / math.sqrt(2), e / math.sqrt(2)
    width, height = a + cut_d + cut_e, b + cut_d + cut_e
    area = width * height - cut_d * cut_d - cut_e * cut_e
    return weights.A * area + weights.C * 2 * (a + b + d + e)


def built_shapes(
    sizes, inventory: Inventory, weights, wanted: int, deadline: float
) -> list[Scored]:
    """The first `wanted` of `sizes` the stock builds, each with its gate on
    whichever side takes one, at the origin, scored."""
    found = []
    for size in sizes:
        if len(found) == wanted or time.process_time() >= deadline:
            break
        sides = shape_sides(*size)
        for gate in range(len(sides)):
            specs = [
                FaceSpec(direction, length, int(i == gate))
                for i, (direction, length) in enumerate(sides)
            ]
            faces = build_faces(specs, inventory)
            if faces is None:
                continue
            shape = Loop((0.0, 0.0), 0.0, tuple(faces))
            result = validate_anywhere(shape, inventory)
            if result.valid:
                found.append(Scored(shape, score_construction(result, weights)))
                break
    return found


def starting_shapes(inventory: Inventory, weights, deadline: float) -> list[Scored]:
    """The smallest shape the stock builds, so that every spot that fits
    anything gets a shape, then the FIT_RECTANGLES rectangles, FIT_HEXAGONS
    hexagons and FIT_OCTAGONS octagons it builds with the highest estimated
    score."""
    sizes = shape_sizes(inventory)
    found = built_shapes(sorted(sizes, key=sum), inventory, weights, 1, math.inf)
    by_estimate = sorted(sizes, key=lambda size: -estimated_score(*size, weights))
    for cut, wanted in enumerate((FIT_RECTANGLES, FIT_HEXAGONS, FIT_OCTAGONS)):
        family = [size for size in by_estimate if corners_cut(size) == cut]
        found += built_shapes(family, inventory, weights, wanted, deadline)
    return found


def room_rotations(room: Room, shape: Loop) -> set[float]:
    """Every direction a room wall runs in, turned in 90-degree steps, or
    45-degree steps for a shape with diagonal sides (the hexagon)."""
    step = 45 if any(face.direction % 90 for face in shape.faces) else 90
    return {
        round(math.degrees(math.atan2(y2 - y1, x2 - x1)) % step, 6) + k * step
        for (x1, y1), (x2, y2) in itertools.pairwise(room.polygon.exterior.coords)
        for k in range(360 // step)
    }


def placements_at(shape: Loop, rotation: float, room: Room, floor) -> list:
    """(centre, placement) for each placement of `shape` that fits on the
    floor, centred on a grid point or cornered on a room corner. Every
    placement is tested at once."""
    min_x, min_y, max_x, max_y = room.polygon.bounds
    step = c.MIN_WALL_LENGTH
    grid = np.array(
        [
            (min_x + i * step, min_y + j * step)
            for i in range(int((max_x - min_x) / step) + 1)
            for j in range(int((max_y - min_y) / step) + 1)
        ]
    )
    turned = replace(shape, rotation=rotation)
    shape_corners = np.array(corners(turned))
    middle = shape_corners.mean(axis=0)
    room_corners = np.array(room.polygon.exterior.coords[:-1])
    starts = np.concatenate(
        [
            grid - middle,
            (room_corners[:, None, :] - shape_corners[None, :, :]).reshape(-1, 2),
        ]
    )
    placed = shapely.polygons(starts[:, None, :] + shape_corners[None, :, :])
    starts = starts[shapely.contains(floor, placed)]
    return [
        (tuple(np.round(start + middle, 1)), turned.moved_to(tuple(start)))
        for start in starts
    ]


def flush_shifts(loop: Loop, room: Room) -> set[tuple[float, float]]:
    """Each move of the whole loop, shorter than FLUSH_REACH, that puts one
    of its faces against a room wall running the same way."""
    points = corners(loop)
    shifts = set()
    for (ax, ay), (bx, by) in zip(points, points[1:] + points[:1]):
        face_angle = math.atan2(by - ay, bx - ax)
        for (x1, y1), (x2, y2) in itertools.pairwise(room.polygon.exterior.coords):
            wall_angle = math.atan2(y2 - y1, x2 - x1)
            if abs(math.sin(face_angle - wall_angle)) > c.TOL:
                continue
            nx, ny = -math.sin(wall_angle), math.cos(wall_angle)
            distance = (x1 - ax) * nx + (y1 - ay) * ny
            if c.TOL < abs(distance) < FLUSH_REACH:
                shifts.add((round(distance * nx, 6), round(distance * ny, 6)))
    return shifts


def initial_solutions(
    shapes: list[Scored], room: Room, floor, deadline: float
) -> list[Loop]:
    """At each spot, the highest-scoring of `shapes` that fits there; spots
    best first, centres spaced apart, at most MAX_STARTS of them."""
    best: dict = {}  # spot -> (score, placement)
    for shape, score in shapes:
        for rotation in sorted(room_rotations(room, shape)):
            if best and time.process_time() >= deadline:
                break  # out of time: keep what fits so far
            for middle, loop in placements_at(shape, rotation, room, floor):
                if middle not in best or best[middle][0] < score:
                    best[middle] = (score, loop)
    ranked = sorted(best.items(), key=lambda item: -item[1][0])
    chosen: list[Loop] = []
    centres: list[tuple[float, float]] = []
    for middle, (_, loop) in ranked:
        if len(chosen) == MAX_STARTS:
            break
        if all(math.dist(middle, other) >= c.MIN_WALL_LENGTH for other in centres):
            chosen.append(loop)
            centres.append(middle)
    return chosen


# ------------------------------------------------------------------- search


class Player1(Player):
    """Group 1: best-change local search from the largest rectangles,
    hexagons and octagons that fit, closure kept by construction."""

    def __init__(self, room, inventory, weights) -> None:
        super().__init__(room, inventory, weights)
        self.floor = room.polygon.buffer(c.TOL)
        shapely.prepare(self.floor)
        self.checks = 0  # shapes scored so far

    def out_of_work(self, deadline: float) -> bool:
        return self.checks >= CHECK_LIMIT or time.process_time() >= deadline

    def evaluate(self, loop: Loop) -> float | None:
        """The simulator's score for `loop`, or None if it would be invalid.
        A loop with a corner outside the room is pruned before the (much
        slower) full validation, which would reject it anyway."""
        self.checks += 1
        xs, ys = zip(*corners(loop))
        if not shapely.contains_xy(self.floor, xs, ys).all():
            return None
        result = validate_anywhere(loop, self.inventory)
        if not result.valid or not self.floor.contains(result.polygon):
            return None
        return score_construction(result, self.weights)

    def scored_neighbors(self, loop: Loop, deadline: float) -> list[Scored] | None:
        """The valid neighbors of `loop`, scored; None if time runs out
        part-way, so that a long step never runs past the deadline."""
        scored = []
        for neighbor in neighboring_loops(loop, self.inventory):
            if time.process_time() >= deadline:
                return None
            score = self.evaluate(neighbor)
            if score is not None:
                scored.append(Scored(neighbor, score))
        return scored

    def hill_climb(self, loop: Loop, deadline: float) -> Scored:
        """Step to the best neighbor until none is better, or out of work."""
        current = Scored(loop, self.evaluate(loop))
        while not self.out_of_work(deadline):
            neighbors = self.scored_neighbors(current.loop, deadline)
            if neighbors is None:
                break  # out of time part-way through this step
            better = [n for n in neighbors if n.score > current.score]
            if not better:
                break
            current = max(better, key=lambda n: n.score)
        return current

    def push_to_walls(self, result: Scored, deadline: float) -> Scored:
        """Move the loop flush against a wall and climb again; keep the first
        move that ends higher, and repeat until none does."""
        improved = True
        while improved and not self.out_of_work(deadline):
            improved = False
            for dx, dy in flush_shifts(result.loop, self.room):
                x, y = result.loop.start
                moved = result.loop.moved_to((x + dx, y + dy))
                if self.evaluate(moved) is None:
                    continue
                climbed = self.hill_climb(moved, deadline)
                if climbed.score > result.score:
                    result, improved = climbed, True
                    break
        return result

    def build_enclosure(self) -> Construction | None:
        now = time.process_time()
        deadline = now + CPU_SECONDS  # covers placing and growing too
        fit_until = now + FIT_SHARE * CPU_SECONDS
        shapes = starting_shapes(self.inventory, self.weights, fit_until)
        placements = initial_solutions(shapes, self.room, self.floor, fit_until)

        results = []
        for placement in placements:
            if results and self.out_of_work(deadline):
                break  # out of work; the first start is always searched
            if self.evaluate(placement) is None:
                continue  # fits only within rounding error
            climbed = self.hill_climb(placement, deadline)
            results.append(self.push_to_walls(climbed, deadline))
        if not results:
            self.note = "no starting shape fits: nothing built"
            return None
        best = max(results, key=lambda result: result.score)
        self.note = f"{len(placements)} initial solutions, {self.checks} shapes scored; best {best.score:.0f} ({len(best.loop.faces)} faces)"
        # A: an enclosure that scores below zero is worse than none at all
        if best.score <= 0:
            self.note += " -- below zero, so nothing is built"
            return None
        return to_construction(best.loop)
