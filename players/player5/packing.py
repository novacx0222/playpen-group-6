"""Assign finite wall, gate, and connector stock to a fixed polygon.

Geometry searches can reuse one ``PolygonPacker`` across many candidates.
Faces with the same lengths share their piece-allocation results, even when
the polygons have different positions, directions, or corner arrangements.
"""

import math
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from shapely.geometry import Polygon
from shapely.prepared import prep

from src.constants import ANGLE_TOL, MAX_FACE_LENGTH, MIN_WALL_LENGTH, TOL
from src.enclosure import Construction, validate_construction
from src.inventory import Inventory
from src.pieces import Connector, ConnectorType, Piece, PieceType
from src.room import Room
from src.weights import Weights


class _DeadlineReached(Exception):
    """Unwind an unfinished allocation without caching it as impossible."""


@dataclass(frozen=True)
class _FaceAllocation:
    pieces: tuple[Piece, ...]
    used: tuple[tuple[int, int], ...]
    gate_count: int

    @property
    def straights(self):
        return len(self.pieces) - 1


@dataclass
class _PackingFrame:
    """One unfinished face assignment and its remaining alternatives."""

    position: int
    straights: int
    gated_faces: int
    state: tuple
    options: Iterator[_FaceAllocation]


class PolygonPacker:
    """Allocate and validate polygons without changing their geometry.

    ``deadline`` is an absolute ``time.process_time()`` value. A call may
    supply an earlier deadline. The inventory is copied and never consumed.
    ``total_length`` is the combined length of all usable walls and gates.
    For a fixed polygon, only the presence of gates on two distinct faces
    can change its score, so that is the only allocation objective.
    """

    def __init__(
        self,
        inventory: Inventory,
        room: Room,
        weights: Weights,
        deadline: float = math.inf,
    ):
        self.inventory = inventory.copy()
        self.room = room
        self.weights = weights
        self.deadline = deadline
        self._prepared_room = prep(room.polygon.buffer(TOL))
        self._piece_choices = tuple(
            Piece(kind, length)
            for kind, stock in (
                (PieceType.WALL, self.inventory.walls),
                (PieceType.GATE, self.inventory.gates),
            )
            for length, count in sorted(stock.items())
            if count > 0 and MIN_WALL_LENGTH <= length <= MAX_FACE_LENGTH
        )
        self._available_counts = tuple(
            (self.inventory.gates if piece.is_gate else self.inventory.walls)[
                piece.length
            ]
            for piece in self._piece_choices
        )
        self._straight_stock = self.inventory.connectors.get(ConnectorType.STRAIGHT, 0)
        self.total_length = sum(
            piece.length * count
            for piece, count in zip(self._piece_choices, self._available_counts)
        )
        self._face_allocations: dict[int, tuple[_FaceAllocation, ...]] = {}
        self._polygon_allocations: dict[
            tuple[tuple[int, ...], int], tuple[_FaceAllocation, ...] | None
        ] = {}
        self.feasible_lengths = self._reachable_lengths()
        usable_gates = sum(
            count
            for piece, count in zip(self._piece_choices, self._available_counts)
            if piece.is_gate
        )
        self.max_possible_gate_bonus = weights.G if usable_gates >= 2 else 0.0

    def _reachable_lengths(self):
        """Face totals that fit stock and its available straight connectors."""
        missing = MAX_FACE_LENGTH + 1
        minimum_pieces = [0] + [missing] * MAX_FACE_LENGTH
        for piece, available in zip(self._piece_choices, self._available_counts):
            previous = minimum_pieces
            minimum_pieces = previous.copy()
            for count in range(1, min(available, MAX_FACE_LENGTH // piece.length) + 1):
                amount = count * piece.length
                for total in range(amount, MAX_FACE_LENGTH + 1):
                    minimum_pieces[total] = min(
                        minimum_pieces[total], previous[total - amount] + count
                    )
        return frozenset(
            length
            for length in range(MIN_WALL_LENGTH, MAX_FACE_LENGTH + 1)
            if minimum_pieces[length] < missing
            and minimum_pieces[length] - 1 <= self._straight_stock
        )

    @staticmethod
    def _check_time(deadline):
        if time.process_time() >= deadline:
            raise _DeadlineReached

    def _face_options(self, length, deadline):
        """Every stock-feasible multiset making this face, without truncation."""
        if length in self._face_allocations:
            return self._face_allocations[length]
        choices = sorted(
            range(len(self._piece_choices)),
            key=lambda index: (
                self._piece_choices[index].length,
                self._piece_choices[index].is_gate,
            ),
        )
        used = Counter()
        selected = []
        found = []
        max_pieces = min(length // MIN_WALL_LENGTH, self._straight_stock + 1)

        def extend(first_choice, remaining):
            self._check_time(deadline)
            if remaining == 0:
                pieces = tuple(self._piece_choices[index] for index in selected)
                found.append(
                    _FaceAllocation(
                        pieces,
                        tuple(sorted(used.items())),
                        sum(p.is_gate for p in pieces),
                    )
                )
                return
            if len(selected) >= max_pieces:
                return
            for position in range(first_choice, len(choices)):
                index = choices[position]
                piece = self._piece_choices[index]
                if piece.length > remaining:
                    break
                if used[index] >= self._available_counts[index]:
                    continue
                used[index] += 1
                selected.append(index)
                extend(position, remaining - piece.length)
                selected.pop()
                used[index] -= 1
                if used[index] == 0:
                    del used[index]

        extend(0, length)
        found.sort(key=lambda option: (len(option.pieces), option.gate_count))
        self._face_allocations[length] = tuple(found)
        return self._face_allocations[length]

    def _assign_faces(self, lengths, gate_faces_needed, deadline):
        """Exact bounded backtracking over all faces and their shared stock."""
        cache_key = (lengths, gate_faces_needed)
        if cache_key in self._polygon_allocations:
            return self._polygon_allocations[cache_key]
        options = {
            length: self._face_options(length, deadline) for length in set(lengths)
        }
        if any(not choices for choices in options.values()):
            self._polygon_allocations[cache_key] = None
            return None

        # Constrained faces first. Store the original index so repeated face
        # lengths and different polygon orderings still share this cache.
        order = sorted(
            range(len(lengths)), key=lambda i: (len(options[lengths[i]]), -lengths[i])
        )
        ordered_lengths = tuple(lengths[index] for index in order)
        minimum_straights = [0] * (len(lengths) + 1)
        for position in range(len(lengths) - 1, -1, -1):
            minimum_straights[position] = minimum_straights[position + 1] + min(
                option.straights for option in options[ordered_lengths[position]]
            )
        remaining = list(self._available_counts)
        selected = [None] * len(lengths)
        failed = set()  # Remaining-stock states proven to have no completion.

        def available_options(position, straights, gated_faces):
            choices = options[ordered_lengths[position]]
            # Establish a valid fallback quickly; when the bonus matters,
            # try putting a gate on each of the first two suitable faces.
            prefer_gate = gated_faces < (2 if self.weights.G > 0 else gate_faces_needed)
            for wants_gate in (prefer_gate, not prefer_gate):
                for option in choices:
                    self._check_time(deadline)
                    if bool(option.gate_count) != wants_gate:
                        continue
                    if option.straights > straights:
                        continue
                    if all(remaining[index] >= count for index, count in option.used):
                        yield option

        def frame_for(position, straights, gated_faces):
            self._check_time(deadline)
            needed = max(0, gate_faces_needed - gated_faces)
            if minimum_straights[position] > straights:
                return None
            if needed > len(order) - position:
                return None
            if needed > sum(
                count
                for count, piece in zip(remaining, self._piece_choices)
                if piece.is_gate
            ):
                return None
            state = (position, tuple(remaining), straights, needed)
            if state in failed:
                return None
            return _PackingFrame(
                position,
                straights,
                gated_faces,
                state,
                available_options(position, straights, gated_faces),
            )

        def restore(option):
            for index, count in option.used:
                remaining[index] += count

        def search():
            # A room-following outline can contain thousands of faces.
            # Store the backtracking frames explicitly rather than using
            # one Python call per face and hitting the recursion limit.
            first = frame_for(0, self._straight_stock, 0)
            stack = [] if first is None else [first]
            while stack:
                self._check_time(deadline)
                frame = stack[-1]
                option = next(frame.options, None)
                if option is None:
                    failed.add(frame.state)
                    stack.pop()
                    if stack:
                        # Undo the parent's choice before trying its next option.
                        restore(selected[order[stack[-1].position]])
                    continue

                for index, count in option.used:
                    remaining[index] -= count
                selected[order[frame.position]] = option
                position = frame.position + 1
                gated_faces = frame.gated_faces + bool(option.gate_count)
                if position == len(order):
                    if gated_faces >= gate_faces_needed:
                        return True
                else:
                    following = frame_for(
                        position, frame.straights - option.straights, gated_faces
                    )
                    if following is not None:
                        stack.append(following)
                        continue

                restore(option)
            return False

        found = search()
        assignment = tuple(selected) if found else None
        self._polygon_allocations[cache_key] = assignment
        return assignment

    @staticmethod
    def _geometry(vertices):
        """Remove redundant points; recover integer faces and real turns."""
        points = [tuple(map(float, point)) for point in vertices]
        if any(
            len(point) != 2 or not all(map(math.isfinite, point)) for point in points
        ):
            return None
        if len(points) > 1 and math.dist(points[0], points[-1]) <= TOL:
            points.pop()
        changed = True
        while changed and len(points) >= 3:
            changed = False
            simplified = []
            for index, point in enumerate(points):
                previous, following = (
                    points[index - 1],
                    points[(index + 1) % len(points)],
                )
                incoming = (point[0] - previous[0], point[1] - previous[1])
                outgoing = (following[0] - point[0], following[1] - point[1])
                first_length, second_length = (
                    math.hypot(*incoming),
                    math.hypot(*outgoing),
                )
                cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
                dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1]
                if first_length <= TOL or (
                    dot > 0 and abs(cross) <= 1e-9 * first_length * second_length
                ):
                    changed = True
                else:
                    simplified.append(point)
            points = simplified
        if len(points) < 3:
            return None
        lengths, headings = [], []
        for first, second in zip(points, points[1:] + points[:1]):
            distance = math.dist(first, second)
            length = round(distance)
            if (
                abs(distance - length) > TOL
                or not MIN_WALL_LENGTH <= length <= MAX_FACE_LENGTH
            ):
                return None
            lengths.append(length)
            headings.append(
                math.degrees(math.atan2(second[1] - first[1], second[0] - first[0]))
            )
        corners = []
        for index, heading in enumerate(headings):
            turn = (heading - headings[index - 1] + 180) % 360 - 180
            magnitude = min((45, 90), key=lambda angle: abs(abs(turn) - angle))
            if abs(abs(turn) - magnitude) > ANGLE_TOL:
                return None
            kind = ConnectorType.RIGHT if magnitude == 90 else ConnectorType.DIAGONAL
            corners.append(Connector(kind, reflex=turn < 0))
        return points, lengths, headings[0], corners

    def can_allocate(self, lengths, deadline=None):
        """Check shared piece and straight stock before searching placements.

        ``None`` means the deadline prevented a decision. Completed answers
        share the same assignment cache as ``pack``. The geometry search
        must still check its particular corner types and room containment.
        """
        deadline = self.deadline if deadline is None else min(deadline, self.deadline)
        if time.process_time() >= deadline:
            return None
        lengths = tuple(lengths)
        if len(lengths) < 3 or not self.inventory.has_gate():
            return False
        corners_available = sum(
            self.inventory.connectors.get(kind, 0)
            for kind in (ConnectorType.RIGHT, ConnectorType.DIAGONAL)
        )
        if len(lengths) > corners_available:
            return False
        if any(
            not math.isfinite(length)
            or abs(length - round(length)) > TOL
            or round(length) not in self.feasible_lengths
            for length in lengths
        ):
            return False
        lengths = tuple(sorted(round(length) for length in lengths))
        if sum(lengths) > self.total_length:
            return False
        try:
            return self._assign_faces(lengths, 1, deadline) is not None
        except _DeadlineReached:
            return None

    def pack(self, vertices, deadline=None):
        """Return a validated allocation, preferring the second-gate bonus.

        If the deadline expires while looking for a better gate allocation,
        the already validated fallback is returned. An interrupted search is
        never cached as an inventory impossibility.
        """
        deadline = self.deadline if deadline is None else min(deadline, self.deadline)
        if time.process_time() >= deadline or not self.inventory.has_gate():
            return None
        geometry = self._geometry(vertices)
        if geometry is None:
            return None
        points, lengths, heading, corners = geometry
        needed = Counter(corner.connector_type for corner in corners)
        if any(
            self.inventory.connectors.get(kind, 0) < count
            for kind, count in needed.items()
        ):
            return None
        if sum(lengths) > self.total_length or any(
            length not in self.feasible_lengths for length in lengths
        ):
            return None
        polygon = Polygon(points)
        if (
            not polygon.is_valid
            or polygon.area <= 0
            or not self._prepared_room.contains(polygon)
        ):
            return None

        # Geometric order is irrelevant to resource feasibility. Canonical
        # length order allows placement and shape searches to reuse results.
        order = sorted(range(len(lengths)), key=lengths.__getitem__)
        sorted_lengths = tuple(lengths[index] for index in order)

        def construct(assignment):
            faces = [None] * len(lengths)
            for index, allocation in zip(order, assignment):
                faces[index] = allocation
            pieces, connectors = [], []
            for face, corner in zip(faces, corners):
                pieces.extend(face.pieces)
                connectors.extend(
                    [corner] + [Connector(ConnectorType.STRAIGHT)] * face.straights
                )
            candidate = Construction(points[0], heading, pieces, connectors)
            result = validate_construction(candidate, self.room, self.inventory)
            return candidate if result.valid else None

        fallback = None
        try:
            assignment = self._assign_faces(sorted_lengths, 1, deadline)
            if assignment is None:
                return None
            fallback = construct(assignment)
            gate_faces = sum(bool(face.gate_count) for face in assignment)
            if fallback is None or gate_faces >= 2 or self.max_possible_gate_bonus <= 0:
                return fallback
            assignment = self._assign_faces(sorted_lengths, 2, deadline)
            if assignment is not None:
                return construct(assignment) or fallback
        except _DeadlineReached:
            pass
        return fallback


def pack_polygon(vertices, inventory, room, weights, deadline=math.inf):
    """Convenience wrapper; reuse ``PolygonPacker`` for repeated candidates."""
    return PolygonPacker(inventory, room, weights, deadline).pack(vertices)
