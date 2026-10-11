from collections import Counter

from shapely import Polygon

from players.player0 import Player
from src.enclosure import Construction
from src.pieces import Connector, ConnectorType, Piece, PieceType

from players.player6.polygon import find_fit
import math
import time


class Player6(Player):
    """Group 6: <one line on your strategy -- fill this in>."""

    GATE = 0
    WALL = 1
    # Stop searching this many CPU seconds into the run and return the best
    # enclosure found so far (the hard limit is 300s; leave headroom, the
    # last fit check / construction still has to run).
    TIME_BUDGET = 240.0
    # Perimeter band added per iteration of the grow-until-it-no-longer-fits loop.
    PERIMETER_STEP = 8
    # Consecutive bands with candidates, none fitting, before we stop growing.
    PATIENCE = 2
    STEPS = (
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
    )

    class OutOfTime(Exception):
        pass

    def __init__(self, *args, **kwargs) -> None:
        # process_time() is what the simulator's CPU limit counts; the
        # constructor counts toward it, so start the clock here.
        self._start = time.process_time()
        super().__init__(*args, **kwargs)

    def time_left(self) -> float:
        return self.TIME_BUDGET - (time.process_time() - self._start)

    def out_of_time(self) -> bool:
        return self.time_left() <= 0

    class Edge:
        """Represent one straight face, possibly assembled from multiple walls
        and gates using straight connectors."""

        def __init__(
            self, pieces: list[tuple[int, int]], length: int, straight_count: int
        ) -> None:
            self.pieces = pieces.copy()
            self.length = length
            self.straight_count = straight_count

        def __str__(self) -> str:
            return f"Edge{{pieces={self.pieces}, length={self.length:d}, straight_count={self.straight_count:d}}}"

    class Shape:
        """Store a candidate boundary in traversal order.
        turns[i] is the turn after edges[i], including the closing joint."""

        def __init__(
            self,
            edges: list["Player6.Edge"],
            turns: list[int],
            vertices: list[tuple[float, float]],
        ) -> None:
            self.edges = edges.copy()
            self.turns = turns.copy()
            self.vertices = vertices.copy()

        def __str__(self) -> str:
            edges_text: str = ", ".join(str(edge) for edge in self.edges)
            return f"Shape{{edges=[{edges_text}], turns={self.turns}, vertices={self.vertices}}}"

        def to_polygon(self) -> Polygon:
            return Polygon(self.vertices)

        def canonical(self) -> "Player6.Shape":
            """Rotate the loop so it starts at its lexicographically smallest
            (length, turn) sequence, and rebuild vertices in that frame. Equal
            outlines then yield identical polygons, so a fit found for one
            applies to all of them."""
            seq = [(e.length, t) for e, t in zip(self.edges, self.turns)]
            n = len(seq)
            k = min(range(n), key=lambda i: seq[i:] + seq[:i])
            edges = self.edges[k:] + self.edges[:k]
            turns = self.turns[k:] + self.turns[:k]
            x = y = 0
            heading = 0
            vertices = [(0, 0)]
            for e, t in zip(edges, turns):
                dx, dy = Player6.STEPS[heading]
                x, y = x + dx * e.length, y + dy * e.length
                vertices.append((x, y))
                heading = (heading + t // 90) % 4
            return Player6.Shape(edges, turns, vertices[:-1])

    class Materials:
        """Track remaining piece counts, connector counts, and total piece length."""

        def __init__(
            self, materials: list[list[int]], right_count: int, straight_count: int
        ) -> None:
            self.materials = materials
            self.total_piece_length = 0
            for material in materials:
                self.total_piece_length += material[1] * material[2]
            self.right_count = right_count
            self.straight_count = straight_count

        def __len__(self) -> int:
            return len(self.materials)

        def __getitem__(self, index: int) -> list[int]:
            return self.materials[index]

        def can_supply(self, edge: "Player6.Edge") -> bool:
            """Check aggregate piece demand and straight connectors without
            modifying the inventory."""
            if self.straight_count < edge.straight_count:
                return False

            needed = Counter(edge.pieces)
            available = Counter()

            for material_type, material_length, material_count in self.materials:
                available[(material_type, material_length)] += material_count

            return all(available[key] >= count for key, count in needed.items())

        def consume(self, edge: "Player6.Edge") -> None:
            """Deduct the pieces and straight connectors used by one edge.
            The corner connector is deducted separately during shape search."""
            if not self.can_supply(edge):
                raise ValueError("Not enough materials or straight connectors")

            needed = Counter(edge.pieces)
            for material in self.materials:
                key = (material[0], material[1])
                take = min(material[2], needed[key])

                material[2] -= take
                needed[key] -= take

            self.straight_count -= edge.straight_count
            self.total_piece_length -= edge.length

        def clone(self) -> "Player6.Materials":
            """Deep-copy the inventory so DFS branches cannot modify each other's
            remaining material counts."""
            return Player6.Materials(
                materials=[material.copy() for material in self.materials],
                right_count=self.right_count,
                straight_count=self.straight_count,
            )

    @staticmethod
    def is_simple_polygon(vertices: list[tuple[float, float]]) -> bool:
        """Require explicit closure and reject self-intersections, overlapping
        edges, and zero-area polygons. Concave polygons are allowed."""
        if len(vertices) < 4:
            return False
        if vertices[0] != vertices[-1]:
            return False

        polygon = Polygon(vertices)
        return polygon.is_valid and polygon.area > 0

    @staticmethod
    def generate_edges(materials: Materials) -> list["Player6.Edge"]:
        """Enumerate material combinations for straight faces of length at most 30.
        Use a fixed material order to avoid duplicate permutations."""
        options: list[Player6.Edge] = []

        def dfs(index: int, selected: list[tuple[int, int]], total_length: int) -> None:
            if index == len(materials):
                if selected:
                    options.append(
                        Player6.Edge(
                            pieces=selected.copy(),
                            length=total_length,
                            straight_count=len(selected) - 1,
                        )
                    )
                return

            material: list[int] = materials[index]
            material_type: int = material[0]
            material_length: int = material[1]
            material_count: int = material[2]

            # Bound by stock, the 30-unit face limit, and available straight
            # connectors: k pieces require k - 1 straight connectors.
            max_count: int = min(
                material_count,
                (30 - total_length) // material_length,
                materials.straight_count + 1 - len(selected),
            )

            for count in range(max_count + 1):
                dfs(
                    index=(index + 1),
                    selected=(selected + [(material_type, material_length)] * count),
                    total_length=(total_length + count * material_length),
                )

        dfs(index=0, selected=[], total_length=0)

        return options

    def generate_shapes(
        self,
        materials: Materials,
        edge_options: list[Edge],
        max_faces: int,
        perimeter_lo: float,
        perimeter_hi: float,
    ) -> list[Shape]:
        """Enumerate closed orthogonal polygons (counter-clockwise, using left
        and right 90-degree turns) whose perimeter is in (perimeter_lo,
        perimeter_hi]. Enforces inventory limits, a gate, and polygon validity.
        If the CPU budget runs out mid-search, OutOfTime is raised and the
        shapes found so far are left in self.found_partial."""
        shapes: list[Player6.Shape] = []
        self.found_partial = shapes
        if materials.right_count < 4:
            return shapes
        max_faces = min(max_faces, materials.right_count)
        # Shortest edges first so `perimeter_hi` pruning can break early.
        edge_options = sorted(edge_options, key=lambda e: e.length)
        seen: set = set()

        def dfs(
            position, heading, edges, turns, vertices, total_turn, perimeter, remaining
        ):
            if self.out_of_time():
                raise Player6.OutOfTime()
            if len(edges) >= max_faces or remaining.right_count == 0:
                return
            dx, dy = Player6.STEPS[heading]
            for edge in edge_options:
                if perimeter + edge.length > perimeter_hi:
                    break  # sorted by length: nothing later fits either
                if not remaining.can_supply(edge):
                    continue

                next_position = (
                    position[0] + dx * edge.length,
                    position[1] + dy * edge.length,
                )
                next_perimeter = perimeter + edge.length
                next_remaining = remaining.clone()
                next_remaining.consume(edge)
                next_remaining.right_count -= 1
                next_edges = edges + [edge]
                next_vertices = vertices + [next_position]

                # Headings use quarter-turn units: +1 is left, -1 is right.
                for turn in (1, -1):
                    next_heading = (heading + turn) % 4
                    next_total_turn = total_turn + turn
                    next_turns = turns + [turn * 90]

                    if next_position == (0, 0):
                        if len(next_edges) < 4 or next_heading != 0:
                            continue
                        if next_total_turn != 4:  # counter-clockwise only
                            continue
                        if next_perimeter <= perimeter_lo:
                            continue
                        if not any(
                            piece_type == Player6.GATE
                            for e in next_edges
                            for piece_type, _ in e.pieces
                        ):
                            continue
                        if not Player6.is_simple_polygon(next_vertices):
                            continue
                        key = Player6.canonical_key(
                            next_edges, next_turns, geometry=False
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        shapes.append(
                            Player6.Shape(
                                edges=next_edges,
                                turns=next_turns,
                                vertices=next_vertices[:-1],
                            )
                        )
                    else:
                        # Must be able to get home: Manhattan distance back
                        # to the origin vs. what we can still lay down.
                        need = abs(next_position[0]) + abs(next_position[1])
                        can = min(
                            30 * (max_faces - len(next_edges)),
                            next_remaining.total_piece_length,
                            perimeter_hi - next_perimeter,
                        )
                        if need > can:
                            continue
                        dfs(
                            next_position,
                            next_heading,
                            next_edges,
                            next_turns,
                            next_vertices,
                            next_total_turn,
                            next_perimeter,
                            next_remaining,
                        )

        dfs((0, 0), 0, [], [], [(0, 0)], 0, 0, materials.clone())
        return shapes

    @staticmethod
    def canonical_key(edges, turns, geometry: bool):
        """Rotation-invariant key of a loop. geometry=True ignores how each
        face is split into pieces, so equal outlines share one fit check."""
        if geometry:
            seq = [(e.length, t) for e, t in zip(edges, turns)]
        else:
            seq = [(tuple(sorted(e.pieces)), t) for e, t in zip(edges, turns)]
        n = len(seq)
        return min(tuple(seq[i:] + seq[:i]) for i in range(n))

    MAX_FACES = 6

    def prepare_inventory(self) -> None:
        straight_count = self.inventory.connectors.get(ConnectorType.STRAIGHT, 0)
        right_count = self.inventory.connectors.get(ConnectorType.RIGHT, 0)
        core: list[list[int]] = []
        for wall, n in self.inventory.walls.items():
            core.append([Player6.WALL, wall, n])
        for gate, n in self.inventory.gates.items():
            core.append([Player6.GATE, gate, n])
        self.materials = Player6.Materials(
            materials=core, straight_count=straight_count, right_count=right_count
        )
        self.edge_options = Player6.generate_edges(materials=self.materials)

    def shape_score(self, shape: Shape, polygon: Polygon) -> float:
        score = self.weights.A * polygon.area + self.weights.C * polygon.length
        gate_faces = sum(
            1 for e in shape.edges if any(t == Player6.GATE for t, _ in e.pieces)
        )
        if gate_faces >= 2:
            score += self.weights.G
        return score

    def fit_angles(self) -> list[float]:
        """Rotations worth trying: 0 plus each room edge's direction mod 90
        (an orthogonal shape is tightest with a side flush on a room wall)."""
        if getattr(self, "_angles", None) is None:
            coords = list(self.room.polygon.exterior.coords)
            angles = {0.0}
            for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
                a = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 90.0
                angles.add(round(a, 6) % 90.0)
            self._angles = sorted(angles)
        return self._angles

    def search(self):
        """Grow the enclosure iteratively: look at shapes in perimeter bands
        (small first), fit-check each, and keep the best-scoring one. Stop
        when bands stop fitting in the room, the inventory is exhausted, or
        the CPU budget is nearly spent -- whichever comes first."""
        self.prepare_inventory()
        room_poly = self.room.polygon
        room_area = room_poly.area
        max_perimeter = self.materials.total_piece_length
        fit_cache: dict = {}
        best = None  # (score, shape, fit)
        misses = 0
        lo = 0
        while lo < max_perimeter and not self.out_of_time():
            hi = lo + self.PERIMETER_STEP
            timed_out = False
            try:
                shapes = self.generate_shapes(
                    self.materials, self.edge_options, self.MAX_FACES, lo, hi
                )
            except Player6.OutOfTime:
                shapes = self.found_partial
                timed_out = True

            tried = fitted = 0
            for shape in shapes:
                if self.out_of_time():
                    timed_out = True
                    break
                shape = shape.canonical()
                poly = shape.to_polygon()
                if poly.area > room_area:  # cheap reject before the fit search
                    continue
                gkey = Player6.canonical_key(shape.edges, shape.turns, geometry=True)
                if gkey not in fit_cache:
                    tried += 1
                    fit_cache[gkey] = find_fit(
                        room_poly, poly, angles=self.fit_angles(), stop=self.out_of_time
                    )
                fit = fit_cache[gkey]
                if fit is None:
                    continue
                fitted += 1
                score = self.shape_score(shape, poly)
                if best is None or score > best[0]:
                    best = (score, shape, fit)
            if timed_out:
                self.note = "CPU budget reached; returning best found so far"
                break
            if tried:
                misses = 0 if fitted else misses + 1
                if misses >= self.PATIENCE:
                    break
            lo = hi
        return best

    def build_enclosure(self) -> Construction | None:
        best = self.search()
        if best is None or best[0] <= 0:
            return None
        _, candidate, (fit, params) = best

        coords = list(fit.exterior.coords)
        (x1, y1), (x2, y2) = coords[0], coords[1]
        angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))

        pieces = []
        connectors = []
        for e, t in zip(candidate.edges, candidate.turns):
            for piece_type, length in e.pieces:
                pieces.append(
                    Piece(
                        piece_type=PieceType.GATE
                        if piece_type == Player6.GATE
                        else PieceType.WALL,
                        length=length,
                    )
                )
            for _ in range(e.straight_count):
                connectors.append(Connector(connector_type=ConnectorType.STRAIGHT))
            # +90 (left turn) is an unflipped right connector; -90 is flipped.
            connectors.append(
                Connector(connector_type=ConnectorType.RIGHT, reflex=(t < 0))
            )

        # connectors[i] is the joint *before* piece i (connectors[0] closes the
        # loop), while the list above is built as the joint *after* each piece.
        connectors = connectors[-1:] + connectors[:-1]

        return Construction(
            start=(x1, y1),
            start_heading=angle_deg,
            pieces=pieces,
            connectors=connectors,
        )
