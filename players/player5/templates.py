"""Search complete, stock-feasible polygons before growing local details.

The families below are geometric descriptions, independent of the room's
name or coordinates. Opposite directions have balanced total lengths, so
every shape closes exactly. The stock allocator splits faces into pieces.
"""

import heapq
import math
import time
from dataclasses import dataclass

from shapely import constrained_delaunay_triangles, union_all
from shapely.geometry import Polygon, box
from shapely.prepared import prep

from src.constants import BASELINE_SCORE, MAX_FACE_LENGTH, TOL
from src.enclosure import score_construction, validate_construction
from src.pieces import ConnectorType

from .packing import PolygonPacker

DIAGONAL_COMPONENT = 1 / math.sqrt(2)
SHAPE_SHORTLIST = 384
FOOTPRINT_SHORTLIST = 96
STOCK_CHECK_SLICE = 0.01
PACKING_SLICE = 0.05


@dataclass(frozen=True)
class Shape:
    family: str
    dimensions: tuple[int, ...]
    area: float
    perimeter: float

    def box_dimensions(self):
        """Width and height used to preserve different footprint shapes."""
        if self.family in ("rectangle", "opposite corner cuts", "four corner cuts"):
            return self.dimensions[:2]
        if self.family == "hexagon":
            width, height, diagonal = self.dimensions
            offset = diagonal * DIAGONAL_COMPONENT
            return width + offset, height + offset
        if self.family == "octagon":
            width, height, first, second = self.dimensions
            offset = (first + second) * DIAGONAL_COMPONENT
            return width + offset, height + offset
        if self.family == "staggered rectangle":
            first, second, _, _, diagonal, height = self.dimensions
            offset = diagonal * DIAGONAL_COMPONENT
            return first + second + offset, height + offset
        raise ValueError(f"unknown complete-shape family: {self.family}")

    def face_lengths(self):
        """Integer face totals, before distributing shared physical stock."""
        if self.family == "rectangle":
            return self.dimensions * 2
        if self.family == "hexagon":
            width, height, diagonal = self.dimensions
            return (width, diagonal, height) * 2
        if self.family == "octagon":
            width, height, first, second = self.dimensions
            return (width, first, height, second) * 2
        if self.family == "opposite corner cuts":
            width, height, cut_width, cut_height = self.dimensions
            return (width - cut_width, height - cut_height, cut_width, cut_height) * 2
        if self.family == "four corner cuts":
            width, height, left, right, bottom, top = self.dimensions
            return (
                width - left - right,
                height - bottom - top,
                left,
                right,
                bottom,
                top,
            ) * 2
        if self.family == "staggered rectangle":
            first, second, third, fourth, diagonal, height = self.dimensions
            return (first, diagonal, second, height, third, diagonal, fourth, height)
        raise ValueError(f"unknown complete-shape family: {self.family}")

    def vertices(self):
        """Return the counterclockwise corners in the shape's own coordinates."""
        if self.family == "rectangle":
            width, height = self.dimensions
            return [(0, 0), (width, 0), (width, height), (0, height)]
        if self.family == "hexagon":
            width, height, diagonal = self.dimensions
            offset = diagonal * DIAGONAL_COMPONENT
            return [
                (0, 0),
                (width, 0),
                (width + offset, offset),
                (width + offset, height + offset),
                (offset, height + offset),
                (0, height),
            ]
        if self.family == "octagon":
            width, height, first_diagonal, second_diagonal = self.dimensions
            first = first_diagonal * DIAGONAL_COMPONENT
            second = second_diagonal * DIAGONAL_COMPONENT
            return [
                (second, 0),
                (second + width, 0),
                (second + width + first, first),
                (second + width + first, first + height),
                (width + first, first + height + second),
                (first, first + height + second),
                (0, second + height),
                (0, second),
            ]
        if self.family == "opposite corner cuts":
            width, height, cut_width, cut_height = self.dimensions
            return [
                (cut_width, 0),
                (width, 0),
                (width, height - cut_height),
                (width - cut_width, height - cut_height),
                (width - cut_width, height),
                (0, height),
                (0, cut_height),
                (cut_width, cut_height),
            ]
        if self.family == "staggered rectangle":
            first, second, third, fourth, diagonal, height = self.dimensions
            offset = diagonal * DIAGONAL_COMPONENT
            width = first + second + offset
            return [
                (0, 0),
                (first, 0),
                (first + offset, offset),
                (width, offset),
                (width, height + offset),
                (width - third, height + offset),
                (fourth, height),
                (0, height),
            ]
        if self.family != "four corner cuts":
            raise ValueError(f"unknown complete-shape family: {self.family}")
        width, height, left, right, bottom, top = self.dimensions
        return [
            (left, 0),
            (width - right, 0),
            (width - right, bottom),
            (width, bottom),
            (width, height - top),
            (width - right, height - top),
            (width - right, height),
            (left, height),
            (left, height - top),
            (0, height - top),
            (0, bottom),
            (left, bottom),
        ]


def _clip_half_plane(vertices, normal_x, normal_y, bound):
    """Clip a convex translation region to normal.dot(point) <= bound."""
    if not vertices:
        return []
    clipped = []
    previous = vertices[-1]
    previous_gap = normal_x * previous[0] + normal_y * previous[1] - bound
    for current in vertices:
        current_gap = normal_x * current[0] + normal_y * current[1] - bound
        previous_inside = previous_gap <= 1e-9
        current_inside = current_gap <= 1e-9
        if previous_inside != current_inside:
            fraction = previous_gap / (previous_gap - current_gap)
            clipped.append(
                (
                    previous[0] + fraction * (current[0] - previous[0]),
                    previous[1] + fraction * (current[1] - previous[1]),
                )
            )
        if current_inside:
            clipped.append(current)
        previous, previous_gap = current, current_gap
    return clipped


class Placement:
    """Find contained translations of a fixed shape at one heading."""

    def __init__(self, room, angle):
        self.angle = angle
        radians = math.radians(angle)
        self.cosine, self.sine = math.cos(radians), math.sin(radians)
        self.room_vertices = [
            self.to_local(point) for point in room.exterior.coords[:-1]
        ]
        self.room = Polygon(self.room_vertices)
        buffered_room = self.room.buffer(TOL)
        self.bounds = buffered_room.bounds
        self.convex = abs(self.room.convex_hull.area - self.room.area) < 1e-7
        self.prepared = prep(buffered_room)
        self.prepared_world = prep(room.buffer(TOL))
        self.void_triangles = None
        self.normals = []
        if self.convex:
            vertices = self.room_vertices
            if not self.room.exterior.is_ccw:
                vertices = list(reversed(vertices))
            for start, end in zip(vertices, vertices[1:] + vertices[:1]):
                normal = (end[1] - start[1], start[0] - end[0])
                bound = normal[0] * start[0] + normal[1] * start[1]
                self.normals.append((*normal, bound))

    def to_local(self, point):
        x, y = point
        return (x * self.cosine + y * self.sine, -x * self.sine + y * self.cosine)

    def to_world(self, point):
        x, y = point
        return (x * self.cosine - y * self.sine, x * self.sine + y * self.cosine)

    def _translation_regions(self, vertices, deadline):
        """Subtract translations that intersect the space outside the room.

        The room's bounding box supplies a finite outside region. Constrained
        triangulation covers that entire region, including its narrow concave
        parts. A triangle plus the reflected shape describes every translation
        where those two geometries overlap. Their union is the forbidden region.
        Nonconvex shapes are first split into triangles by the same procedure.

        A zero-area set of boundary contacts can disappear during polygon
        subtraction. If no region remains, small inward offsets of the
        forbidden pieces propose nearby contact placements. This fallback is
        only a proposal: ``fit`` checks every point in official world geometry.
        """
        if time.process_time() >= deadline:
            return []
        if self.void_triangles is None:
            outside = box(*self.room.bounds).difference(self.room)
            self.void_triangles = [
                list(triangle.exterior.coords[:-1])
                for triangle in constrained_delaunay_triangles(outside).geoms
            ]
        shape = Polygon(vertices)
        if abs(shape.convex_hull.area - shape.area) < 1e-8:
            parts = [vertices]
        else:
            parts = [
                list(triangle.exterior.coords[:-1])
                for triangle in constrained_delaunay_triangles(shape).geoms
            ]
        forbidden = []
        for triangle in self.void_triangles:
            if time.process_time() >= deadline:
                return []
            for part in parts:
                # Both summands are convex, so the hull of all pairwise vertex
                # sums is their exact Minkowski sum.
                forbidden.append(
                    Polygon(
                        [(x - u, y - v) for x, y in triangle for u, v in part]
                    ).convex_hull
                )
        min_x, min_y, max_x, max_y = self.room.bounds
        xs, ys = zip(*vertices)
        left, right = min_x - min(xs), max_x - max(xs)
        bottom, top = min_y - min(ys), max_y - max(ys)
        if left >= right or bottom >= top:
            return []
        if time.process_time() >= deadline:
            return []
        domain = box(left, bottom, right, top)
        forbidden_region = union_all(forbidden)
        regions = domain.difference(forbidden_region)
        if regions.is_empty and time.process_time() < deadline:
            relaxed_obstacles = []
            for obstacle in forbidden:
                if time.process_time() >= deadline:
                    return []
                relaxed_obstacles.append(obstacle.buffer(-TOL / 4, quad_segs=1))
            regions = domain.difference(union_all(relaxed_obstacles))
        return list(regions.geoms) if hasattr(regions, "geoms") else [regions]

    def fit(self, vertices, deadline=math.inf):
        """Yield up to three distinct contained placements of these corners."""
        xs, ys = zip(*vertices)
        min_x, min_y, max_x, max_y = self.bounds
        left, right = min_x - min(xs), max_x - max(xs)
        bottom, top = min_y - min(ys), max_y - max(ys)
        if left > right + TOL or bottom > top + TOL:
            return
        if self.convex:
            translations = [(left, bottom), (right, bottom), (right, top), (left, top)]
            for normal_x, normal_y, bound in self.normals:
                support = max(normal_x * x + normal_y * y for x, y in vertices)
                allowance = TOL * math.hypot(normal_x, normal_y)
                translations = _clip_half_plane(
                    translations, normal_x, normal_y, bound - support + allowance
                )
                if not translations:
                    return
            center = (
                sum(x for x, _ in translations) / len(translations),
                sum(y for _, y in translations) / len(translations),
            )
            anchors = [center, *translations]
        else:
            regions = self._translation_regions(vertices, deadline)
            if time.process_time() >= deadline:
                return
            anchors = []
            for region in sorted(
                regions, key=lambda geometry: geometry.area, reverse=True
            ):
                if region.is_empty:
                    continue
                point = region.representative_point()
                anchors.append((point.x, point.y))
                center = region.centroid
                if region.covers(center):
                    anchors.append((center.x, center.y))
            anchors.extend(
                [
                    ((left + right) / 2, (bottom + top) / 2),
                    (left, bottom),
                    (left, top),
                    (right, bottom),
                    (right, top),
                ]
            )
            anchors.extend(
                (left + (right - left) * column / 4, bottom + (top - bottom) * row / 4)
                for column in range(5)
                for row in range(5)
            )
            anchors.extend(
                (room_x - x, room_y - y)
                for room_x, room_y in self.room_vertices
                for x, y in vertices
            )
        seen = set()
        accepted = 0
        for x, y in anchors:
            if time.process_time() >= deadline:
                return
            if not (left - TOL <= x <= right + TOL and bottom - TOL <= y <= top + TOL):
                continue
            key = (round(x, 5), round(y, 5))
            if key in seen:
                continue
            seen.add(key)
            placed = [(x + point_x, y + point_y) for point_x, point_y in vertices]
            if self.prepared.covers(Polygon(placed)):
                world_vertices = [self.to_world(point) for point in placed]
                # Rotation roundoff can move a boundary-touching vertex across
                # the tolerance boundary; the simulator checks world coordinates.
                if not self.prepared_world.covers(Polygon(world_vertices)):
                    continue
                yield world_vertices
                accepted += 1
                if accepted == 3:
                    return


def _headings(room):
    """Try boundary-aligned axes first, followed by a small rotation sweep."""
    vertices = list(room.exterior.coords[:-1])
    angles = [0.0]
    for start, end in zip(vertices, vertices[1:] + vertices[:1]):
        angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 90
        if all(abs(angle - existing) > 1e-6 for existing in angles):
            angles.append(angle)
    for angle in (45.0, 15.0, 30.0, 60.0, 75.0):
        if all(abs(angle - existing) > 1e-6 for existing in angles):
            angles.append(angle)
    return angles


def _axis_options(lengths, maximum, *, four_cuts):
    """Useful ways to split a box side around its inward corner cuts."""
    maximum = min(maximum, MAX_FACE_LENGTH * (3 if four_cuts else 2))
    extents = {math.floor(maximum + TOL) - offset for offset in range(5)}
    extents.update(length for length in lengths if length <= maximum + TOL)
    by_extent = {}
    for extent in sorted(extents, reverse=True):
        options = []
        for first in lengths:
            if four_cuts:
                for second in lengths:
                    middle = extent - first - second
                    if middle in lengths:
                        options.append((first + second, first, second))
            elif extent - first in lengths:
                options.append((first, first, 0))
        if options:
            smallest_cut = min(option[0] for option in options)
            # Keep alternatives with similar geometry when their stock demands differ.
            by_extent[extent] = [
                option for option in options if option[0] <= smallest_cut + 4
            ]
    return [
        (extent, *option) for extent, options in by_extent.items() for option in options
    ]


def _shapes(family, lengths, width, height, deadline):
    """Enumerate geometric dimensions; actual stock sharing is checked later."""
    if family == "rectangle":
        for horizontal in lengths:
            if horizontal > width + TOL:
                continue
            for vertical in lengths:
                if vertical <= height + TOL:
                    yield Shape(
                        family,
                        (horizontal, vertical),
                        horizontal * vertical,
                        2 * (horizontal + vertical),
                    )
        return
    if family in ("hexagon", "octagon"):
        for first in lengths:
            seconds = lengths if family == "octagon" else (0,)
            for second in seconds:
                if time.process_time() >= deadline:
                    return
                offset = (first + second) * DIAGONAL_COMPONENT
                for horizontal in lengths:
                    if horizontal + offset > width + TOL:
                        continue
                    for vertical in lengths:
                        if vertical + offset > height + TOL:
                            continue
                        area = (
                            horizontal * vertical
                            + offset * (horizontal + vertical)
                            + first * second
                        )
                        dimensions = (
                            (horizontal, vertical, first, second)
                            if second
                            else (horizontal, vertical, first)
                        )
                        yield Shape(
                            family,
                            dimensions,
                            area,
                            2 * (horizontal + vertical + first + second),
                        )
        return
    if family == "staggered rectangle":
        pairs_by_sum = {}
        for first in reversed(lengths):
            for second in lengths:
                if first >= second:
                    pairs_by_sum.setdefault(first + second, []).append((first, second))
        for diagonal in lengths:
            if time.process_time() >= deadline:
                return
            offset = diagonal * DIAGONAL_COMPONENT
            for total, pairs in pairs_by_sum.items():
                box_width = total + offset
                if box_width > width + TOL:
                    continue
                # The largest first segment places the upper diagonal near the
                # left and the lower one near the right, preserving more area.
                for first, second in pairs[:3]:
                    for third, fourth in pairs[:3]:
                        for vertical in lengths:
                            if vertical + offset <= height + TOL:
                                area = box_width * vertical + offset * (third - second)
                                yield Shape(
                                    family,
                                    (first, second, third, fourth, diagonal, vertical),
                                    area,
                                    2 * (total + diagonal + vertical),
                                )
        return
    four_cuts = family == "four corner cuts"
    horizontal_options = _axis_options(lengths, width, four_cuts=four_cuts)
    vertical_options = _axis_options(lengths, height, four_cuts=four_cuts)
    for box_width, cut_width, left, right in horizontal_options:
        if time.process_time() >= deadline:
            return
        for box_height, cut_height, bottom, top in vertical_options:
            if four_cuts:
                dimensions = (box_width, box_height, left, right, bottom, top)
                area = box_width * box_height - cut_width * cut_height
            else:
                dimensions = (box_width, box_height, left, bottom)
                area = box_width * box_height - 2 * cut_width * cut_height
                if 2 * cut_width >= box_width and 2 * cut_height >= box_height:
                    continue  # Opposite missing corners may not touch or overlap.
            if area > 0:
                yield Shape(family, dimensions, area, 2 * (box_width + box_height))


def search_templates(room, inventory, weights, deadline, consider, *, packer=None):
    """Report strong complete polygons until an absolute process-CPU deadline.

    ``consider`` receives validated constructions immediately. The return value
    is the highest-scoring construction found. A few placements per family and
    orientation are also reported, providing distinct starting points for later
    boundary refinement.
    """
    if packer is None:
        packer = PolygonPacker(inventory, room, weights, deadline)
    lengths = sorted(packer.feasible_lengths)
    if not lengths:
        return None
    right = inventory.connectors.get(ConnectorType.RIGHT, 0)
    diagonal = inventory.connectors.get(ConnectorType.DIAGONAL, 0)
    families = []
    if right >= 4:
        families.append("rectangle")
    if diagonal >= 8:
        families.append("octagon")
    if right >= 12:
        families.append("four corner cuts")
    if right >= 8:
        families.append("opposite corner cuts")
    if right >= 2 and diagonal >= 4:
        families.append("hexagon")
    if right >= 4 and diagonal >= 4:
        families.append("staggered rectangle")

    best, best_score = None, float("-inf")
    inventory_feasibility = {}
    area_limit = room.polygon.buffer(TOL).area
    for angle in _headings(room.polygon):
        placement = Placement(room.polygon, angle)
        min_x, min_y, max_x, max_y = placement.bounds
        for family in families:
            if time.process_time() >= deadline:
                return best
            # A bounded heap avoids storing hundreds of thousands of similar sizes.
            candidates = []
            footprint_candidates = {}
            order = 0
            for shape in _shapes(
                family, lengths, max_x - min_x, max_y - min_y, deadline
            ):
                if (
                    shape.area > area_limit
                    or shape.perimeter > packer.total_length + TOL
                ):
                    continue
                bound = (
                    BASELINE_SCORE
                    + weights.A * shape.area
                    + weights.C * shape.perimeter
                    + packer.max_possible_gate_bonus
                )
                if not placement.convex:
                    lengths_key = tuple(sorted(shape.face_lengths()))
                    feasible = inventory_feasibility.get(lengths_key)
                    if lengths_key not in inventory_feasibility:
                        feasible = packer.can_allocate(
                            lengths_key,
                            deadline=min(
                                deadline, time.process_time() + STOCK_CHECK_SLICE
                            ),
                        )
                        if feasible is not None:
                            inventory_feasibility[lengths_key] = feasible
                    if feasible is False:
                        continue
                item = (bound, order, shape)
                order += 1
                if len(candidates) < SHAPE_SHORTLIST:
                    heapq.heappush(candidates, item)
                elif bound > candidates[0][0]:
                    heapq.heapreplace(candidates, item)
                if not placement.convex:
                    width, height = shape.box_dimensions()
                    # Preserve elongated and heavily chamfered footprints.
                    # A single score heap can lose every shape that fits a
                    # narrow concave region, even after inventory filtering.
                    footprint = (
                        math.floor(4 * math.log2(width / height)),
                        min(3, math.floor(4 * shape.area / (width * height))),
                    )
                    heap = footprint_candidates.setdefault(footprint, [])
                    if len(heap) < FOOTPRINT_SHORTLIST:
                        heapq.heappush(heap, item)
                    elif bound > heap[0][0]:
                        heapq.heapreplace(heap, item)
            selected = {item[1]: item for item in candidates}
            for heap in footprint_candidates.values():
                selected.update((item[1], item) for item in heap)
            family_score = float("-inf")
            reported = 0
            for bound, _, shape in sorted(selected.values(), reverse=True):
                if time.process_time() >= deadline:
                    return best
                if bound < family_score - 1e-7:
                    break
                for vertices in placement.fit(shape.vertices(), deadline):
                    if time.process_time() >= deadline:
                        return best
                    candidate = packer.pack(
                        vertices,
                        deadline=min(deadline, time.process_time() + PACKING_SLICE),
                    )
                    if candidate is None:
                        continue
                    result = validate_construction(candidate, room, inventory)
                    if not result.valid:
                        continue
                    score = score_construction(result, weights)
                    improves_best = score > best_score + 1e-7
                    if improves_best:
                        best, best_score = candidate, score
                    if improves_best or (score >= family_score - 1e-7 and reported < 3):
                        consider(candidate)
                        reported += 1
                    family_score = max(family_score, score)
    return best
