"""Finding where in the room a shape fits.

A good placement nearly always touches the room's walls and lines up with them, so
the headings come from the room's own edges (PLACEMENT_GUIDE.md section 7.3). For
each heading the positions tried are, cheapest first:

1. anchors: a corner or edge midpoint of the shape on a corner or edge midpoint of
   the room, the shape's bounding box flush against the room's wall lines, and the
   shape centred on the room's centroid;
2. contacts: every position where the shape touches the room in two places at once
   (a shape corner on a room wall, or a room corner on a shape side), which is where
   a shape that only just fits has to sit;
3. the shape centred on the room's middle and turned through every heading, for a
   fit that lines up with none of the room's walls (a square in an octagon);
4. a grid of interior points.

All the candidates for one heading are tested against the room in one call.
"""

import itertools
import math

import numpy as np
import shapely

import src.constants as c
from src.room import Room

MAX_BASE_HEADINGS = 6
MAX_BOX_CANDIDATES = 20000
MAX_CONTACT_LINES = 240
GRID_STEPS = 14
SWEEP_STEP_DEG = 2.5
EPS = 1e-6


def _midpoints(points: np.ndarray) -> np.ndarray:
    return (points + np.roll(points, -1, axis=0)) / 2.0


def _edge_normals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals and lengths of a closed polygon's edges."""
    d = np.roll(points, -1, axis=0) - points
    lengths = np.hypot(d[:, 0], d[:, 1])
    return np.column_stack([-d[:, 1], d[:, 0]]) / lengths[:, None], lengths


class RoomSearch:
    def __init__(self, room: Room):
        self.polygon = room.polygon
        # the same test as Room.contains_enclosure, prepared so it is fast
        self.region = room.polygon.buffer(c.TOL)
        shapely.prepare(self.region)
        self.corners = np.array(room.get_boundary_points(), dtype=float)
        self.area = room.polygon.area
        # the radius of the biggest circle the room has space for, rounded up
        self.inradius = (
            shapely.maximum_inscribed_circle(room.polygon, tolerance=0.01).length + 0.05
        )
        centroid = room.polygon.centroid
        inside = room.polygon.representative_point()
        self.centres = np.array([[centroid.x, centroid.y], [inside.x, inside.y]])

        # the room's walls, longest first, and its inward-pointing corners, those
        # between the longest walls first: what the contact search works from
        before = self.corners - np.roll(self.corners, 1, axis=0)
        after = np.roll(self.corners, -1, axis=0) - self.corners
        cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
        reflex = cross * (1.0 if room.polygon.exterior.is_ccw else -1.0) < -EPS
        lengths = np.hypot(after[:, 0], after[:, 1])
        self._long_edges = np.argsort(-lengths)
        reach = lengths + np.roll(lengths, 1)
        self._reflex = np.array(
            sorted(np.flatnonzero(reflex), key=lambda i: -reach[i]), dtype=int
        )

        self._frames: dict[float, tuple] = {}
        self._bases: dict[int, list[float]] = {}

    def main_direction(self) -> float:
        """The room's average wall direction, modulo 90 degrees: each wall's angle
        counted four times round and weighted by its length, so walls at right
        angles agree instead of cancelling."""
        points = np.vstack([self.corners, self.corners[:1]])
        d = np.diff(points, axis=0)
        lengths = np.hypot(d[:, 0], d[:, 1])
        angles = np.arctan2(d[:, 1], d[:, 0])
        mean = np.sum(lengths * np.exp(4j * angles))
        return float(np.degrees(np.angle(mean) / 4.0) % 90.0)

    def base_headings(self, step: int) -> list[float]:
        """The directions of the room's longest edges, modulo `step` degrees."""
        if step not in self._bases:
            weight: dict[float, float] = {}
            exact: dict[float, float] = {}
            longest: dict[float, float] = {}
            points = np.vstack([self.corners, self.corners[:1]])
            for (x1, y1), (x2, y2) in itertools.pairwise(points):
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % step
                # walls within a hundredth of a degree are one direction: rooms
                # given to three decimals have walls at 22.5 and 22.501
                key = round(angle, 2) % step
                length = math.hypot(x2 - x1, y2 - y1)
                weight[key] = weight.get(key, 0.0) + length
                if length > longest.get(key, 0.0):
                    longest[key] = length
                    exact[key] = 0.0 if key == 0 else angle
            best = sorted(weight, key=lambda k: -weight[k])[:MAX_BASE_HEADINGS]
            headings = [exact[k] for k in best]
            if 0.0 not in headings:
                headings.append(0.0)
            self._bases[step] = headings
        return self._bases[step]

    def headings(self, step: int, symmetry: int) -> list[float]:
        bases = self.base_headings(step)
        return [base + turn for base in bases for turn in range(0, symmetry, step)]

    def _frame(self, heading: float) -> tuple:
        """The room as seen from a frame turned by `heading`, in which the shape
        is axis-aligned: (rotation, corners, xs, ys, low, high, grid, centres)."""
        if heading not in self._frames:
            t = math.radians(heading)
            rotation = np.array(
                [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]
            )
            corners = self.corners @ rotation
            xs = np.unique(np.round(corners[:, 0], 6))
            ys = np.unique(np.round(corners[:, 1], 6))
            low, high = corners.min(axis=0), corners.max(axis=0)
            gx, gy = np.meshgrid(
                np.linspace(low[0], high[0], GRID_STEPS + 2)[1:-1],
                np.linspace(low[1], high[1], GRID_STEPS + 2)[1:-1],
            )
            grid = np.column_stack([gx.ravel(), gy.ravel()])
            world = grid @ rotation.T
            grid = grid[shapely.contains_xy(self.polygon, world[:, 0], world[:, 1])]
            self._frames[heading] = (
                rotation,
                corners,
                xs,
                ys,
                low,
                high,
                grid,
                self.centres @ rotation,
            )
        return self._frames[heading]

    def _fitting(self, local, heading, offsets) -> list[tuple[float, float]]:
        """The start points, among `offsets` (in the turned frame), at which the
        shape lies inside the room."""
        rotation, _, _, _, low, high, _, _ = self._frame(heading)
        if len(offsets) == 0:
            return []
        offsets = np.unique(np.round(offsets, 6), axis=0)
        lo, hi = local.min(axis=0), local.max(axis=0)
        inside = np.all(offsets + lo >= low - EPS, axis=1) & np.all(
            offsets + hi <= high + EPS, axis=1
        )
        offsets = offsets[inside]
        if len(offsets) == 0:
            return []
        ring = np.vstack([local, local[:1]])
        world = (ring[None, :, :] + offsets[:, None, :]) @ rotation.T
        fits = shapely.contains(self.region, shapely.polygons(world))
        return [(float(x), float(y)) for x, y in world[fits, 0, :]]

    def _anchor_offsets(self, local, heading) -> np.ndarray:
        _, corners, xs, ys, _, _, _, centres = self._frame(heading)
        lo, hi = local.min(axis=0), local.max(axis=0)
        shape_points = np.vstack([local, _midpoints(local)])
        room_points = np.vstack([corners, _midpoints(corners)])
        offsets = [
            (room_points[:, None, :] - shape_points[None, :, :]).reshape(-1, 2),
            centres - (lo + hi) / 2.0,
        ]
        # the shape's bounding box flush against two of the room's wall lines
        bx = np.concatenate([xs - lo[0], xs - hi[0]])
        by = np.concatenate([ys - lo[1], ys - hi[1]])
        if len(bx) * len(by) <= MAX_BOX_CANDIDATES:
            mx, my = np.meshgrid(bx, by)
            offsets.append(np.column_stack([mx.ravel(), my.ravel()]))
        return np.vstack(offsets)

    def _contact_offsets(self, local, heading) -> np.ndarray:
        """Positions where the shape touches the room twice. Each single contact
        confines the shape's position to a line; two of them pin it to a point."""
        corners = self._frame(heading)[1]
        n = len(local)
        edge_budget = max(3, int(MAX_CONTACT_LINES * 0.7) // n)
        corner_budget = max(1, int(MAX_CONTACT_LINES * 0.3) // n)
        edges = self._long_edges[:edge_budget]
        poking = self._reflex[:corner_budget]

        room_normals, _ = _edge_normals(corners)
        shape_normals, _ = _edge_normals(local)
        # a corner of the shape on a wall of the room
        normals = [np.repeat(room_normals[edges], n, axis=0)]
        gaps = corners[edges][:, None, :] - local[None, :, :]
        distances = [(room_normals[edges][:, None, :] * gaps).sum(axis=2).ravel()]
        if len(poking):
            # an inward corner of the room on a side of the shape
            normals.append(np.tile(shape_normals, (len(poking), 1)))
            gaps = corners[poking][:, None, :] - local[None, :, :]
            distances.append((shape_normals[None, :, :] * gaps).sum(axis=2).ravel())
        lines = np.unique(
            np.round(
                np.column_stack([np.vstack(normals), np.concatenate(distances)]), 7
            ),
            axis=0,
        )

        i, j = np.triu_indices(len(lines), 1)
        det = lines[i, 0] * lines[j, 1] - lines[i, 1] * lines[j, 0]
        crossing = np.abs(det) > 1e-9
        i, j, det = i[crossing], j[crossing], det[crossing]
        x = (lines[i, 2] * lines[j, 1] - lines[j, 2] * lines[i, 1]) / det
        y = (lines[i, 0] * lines[j, 2] - lines[j, 0] * lines[i, 2]) / det
        return np.column_stack([x, y])

    def _grid_offsets(self, local, heading) -> np.ndarray:
        centre = (local.min(axis=0) + local.max(axis=0)) / 2.0
        return self._frame(heading)[6] - centre

    def _centred_sweep(
        self, local, symmetry
    ) -> list[tuple[tuple[float, float], float]]:
        """The shape centred on the room's middle, turned through every heading:
        the one way to find a fit that lines up with none of the room's walls."""
        ring = np.vstack([local, local[:1]])
        ring = ring - (local.min(axis=0) + local.max(axis=0)) / 2.0
        angles = np.arange(0.0, symmetry, SWEEP_STEP_DEG)
        t = np.radians(angles)
        rotations = np.stack(
            [
                np.stack([np.cos(t), np.sin(t)], axis=1),
                np.stack([-np.sin(t), np.cos(t)], axis=1),
            ],
            axis=1,
        )  # ring @ rotations[k] turns the ring by angles[k]
        turned = ring[None, :, :] @ rotations
        found = []
        for centre in self.centres:
            world = turned + centre
            fits = shapely.contains(self.region, shapely.polygons(world))
            found += [
                ((float(x), float(y)), float(angle))
                for (x, y), angle in zip(world[fits, 0, :], angles[fits])
            ]
        return found

    def place(
        self,
        vertices: list[tuple[float, float]],
        step: int,
        symmetry: int,
        limit: int = 3,
        only: list[float] | None = None,
    ) -> list[tuple[tuple[float, float], float]]:
        """Up to `limit` (start, heading) pairs at which the shape fits. `step` and
        `symmetry` say which headings are worth trying (see `headings`); `only`
        replaces them with a list of its own."""
        local = np.array(vertices, dtype=float)
        headings = only if only is not None else self.headings(step, symmetry)
        found: list[tuple[tuple[float, float], float]] = []
        for candidates in (self._anchor_offsets, self._contact_offsets):
            for heading in headings:
                offsets = candidates(local, heading)
                for start in self._fitting(local, heading, offsets):
                    found.append((start, heading))
                    if len(found) >= limit:
                        return found
            if found:
                return found
        found = self._centred_sweep(local, symmetry)[:limit]
        if found:
            return found
        for heading in headings:
            offsets = self._grid_offsets(local, heading)
            for start in self._fitting(local, heading, offsets):
                found.append((start, heading))
                if len(found) >= limit:
                    return found
        return found
