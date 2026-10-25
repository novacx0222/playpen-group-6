"""Search complete shapes, follow the room, then improve valid enclosures."""

import math
import signal
import time
from dataclasses import dataclass, field
from functools import partial

from players.player0 import Player
from players.player5.refinement import _outline_key, _simplify_outline
from src.constants import MAX_FACE_LENGTH, MIN_WALL_LENGTH
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import ConnectorType

MAX_STARTING_SHAPES = 12


@dataclass
class SearchStats:
    candidates: int = 0
    valid_candidates: int = 0
    first_valid_cpu: float | None = None  # First valid construction scoring above zero.
    best_stage: str = "none"
    stage_cpu: dict[str, float] = field(default_factory=dict)
    best_after_stage: dict[str, float] = field(default_factory=dict)
    cpu: float = 0.0


@dataclass
class _StartingShape:
    """A valid fence retained as a starting point for later improvements."""

    construction: Construction
    score: float
    category: tuple[int, ...]
    outline: tuple[tuple[float, float], ...]


class Player5(Player):
    """Keep the best official score found by complementary geometry searches.

    The default budget is 240 CPU seconds. When the simulator supplies a
    smaller CPU limit, set an internal deadline at 80% of the remaining time.
    Deadline checks are cooperative. Every search reports complete enclosures
    as it finds them, so an unfinished phase never replaces the best valid result.
    """

    def __init__(self, room, inventory, weights, *, cpu_budget=240.0):
        super().__init__(room, inventory, weights)
        if not math.isfinite(cpu_budget) or cpu_budget <= 0:
            raise ValueError("cpu_budget must be finite and positive")
        self.cpu_budget = float(cpu_budget)
        self.stats = SearchStats()
        self.best_score = 0.0
        self._best_construction = None
        self._starting_shapes = []

    def _keep_candidate(self, construction, stage, started):
        """Save the best positive official score and useful starting shapes."""
        if construction is None:
            return
        self.stats.candidates += 1
        checked = validate_construction(construction, self.room, self.inventory)
        if not checked.valid:
            return
        self.stats.valid_candidates += 1
        score = score_construction(checked, self.weights)
        if score <= 0:
            return  # Returning None earns zero.
        if self.stats.first_valid_cpu is None:
            self.stats.first_valid_cpu = time.process_time() - started
        if score > self.best_score:
            self._best_construction, self.best_score = construction, score
            self.stats.best_stage = stage
        self._remember_starting_shape(construction, checked.polygon, score)

    def _remember_starting_shape(self, construction, polygon, score):
        """Keep up to twelve outlines with varied positions and connector use."""
        # Ignore straight piece junctions, just as the refinement search does.
        outline = _outline_key(_simplify_outline(polygon.exterior.coords))
        existing = next(
            (shape for shape in self._starting_shapes if shape.outline == outline),
            None,
        )
        if existing is None:
            left, bottom, right, top = self.room.polygon.bounds
            center = polygon.centroid
            # Position and groups of four corners distinguish useful alternatives.
            category = (
                int(4 * (center.x - left) / max(right - left, 1.0)),
                int(4 * (center.y - bottom) / max(top - bottom, 1.0)),
                sum(
                    joint.connector_type == ConnectorType.RIGHT
                    for joint in construction.connectors
                )
                // 4,
                sum(
                    joint.connector_type == ConnectorType.DIAGONAL
                    for joint in construction.connectors
                )
                // 4,
            )
            self._starting_shapes.append(
                _StartingShape(construction, score, category, outline)
            )
        elif score > existing.score:
            existing.construction = construction
            existing.score = score
        else:
            return

        self._starting_shapes.sort(key=lambda candidate: candidate.score, reverse=True)
        diverse, seen_keys = [], set()
        for candidate in self._starting_shapes:
            if candidate.category not in seen_keys:
                diverse.append(candidate)
                seen_keys.add(candidate.category)
        selected = {candidate.outline for candidate in diverse}
        diverse.extend(
            candidate
            for candidate in self._starting_shapes
            if candidate.outline not in selected
        )
        self._starting_shapes = diverse[:MAX_STARTING_SHAPES]

    def build_enclosure(self):
        from players.player5.packing import PolygonPacker
        from players.player5.path_refinement import refine_paths
        from players.player5.refinement import refine
        from players.player5.room_grid import search_grid
        from players.player5.templates import search_templates

        started = time.process_time()
        budget = self.cpu_budget
        if hasattr(signal, "getitimer") and hasattr(signal, "ITIMER_PROF"):
            remaining, _ = signal.getitimer(signal.ITIMER_PROF)
            if remaining > 0:
                budget = min(budget, 0.8 * remaining)
        deadline = started + budget
        context = self.room, self.inventory, self.weights
        self.stats = SearchStats()
        self.best_score, self._best_construction, self._starting_shapes = 0.0, None, []

        def run_stage(name, function, allowance):
            """Give one search a deadline and the callback that saves its results."""
            before = time.process_time()
            stage_deadline = min(deadline, before + allowance)
            if stage_deadline <= before:
                return
            function(
                stage_deadline,
                lambda construction: self._keep_candidate(construction, name, started),
            )
            self.stats.stage_cpu[name] = time.process_time() - before
            self.stats.best_after_stage[name] = self.best_score

        legal_gates = sum(
            count
            for length, count in self.inventory.gates.items()
            if MIN_WALL_LENGTH <= length <= MAX_FACE_LENGTH
        )
        corner_turns = 90 * self.inventory.connectors.get(
            ConnectorType.RIGHT, 0
        ) + 45 * self.inventory.connectors.get(ConnectorType.DIAGONAL, 0)
        if legal_gates and corner_turns >= 360:
            packer = PolygonPacker(self.inventory, self.room, self.weights, deadline)
            run_stage(
                "room boundary",
                lambda end, keep: keep(
                    packer.pack(self.room.get_boundary_points(), deadline=end)
                ),
                min(1.0, 0.05 * budget),
            )
            run_stage(
                "templates",
                partial(search_templates, *context, packer=packer),
                min(12.0, 0.2 * budget),
            )
            run_stage(
                "room grid",
                partial(search_grid, *context),
                min(70.0, 0.45 * budget),
            )
            if self._starting_shapes:
                run_stage(
                    "refinement",
                    partial(
                        refine,
                        *context,
                        [candidate.construction for candidate in self._starting_shapes],
                    ),
                    min(40.0, 0.2 * budget),
                )
                run_stage(
                    "face paths",
                    # Face moves may have added better starting shapes.
                    partial(
                        refine_paths,
                        *context,
                        [candidate.construction for candidate in self._starting_shapes],
                    ),
                    50.0,
                )
            else:
                # Partial chains remain useful when the geometry searches
                # cannot construct a seed with an unusual inventory.
                from players.player5.beam_fallback import BeamFallback

                def fallback(end, keep):
                    player = BeamFallback(*context, deadline=end)
                    keep(player.build_enclosure())

                run_stage("beam fallback", fallback, deadline - time.process_time())

        self.stats.cpu = time.process_time() - started
        self.note = (
            f"score {self.best_score:g}; best from {self.stats.best_stage}; "
            f"{self.stats.valid_candidates} valid candidates; "
            f"{self.stats.cpu:.2f}s CPU"
        )
        return self._best_construction
