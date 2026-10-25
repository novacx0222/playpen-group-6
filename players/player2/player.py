"""Search useful shapes and retain the best construction the simulator accepts."""

from players.player0 import Player
from src.constants import BASELINE_SCORE
from src.enclosure import Construction, score_construction, validate_construction

from .geometry import Budget, RoomGeometry, SearchStopped, Shape, record
from .inventory import Stock
from .refine import improve
from .search import boundaries, templates


class Player2(Player):
    SEARCH_CPU_SECONDS = 270.0
    USE_BOUNDARIES = True
    USE_DIAGONALS = True

    def build_enclosure(self) -> Construction | None:
        """Find fallbacks, follow the room, then refine the best placed shapes."""
        budget = Budget.start(self.SEARCH_CPU_SECONDS)
        self.best = None
        self.best_score = 0.0
        self.best_stage = "none"
        self.seeds = []
        self.candidates = 0
        self.incomplete_allocations = 0
        try:
            budget.check()
            self.geometry = RoomGeometry(self.room)
            self.stock = Stock(
                self.inventory, budget.slice(min(0.3, budget.remaining / 10))
            )
            if not self.stock.options or not self.stock.gates:
                self.note = "No usable gate or face found."
                return None
            discovery = budget.slice(min(10, budget.remaining * 0.75))
            fallback = discovery.slice(min(2, discovery.remaining / 3))
            record("templates", "start")
            try:
                for shape in templates(
                    self.stock, self.weights, self.room.polygon.area, fallback
                ):
                    self._evaluate(shape, None, fallback, stage="template")
            except SearchStopped:
                record("templates", "stop", reason="deadline")
            else:
                record("templates", "stop", reason="candidate_stream_exhausted")
            if self.USE_BOUNDARIES:
                record("boundaries", "start")
                for shape, pose in boundaries(
                    self.geometry, self.stock, self.weights, discovery
                ):
                    self._evaluate(shape, pose, discovery, stage="boundary")
                record(
                    "boundaries",
                    "stop",
                    reason="deadline" if not discovery.remaining else "passes_finished",
                )
            else:
                record("boundaries", "stop", reason="disabled")
        except SearchStopped:
            record("discovery", "stop", reason="deadline")
        if self.USE_DIAGONALS and self.best is not None and budget.remaining:
            # Keep the initial states even when improvements replace live seeds.
            seeds = tuple(self.seeds)
            for margin in (0.01, 0.05, 0.20):
                if not budget.remaining:
                    break
                improve(
                    seeds,
                    self.geometry,
                    self.stock,
                    budget,
                    lambda shape, pose, local, floor: self._evaluate(
                        shape, pose, local, floor, stage="diagonal"
                    ),
                    margin=margin,
                )
        else:
            record(
                "diagonal",
                "stop",
                reason="disabled" if not self.USE_DIAGONALS else "no_seed_or_time",
            )
        self.note = (
            f"{self.best_stage}; score {self.best_score:.2f}; "
            f"{self.candidates} candidates; "
            f"{self.incomplete_allocations} unfinished allocations"
        )
        return self.best

    def _evaluate(self, shape: Shape, pose, budget: Budget, floor=None, *, stage):
        """All searches share stock, placement, validation, and score checks."""
        budget.check()
        self.candidates += 1
        if not self.stock.supports(shape):
            return None
        floor = self.best_score if floor is None else floor
        base = (
            BASELINE_SCORE
            + self.weights.A * shape.polygon.area
            + self.weights.C * sum(shape.lengths)
        )
        bonus = max(0, self.weights.G) if self.stock.gates >= 2 else 0
        if base + bonus <= floor + 1e-7:
            return None
        for required in (2, 1) if bonus > 0 else (1,):
            try:
                allocation = self.stock.allocate(
                    shape.lengths, required, budget.slice(0.03)
                )
            except SearchStopped:
                self.incomplete_allocations += 1
                continue
            if allocation is None:
                continue
            score_bound = base + (self.weights.G if allocation.gate_faces >= 2 else 0)
            if score_bound <= floor + 1e-7:
                continue
            try:
                if pose is None:
                    pose = self.geometry.place(shape, budget.slice(0.05))
            except SearchStopped:
                return None
            if pose is None:
                return None
            construction = shape.construction(allocation.faces, pose)
            validation = validate_construction(construction, self.room, self.inventory)
            if not validation.valid:
                return None
            score = score_construction(validation, self.weights)
            if score > self.best_score:
                self.best, self.best_score, self.best_stage = construction, score, stage
                record(
                    stage,
                    "improvement",
                    score=score,
                    area=validation.area,
                    perimeter=validation.perimeter,
                )
                self.seeds.append((shape, pose, score))
                self.seeds.sort(key=lambda item: -item[2])
                del self.seeds[12:]
            return score
        return None
