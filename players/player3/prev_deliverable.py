"""Anytime, room-guided face beam search with a validated rectangle incumbent."""

import math
import time

from shapely.geometry import Polygon
from shapely.prepared import prep

from players.player0 import Player
from src.constants import MIN_WALL_LENGTH
from src.enclosure import score_construction, validate_construction
from src.pieces import ConnectorType

from .beam import face_beam
from .faces import FaceInventory
from .placement import rectangle_seeds


class Player3(Player):
    # One CPU deadline shared by seeding, beam search, and mapped growth.
    SEARCH_SECONDS = 250.0
    RECTANGLE_CANDIDATES = 2000
    EXPANSIONS_PER_START = 128
    BEAM_WIDTH = 24
    MAX_FACES = 14
    USE_BEAM = True

    def _beam_parameters(self):
        """Trade breadth for depth using scale-independent room compactness.

        Compactness approaches one for open, compact rooms and zero for
        thin or winding rooms. Keep some breadth even at the narrow end;
        this is still a beam search, with a larger face budget for detours.
        BEAM_WIDTH and MAX_FACES remain the configurable baseline budgets.
        """
        polygon = self.room.polygon
        compactness = max(0.0, min(1.0, 4 * math.pi * polygon.area / polygon.length**2))
        width = max(1, round(self.BEAM_WIDTH * (0.25 + 1.75 * compactness)))
        faces = max(4, round(self.MAX_FACES * (2.0 - compactness)))
        return compactness, width, faces

    def build_enclosure(self):
        begun = time.process_time()
        deadline = getattr(self, "SEARCH_DEADLINE", begun + self.SEARCH_SECONDS)
        self.search_stats = {"candidates": 0, "valid_candidates": 0, "improvements": 0}
        self.progress = []
        if not self.inventory.has_gate() or self.inventory.count_connectors() < 3:
            self.note = "No usable gate or insufficient connectors"
            return None
        if any(
            length < MIN_WALL_LENGTH
            for pool in (self.inventory.walls, self.inventory.gates)
            for length, count in pool.items()
            if count > 0
        ):
            self.note = "Inventory contains a piece shorter than the minimum"
            return None
        catalog = FaceInventory(self.inventory, deadline)
        prepared_room = prep(self.room.polygon.buffer(1e-7))
        best, best_score = None, 0.0
        seen = {}
        stage = "rectangle"

        def consider(vertices):
            nonlocal best, best_score
            if time.process_time() >= deadline:
                return None
            key = tuple((round(x, 6), round(y, 6)) for x, y in vertices)
            if key in seen:
                return seen[key]
            self.search_stats["candidates"] += 1
            polygon = Polygon(vertices)
            if not polygon.is_valid or polygon.area <= 1e-7:
                seen[key] = None
                return None
            if not prepared_room.covers(polygon):
                seen[key] = None
                return None
            construction = catalog.build(
                vertices, prefer_bonus=self.weights.G > 0, deadline=deadline
            )
            if construction is None:
                if time.process_time() < deadline:
                    seen[key] = None
                return None
            result = validate_construction(construction, self.room, self.inventory)
            if not result.valid:
                seen[key] = None
                return None
            score = score_construction(result, self.weights)
            seen[key] = score
            self.search_stats["valid_candidates"] += 1
            if score > best_score + 1e-8:
                best, best_score = construction, score
                self.search_stats["improvements"] += 1
                self.progress.append(
                    {"cpu": time.process_time() - begun, "score": score, "stage": stage}
                )
            return score

        self.search_stats.update(
            rectangle_seeds(
                self.room,
                catalog,
                self.weights,
                deadline,
                consider,
                max_candidates=self.RECTANGLE_CANDIDATES,
            )
        )
        if self.USE_BEAM:
            stage = "beam"
            compactness, beam_width, max_faces = self._beam_parameters()
            max_faces = min(
                max_faces,
                sum(
                    self.inventory.connectors.get(kind, 0)
                    for kind in (ConnectorType.RIGHT, ConnectorType.DIAGONAL)
                ),
            )
            self.search_stats.update(
                room_compactness=compactness,
                beam_width=beam_width,
                max_faces=max_faces,
            )
            # Narrowing the first beam can permanently discard useful paths.
            # Preserve the original search first, then spend unused time on
            # the room-guided alternative, sharing the validated incumbent.
            passes = [("baseline", self.BEAM_WIDTH, self.MAX_FACES)]
            if (beam_width, max_faces) != (self.BEAM_WIDTH, self.MAX_FACES):
                passes.append(("adaptive", beam_width, max_faces))
            self.search_stats["beam_passes"] = []
            for name, width, faces in passes:
                if time.process_time() >= deadline:
                    break
                stage = f"beam_{name}"
                pass_stats = face_beam(
                    self.room,
                    self.inventory,
                    catalog,
                    self.weights,
                    deadline,
                    consider,
                    beam_width=width,
                    max_faces=faces,
                    expansions_per_start=self.EXPANSIONS_PER_START,
                )
                self.search_stats["beam_passes"].append(
                    {
                        "name": name,
                        "beam_width": width,
                        "max_faces": faces,
                        "best_score": best_score,
                        **pass_stats,
                    }
                )
                for key, value in pass_stats.items():
                    self.search_stats[key] = self.search_stats.get(key, 0) + value
        if best is None and time.process_time() < deadline:
            from .legacy import Player3 as LegacyPlayer

            fallback = LegacyPlayer(self.room, self.inventory, self.weights)
            fallback.SEARCH_DEADLINE = deadline
            construction = fallback.build_enclosure()
            if construction is not None:
                result = validate_construction(construction, self.room, self.inventory)
                score = score_construction(result, self.weights) if result.valid else 0
                if score > best_score:
                    best, best_score = construction, score
                    self.progress.append(
                        {
                            "cpu": time.process_time() - begun,
                            "score": score,
                            "stage": "fallback",
                        }
                    )
        self.search_stats["cpu_seconds"] = time.process_time() - begun
        self.search_stats["best_score"] = best_score
        self.note = (
            f"Face beam: {self.search_stats.get('states', 0)} states, "
            f"{self.search_stats.get('starts', 0)} starts, "
            f"{self.search_stats['valid_candidates']} valid candidates; best {best_score:.1f}"
        )
        return best
