import time

from src.enclosure import score_construction, validate_construction

from .faces import FaceInventory
from .growth import grow_enclosure
from .prev_deliverable import Player3 as PreviousPlayer3


class Player3(PreviousPlayer3):
    HEXAGON_SECONDS = 5.0
    HEXAGON_GROWTH_SECONDS = 15.0
    SECONDARY_SEARCH_SECONDS = 40.0
    SECONDARY_EXPANSIONS_PER_START = 512

    def build_enclosure(self):
        begun = time.process_time()
        deadline = begun + self.SEARCH_SECONDS
        best, best_score, best_area = None, 0.0, 0.0
        best_stage = "none"
        grown_seeds = set()
        catalog = None

        def make_seed_player(until, expansions=None):
            player = PreviousPlayer3(self.room, self.inventory, self.weights)
            player.SEARCH_DEADLINE = until
            player.BEAM_WIDTH = self.BEAM_WIDTH
            player.MAX_FACES = self.MAX_FACES
            player.USE_BEAM = self.USE_BEAM
            if expansions is not None:
                player.EXPANSIONS_PER_START = expansions
            return player

        seed_player = make_seed_player(deadline)
        construction = seed_player.build_enclosure()
        self.progress = list(seed_player.progress)
        self.search_stats = {"seed": dict(seed_player.search_stats), "alternatives": []}

        def consider(candidate, stage):
            nonlocal best, best_score, best_area, best_stage
            if candidate is None:
                return None
            result = validate_construction(candidate, self.room, self.inventory)
            if not result.valid:
                return None
            score = score_construction(result, self.weights)
            if score > best_score + 1e-8:
                best, best_score, best_area = candidate, score, result.area
                best_stage = stage
                self.progress.append(
                    {"cpu": time.process_time() - begun, "score": score, "stage": stage}
                )
            return result

        def grow(candidate, stage, until):
            nonlocal catalog
            result = consider(candidate, stage + "_seed")
            if result is None or time.process_time() >= until:
                return {}
            signature = result.polygon.normalize().wkb
            if signature in grown_seeds:
                return {"skipped_duplicate_seed": True}
            grown_seeds.add(signature)
            if catalog is None:
                catalog = FaceInventory(self.inventory, deadline)
            if time.process_time() >= until or not catalog.lengths:
                return {}
            grown, stats = grow_enclosure(
                self.room, self.inventory, self.weights, catalog, candidate, until
            )
            consider(grown, stage)
            return stats

        self.search_stats.update(grow(construction, "mapped_face_growth", deadline))
        self.search_stats["baseline_score"] = best_score

        if time.process_time() < deadline:
            from .seeds import symmetric_hexagon_seeds

            started = time.process_time()
            hexagon_deadline = min(deadline, started + self.HEXAGON_SECONDS)
            if catalog is None:
                catalog = FaceInventory(self.inventory, hexagon_deadline)

                catalog.deadline = deadline
            candidate, stats = symmetric_hexagon_seeds(
                self.room, self.inventory, self.weights, catalog, hexagon_deadline
            )
            result = consider(candidate, "hexagon_seed")
            growth_stats = grow(
                candidate,
                "hexagon_growth",
                min(deadline, time.process_time() + self.HEXAGON_GROWTH_SECONDS),
            )
            self.search_stats["alternatives"].append(
                {
                    "name": "hexagon",
                    "cpu_seconds": time.process_time() - started,
                    "seed_score": score_construction(result, self.weights)
                    if result
                    else None,
                    "search": stats,
                    "growth": growth_stats,
                    "best_score": best_score,
                }
            )

        if self.USE_BEAM and time.process_time() + 1.0 < deadline:
            started = time.process_time()

            search_deadline = min(
                deadline,
                started + min(self.SECONDARY_SEARCH_SECONDS, (deadline - started) / 2),
            )
            alternative = make_seed_player(
                search_deadline, self.SECONDARY_EXPANSIONS_PER_START
            )
            candidate = alternative.build_enclosure()
            result = consider(candidate, "deeper_seed")
            growth_stats = grow(candidate, "deeper_growth", deadline)
            self.search_stats["alternatives"].append(
                {
                    "name": "deeper_beam",
                    "cpu_seconds": time.process_time() - started,
                    "seed_score": score_construction(result, self.weights)
                    if result
                    else None,
                    "search": dict(alternative.search_stats),
                    "growth": growth_stats,
                    "best_score": best_score,
                }
            )

        self.search_stats.update(
            cpu_seconds=time.process_time() - begun,
            best_score=best_score,
            final_area=best_area,
            best_stage=best_stage,
        )
        self.note = (
            f"Seed portfolio: best {best_score:.1f} from {best_stage}; "
            f"baseline {self.search_stats['baseline_score']:.1f}; area {best_area:.1f}"
        )
        return best
