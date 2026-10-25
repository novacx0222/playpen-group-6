"""Improve placed shapes with paired diagonal faces that keep exact closure."""

import math
from collections import Counter, deque
from itertools import combinations

from .geometry import Budget, SearchStopped, Shape, offset, record


def moved(start, angle, direction, distance):
    x, y = offset(direction, distance)
    co, si = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return start[0] + co * x - si * y, start[1] + si * x + co * y


def clean_edges(edges, start, angle):
    """Remove vanished faces and merge straight runs without changing their pose."""
    merged = []
    for direction, length in edges:
        if length == 0:
            continue
        if merged and direction == merged[-1][0]:
            previous, size = merged.pop()
            merged.append((previous, size + length))
        else:
            merged.append((direction, length))
    if len(merged) > 1 and merged[0][0] == merged[-1][0]:
        direction, length = merged.pop()
        start = moved(start, angle, direction, -length)
        merged[0] = direction, merged[0][1] + length
    try:
        shape = Shape(tuple(d for d, _ in merged), tuple(n for _, n in merged))
    except ValueError:
        return None
    return shape, (start, angle)


def diagonal_edits(shape, pose, stock, budget: Budget):
    """Insert equal diagonals at two corners with opposite bisectors.

    Equal diagonal displacements cancel. The same integer inset on opposite
    neighboring faces also cancels, even when the inset differs from the
    geometric chamfer. Placement and stock must still be checked afterward.
    """
    directions, lengths = shape.directions, shape.lengths
    corners = [i for i, t in enumerate(shape.turns) if abs(t) == 2]
    bisectors = {i: (directions[i - 1] + shape.turns[i] // 2) % 8 for i in corners}
    for first, second in combinations(corners, 2):
        budget.check()
        if (bisectors[first] - bisectors[second]) % 8 != 4:
            continue
        uses = Counter(
            ((first - 1) % len(lengths), first, (second - 1) % len(lengths), second)
        )
        choices = []
        for i, times in uses.items():
            choices.append(
                {
                    (lengths[i] - target) // times
                    for target in (0, *stock.options)
                    if (lengths[i] - target) % times == 0
                }
            )
        insets = set.intersection(*choices)
        for diagonal in sorted(stock.options, reverse=True):
            for inset in sorted(
                insets, key=lambda n: (abs(n - diagonal / math.sqrt(2)), n)
            ):
                budget.check()
                changed = [n - inset * uses[i] for i, n in enumerate(lengths)]
                if any(n < 0 or (n and n not in stock.options) for n in changed):
                    continue
                edges = []
                for i, (direction, length) in enumerate(zip(directions, changed)):
                    if i in (first, second):
                        edges.append((bisectors[i], diagonal))
                    edges.append((direction, length))
                start, angle = pose
                if first == 0:
                    start = moved(start, angle, directions[-1], -inset)
                result = clean_edges(edges, start, angle)
                if result is None:
                    continue
                child, child_pose = result
                if not stock.supports(child) or len(child.lengths) > 512:
                    continue
                # Share the small displacement between the two modified arcs.
                remainder = diagonal - inset * math.sqrt(2)
                for fraction in (0, -0.5, -1):
                    p, angle = child_pose
                    yield (
                        child,
                        (
                            moved(p, angle, bisectors[first], fraction * remainder),
                            angle,
                        ),
                    )


def improve(seeds, room, stock, budget: Budget, evaluate, *, margin=0.01):
    """Resume each parent's edits in short batches; keep a small set of states."""
    record("diagonal", "start", margin=margin)
    if not seeds or stock.diagonals < 4:
        record("diagonal", "stop", reason="no_seeds_or_insufficient_connectors")
        return
    queue = deque()
    seen = set()
    best_score = max(score for _, _, score in seeds)

    def add(shape, pose, score):
        key = shape, tuple(round(x, 7) for x in pose[0]), round(pose[1], 7)
        if key in seen:
            return
        seen.add(key)
        queue.append((score, diagonal_edits(shape, pose, stock, budget)))

    for shape, pose, score in sorted(seeds, key=lambda item: -item[2])[:12]:
        add(shape, pose, score)
    batches = 0
    proposals = 0
    pruned = 0
    try:
        for _ in range(20000):
            batches += 1
            budget.check()
            if not queue:
                record(
                    "diagonal",
                    "stop",
                    reason="queue_exhausted",
                    batches=batches,
                    proposals=proposals,
                    queue_pruned=pruned,
                )
                return
            score, stream = queue.popleft()
            for _ in range(64):
                budget.check()
                candidate = next(stream, None)
                if candidate is None:
                    break
                proposals += 1
                shape, pose = candidate
                if not room.allowed.covers(shape.placed(pose)):
                    continue
                floor = max(0, best_score - max(10, best_score * margin))
                result = evaluate(shape, pose, budget, floor)
                if result is not None:
                    best_score = max(best_score, result)
                    add(shape, pose, result)
            else:
                queue.append((score, stream))
            if len(queue) > 24:
                pruned += len(queue) - 24
                queue = deque(sorted(queue, key=lambda item: -item[0])[:24])
    except SearchStopped:
        record(
            "diagonal",
            "stop",
            reason="deadline",
            batches=batches,
            proposals=proposals,
            queue_pruned=pruned,
        )
        return
    record(
        "diagonal",
        "stop",
        reason="batch_limit",
        batches=batches,
        proposals=proposals,
        queue_pruned=pruned,
    )
