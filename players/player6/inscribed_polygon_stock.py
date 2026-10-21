#!/usr/bin/env python3
"""
Stock-constrained extension of inscribed_polygon.py.

stock = {piece_length: count}, e.g. {5: 40, 10: 20, 15: 12, 30: 4}
A straight run (5..30 total) may be built from several collinear pieces,
each piece >= 5. Needs only shapely and numpy.

Strategy
  1. Geometry-first: run the normal solver, ask "can this boundary be built
     from stock?" (bounded assignment search).
  2. Overconstrained: wall is the bottleneck, so also try
       a) the largest buildable rectangle (best when the budget is tiny)
       b) shrinking the input inward by d units (mitre buffer) and re-solving,
          bisecting for the smallest d whose boundary is buildable.
  3. Publish buildable candidates immediately; a caller may validate and score
     each one. Return the best-scoring candidate plus its piece plan.
"""

from collections import Counter

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Polygon, box

import players.player6.inscribed_polygon as ip


# --------------------------------------------------------- piece assignment
def clean_stock(stock):
    return {
        int(length): int(c)
        for length, c in stock.items()
        if 5 <= length <= 30 and c > 0
    }


def compositions(total, lengths, stop=None):
    """All multisets of `lengths` (non-increasing tuples) summing to total."""
    lengths = sorted(lengths, reverse=True)
    out = []

    def rec(rem, maxpart, cur):
        ip.check_stop(stop)
        if rem == 0:
            out.append(tuple(cur))
            return
        for length in lengths:
            if length <= rem and length <= maxpart:
                cur.append(length)
                rec(rem - length, length, cur)
                cur.pop()

    rec(total, max(lengths), [])
    return out


def run_lengths(poly):
    """Maximal straight runs (collinear edges merged cyclically), rounded."""
    return np.rint(ip.merged_edge_lengths(poly)).astype(int)


def _build(runs, stock, cls=(), prefer_few=False, stop=None):
    """Precompute compositions etc. `cls` = class piece lengths (may be empty).
    Returns ctx or None if some run length can't be built at all."""
    cls = set(cls)
    lens = sorted(stock)
    need = sorted(Counter(int(r) for r in runs).items())  # short runs first
    comps = []
    for L, _ in need:
        cs = compositions(L, lens, stop=stop)
        if not cs:
            return None
        # class compositions first (when a class is active), then plentiful
        # pieces, then fewer pieces
        cs.sort(
            key=lambda c: (
                not any(p in cls for p in c),
                len(c) if prefer_few else 0,
                sum(1.0 / stock[p] for p in c),
                len(c),
                c,
            )
        )
        comps.append(
            [
                (c, tuple(c.count(length) for length in lens), any(p in cls for p in c))
                for c in cs
            ]
        )
    suffix_len = [0] * (len(need) + 1)
    suffix_runs = [0] * (len(need) + 1)
    suffix_cls = [0] * (len(need) + 1)  # runs that COULD hold a class piece
    minp = [min(len(c) for c, _, _ in cs) for cs in comps]  # fewest pieces per run
    suffix_minp = [0] * (len(need) + 1)
    for i in range(len(need) - 1, -1, -1):
        suffix_len[i] = suffix_len[i + 1] + need[i][0] * need[i][1]
        suffix_runs[i] = suffix_runs[i + 1] + need[i][1]
        has = any(h for _, _, h in comps[i])
        suffix_cls[i] = suffix_cls[i + 1] + (need[i][1] if has else 0)
        suffix_minp[i] = suffix_minp[i + 1] + need[i][1] * minp[i]
    return {
        "lens": lens,
        "need": need,
        "comps": comps,
        "suffix_len": suffix_len,
        "suffix_runs": suffix_runs,
        "suffix_cls": suffix_cls,
        "stock": stock,
        "minp": minp,
        "suffix_minp": suffix_minp,
        "cls_idx": [j for j, length in enumerate(lens) if length in cls],
    }


def _dfs(ctx, k, node_limit, piece_cap=None, stop=None):
    """
    Exact search: split every run into pieces from stock so that at least k
    RUNS contain a class piece (k = 0: no class requirement; two class pieces
    in the same run count once). Returns {run_length: [(comp, count)]} or None.
    """
    lens, need, comps, stock = ctx["lens"], ctx["need"], ctx["comps"], ctx["stock"]
    cls_idx, suffix_cls = ctx["cls_idx"], ctx["suffix_cls"]
    start = tuple(stock[length] for length in lens)
    total0 = sum(start)
    minp, suffix_minp = ctx["minp"], ctx["suffix_minp"]
    piece_lengths = ctx.get("piece_lengths", lens)
    if k and suffix_cls[0] < k:
        return None
    if piece_cap is not None and suffix_minp[0] > piece_cap:
        return None
    failed, chosen, nodes = set(), [], 0

    def rec(ti, ki, left, stk, cr):
        nonlocal nodes
        ip.check_stop(stop)
        if left == 0:  # type done
            if piece_cap is not None and total0 - sum(stk) > piece_cap:
                return False
            if ti + 1 == len(need):
                return cr >= k
            return rec(ti + 1, 0, need[ti + 1][1], stk, cr)
        nodes += 1
        if nodes > node_limit or ki == len(comps[ti]):
            return False
        done = need[ti][1] - left
        wall_left = ctx["suffix_len"][ti] - done * need[ti][0]
        runs_left = ctx["suffix_runs"][ti] - done
        if wall_left > sum(
            length * c for length, c in zip(piece_lengths, stk)
        ) or runs_left > sum(stk):
            return False
        # Pieces so far + fewest still needed.
        if (
            piece_cap is not None
            and total0 - sum(stk) + left * minp[ti] + suffix_minp[ti + 1] > piece_cap
        ):
            return False
        has_type = any(h for _, _, h in comps[ti])
        if k and cr < k:  # can enough class runs still happen?
            bound = min(
                (left if has_type else 0) + suffix_cls[ti + 1],
                sum(stk[j] for j in cls_idx),
            )
            if cr + bound < k:
                return False
        key = (ti, ki, left, stk, cr)
        if key in failed:
            return False
        comp, mult, has = comps[ti][ki]
        cmax = left
        for m, have in zip(mult, stk):
            if m:
                cmax = min(cmax, have // m)
        order = list(range(cmax, -1, -1))
        if has and k:  # try "just enough" class runs first
            first = min(cmax, max(k - cr, 0))
            order = [first] + [c for c in order if c != first]
        for c in order:
            new = tuple(h - c * m for h, m in zip(stk, mult))
            ncr = min(cr + (c if has else 0), k)
            if c:
                chosen.append((need[ti][0], comp, c))
            if rec(ti, ki + 1, left - c, new, ncr):
                return True
            if c:
                chosen.pop()
            if nodes > node_limit:
                return False
        failed.add(key)
        return False

    import sys

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))
    if not rec(0, 0, need[0][1], start, 0):
        return None
    plan = {}
    for L, comp, c in chosen:
        plan.setdefault(L, []).append((comp, c))
    return plan


def assign_inventory(
    runs,
    walls,
    gates,
    angle_stock=None,
    *,
    stop=None,
    node_limit=200_000,
    min_gate_faces=1,
    prefer_two=True,
):
    """Assign typed (0=gate, 1=wall) pieces, preferring two actual gate faces.

    A one-gate feasible plan is retained if the optional bonus search times out.
    Counts of equal-length walls and gates remain independent throughout DFS.
    min_gate_faces=0 allows spare, ungated faces for an already gated seed;
    prefer_two=False disables the optional second-gate assignment search.
    """
    stock = {(1, length): n for length, n in clean_stock(walls).items()}
    stock.update({(0, length): n for length, n in clean_stock(gates).items()})
    if (min_gate_faces and not any(kind == 0 for kind, _ in stock)) or not len(runs):
        return None
    angles = clean_angle_stock(angle_stock)
    if len(runs) > angles.get("right", len(runs)):
        return None
    cap = len(runs) + angles["straight"] if "straight" in angles else None
    keys = sorted(stock)
    need = sorted(Counter(int(r) for r in runs).items())
    comps = []
    max_pieces = min(6, angles.get("straight", 5) + 1)
    for length, _ in need:
        options = []

        def rec(i, remaining, chosen, options):
            ip.check_stop(stop)
            if remaining == 0:
                pieces = tuple(chosen)
                options.append(
                    (
                        pieces,
                        tuple(pieces.count(key) for key in keys),
                        any(kind == 0 for kind, _ in pieces),
                    )
                )
                return
            if i == len(keys) or len(chosen) >= max_pieces:
                return
            key = keys[i]
            most = min(stock[key], remaining // key[1], max_pieces - len(chosen))
            for count in range(most + 1):
                rec(i + 1, remaining - count * key[1], chosen + [key] * count, options)

        rec(0, length, [], options)
        if not options:
            return None
        options.sort(key=lambda c: (not c[2], sum(p[0] == 0 for p in c[0]), len(c[0])))
        comps.append(options)
    suffix_len = [0] * (len(need) + 1)
    suffix_runs = [0] * (len(need) + 1)
    suffix_cls = [0] * (len(need) + 1)
    suffix_minp = [0] * (len(need) + 1)
    minp = [min(len(pieces) for pieces, _, _ in options) for options in comps]
    for i in range(len(need) - 1, -1, -1):
        length, count = need[i]
        suffix_len[i] = suffix_len[i + 1] + length * count
        suffix_runs[i] = suffix_runs[i + 1] + count
        suffix_cls[i] = suffix_cls[i + 1] + count * any(c[2] for c in comps[i])
        suffix_minp[i] = suffix_minp[i + 1] + count * minp[i]
    ctx = {
        "lens": keys,
        "piece_lengths": [key[1] for key in keys],
        "need": need,
        "comps": comps,
        "stock": stock,
        "minp": minp,
        "suffix_len": suffix_len,
        "suffix_runs": suffix_runs,
        "suffix_cls": suffix_cls,
        "suffix_minp": suffix_minp,
        "cls_idx": [i for i, key in enumerate(keys) if key[0] == 0],
    }
    plan = _dfs(ctx, min_gate_faces, node_limit, cap, stop=stop)
    if prefer_two and plan is not None and sum(clean_stock(gates).values()) >= 2:
        try:
            bonus_plan = _dfs(ctx, 2, node_limit, cap, stop=stop)
        except ip.SearchStopped:
            return plan
        if bonus_plan is not None:
            return bonus_plan
    return plan


def plan_usage(plan):
    used = Counter()
    for items in plan.values():
        for comp, c in items:
            for p in comp:
                used[p] += c
    return used


def class_runs(plan, must_use):
    """Number of RUNS containing at least one class piece (two class pieces in
    the same run count once)."""
    cls = set(must_use or [])
    return sum(c for items in plan.values() for comp, c in items if cls & set(comp))


def bonus_earned(plan, must_use):
    """One-time bonus: at least 2 separate runs contain a class piece."""
    return class_runs(plan, must_use) >= 2


# ------------------------------------------------- must-use class: feasibility
def class_usable(stock, must_use, copy_cap=64, stop=None):
    """
    Geometry-independent NECESSARY test for using a class piece.
    A closed rectilinear polygon has rightward wall == leftward wall (and
    up == down), so the pieces used must split into two equal-sum sides.
    Bitset DP over the side difference; `used_class` is tracked as a flag.
    (Copies beyond copy_cap can't matter: equal pairs cancel across sides.)
    """
    stock = clean_stock(stock)
    cls = {p for p in (must_use or []) if p in stock}
    if not cls:
        return False
    S = sum(length * min(c, copy_cap) for length, c in stock.items())
    off, full = S, (1 << (2 * S + 1)) - 1
    no_cls, with_cls = 1 << off, 0
    for length, c in stock.items():
        for _ in range(min(c, copy_cap)):
            ip.check_stop(stop)

            def move(x, length=length):
                return (
                    x | (x << length) | (x >> length)
                ) & full  # skip / side A / side B

            if length in cls:
                with_cls = (
                    move(with_cls) | ((no_cls << length) | (no_cls >> length)) & full
                )
            else:
                no_cls, with_cls = move(no_cls), move(with_cls)
    return bool((with_cls >> off) & 1)


def class_fits_in_run(stock, must_use, cell, stop=None):
    """Can a class piece sit inside a run whose total is a multiple of `cell`
    (so the shape needs NO wall shifting and loses no area)?"""
    stock = clean_stock(stock)
    cls = {p for p in (must_use or []) if p in stock}
    if not cls:
        return False
    lens = sorted(stock)
    for L in range(cell, 31, cell):
        for comp in compositions(L, lens, stop=stop):
            cnt = Counter(comp)
            if any(p in cnt for p in cls) and all(cnt[q] <= stock[q] for q in cnt):
                return True
    return False


def assign_pieces(
    runs, stock, must_use=None, node_limit=200_000, piece_cap=None, stop=None
):
    """
    Can every run be split into collinear pieces from stock?

    must_use: ONE class of piece lengths, e.g. [9, 13].
      HARD: at least one RUN must contain a class piece.
      BONUS (one-time): at least two separate runs contain one. Pieces in the
            same run count once, so 9+21 inside one 30 is a single instance.
    Tries the bonus level first, then the hard level. Returns
    {run_length: [(composition, count), ...]} or None. Never returns a plan that
    breaks stock or class rules; the node limit can only cause a false "no".
    """
    stock = clean_stock(stock)
    if not stock:
        return None
    runs = [int(r) for r in runs]
    if sum(runs) > sum(length * c for length, c in stock.items()):
        return None
    if len(runs) > sum(stock.values()):
        return None
    cls = sorted({p for p in must_use if p in stock}) if must_use else []
    if must_use and not cls:
        return None
    if piece_cap is not None and piece_cap < len(runs):
        return None  # every run needs >= 1 piece
    ctx = _build(runs, stock, cls, prefer_few=piece_cap is not None, stop=stop)
    if ctx is None:
        return None
    if not cls:
        return _dfs(ctx, 0, node_limit, piece_cap, stop=stop)
    return _dfs(ctx, 2, node_limit, piece_cap, stop=stop) or _dfs(
        ctx, 1, node_limit, piece_cap, stop=stop
    )


# ------------------------------------------------------------ angle inventory
def clean_angle_stock(angle_stock):
    """{"straight": n, "right": n, "diagonal": n}; a missing key = unlimited."""
    return {
        getattr(k, "value", k): int(v)
        for k, v in (angle_stock or {}).items()
        if v is not None
    }


def buildable(runs, stock, must_use=None, angle_stock=None, **kw):
    """
    assign_pieces + the angle inventory.
      right    : every vertex of a rectilinear polygon is a 90 or 270 degree
                 corner, and #corners == #maximal runs.
      straight : a run built from k pieces has k-1 straight (180) joints, so
                 total pieces <= #runs + straight inventory.
      diagonal : unused (no 45-degree edges are generated).
    """
    a = clean_angle_stock(angle_stock)
    R = len(runs)
    if "right" in a and R > a["right"]:
        return None
    cap = R + a["straight"] if "straight" in a else None
    return assign_pieces(runs, stock, must_use, piece_cap=cap, **kw)


def angle_usage(plan):
    """Angle pieces a plan consumes."""
    runs = sum(c for items in plan.values() for _, c in items)
    pieces = sum(c * len(comp) for items in plan.values() for comp, c in items)
    return {"right": runs, "straight": pieces - runs, "diagonal": 0}


def plan_is_valid(runs, plan, stock, must_use=None):
    """Independent check of a plan (used in tests)."""
    stock = clean_stock(stock)
    used = Counter()
    for L, items in plan.items():
        if sum(c for _, c in items) != sum(1 for r in runs if int(r) == L):
            return False
        for comp, c in items:
            if sum(comp) != L or min(comp) < 5:
                return False
            for p in comp:
                used[p] += c
    if not all(used[p] <= stock.get(p, 0) for p in used):
        return False
    return (not must_use) or class_runs(plan, must_use) >= 1


# ---------------------------------------------------------------- candidates
def best_rectangle(
    poly,
    stock,
    must_use=None,
    angle_stock=None,
    angles=None,
    n_off=3,
    *,
    stop=None,
    plan_for=None,
    on_candidate=None,
):
    """Largest axis-aligned (in some candidate frame) rectangle, sides
    5..30 in steps of 5, whose 4 walls are buildable from stock."""
    sizes = []
    if plan_for is None:

        def plan_for(runs):
            return buildable(runs, stock, must_use, angle_stock, stop=stop)

    for wc in range(1, ip.MAX_RUN + 1):
        for hc in range(1, ip.MAX_RUN + 1):
            ip.check_stop(stop)
            if plan_for([wc * 5, wc * 5, hc * 5, hc * 5]):
                sizes.append((wc * hc, wc, hc))
    if not sizes:
        return None
    sizes.sort(reverse=True)

    if angles is None:
        angles = ip.dominant_angles(poly)
    c0 = poly.centroid
    offs = np.linspace(0, ip.CELL, n_off, endpoint=False)
    frames = []
    for ang in angles:
        ip.check_stop(stop)
        rp = affinity.rotate(poly, -ang, origin=c0) if ang else Polygon(poly.exterior)
        shapely.prepare(rp)
        minx, miny, maxx, maxy = rp.bounds
        for ox in offs:
            for oy in offs:
                ip.check_stop(stop)
                x0, y0 = minx + ox, miny + oy
                W, H = int((maxx - x0) // ip.CELL), int((maxy - y0) // ip.CELL)
                if W < 1 or H < 1:
                    continue
                g = ip.available_mask(rp, x0, y0, W, H, stop=stop)
                S = np.zeros((H + 1, W + 1), dtype=np.int32)
                S[1:, 1:] = g.cumsum(0).cumsum(1)
                frames.append((ang, x0, y0, W, H, S))

    best = None
    for _, wc, hc in sizes:  # biggest area first
        for ang, x0, y0, W, H, S in frames:
            ip.check_stop(stop)
            if wc > W or hc > H:
                continue
            win = (
                S[hc:, wc:]
                - S[: H + 1 - hc, wc:]
                - S[hc:, : W + 1 - wc]
                + S[: H + 1 - hc, : W + 1 - wc]
            )
            hits = np.argwhere(win == wc * hc)
            if len(hits):
                r, c = hits[0]
                rect = box(x0 + c * 5, y0 + r * 5, x0 + (c + wc) * 5, y0 + (r + hc) * 5)
                if ang:
                    rect = affinity.rotate(rect, ang, origin=c0)
                if ip.validate(rect, poly):
                    if on_candidate is None:
                        return rect
                    on_candidate(rect)
                    if best is None or rect.area > best.area:
                        best = rect
    return best


def _shrunk(poly, d):
    p = poly.buffer(-d, join_style="mitre")
    if p.is_empty:
        return None
    if p.geom_type == "MultiPolygon":
        p = max(p.geoms, key=lambda g: g.area)
    return Polygon(p.exterior)


def shrink_search(
    poly,
    stock,
    must_use=None,
    angle_stock=None,
    *,
    stop=None,
    plan_for=None,
    on_candidate=None,
    **kw,
):
    """Smallest inward offset d (integer units) whose re-solved boundary is
    buildable. Probes use cheap solver settings; the winning d is re-solved at
    full quality. Not strictly monotone in d, so this is a heuristic bisection."""
    fast = dict(kw)
    fast.setdefault("n_off", 1)
    fast.setdefault("top_k", 2)
    if plan_for is None:

        def plan_for(runs):
            return buildable(runs, stock, must_use, angle_stock, stop=stop)

    def attempt(d, kwargs):
        ip.check_stop(stop)
        sh = _shrunk(poly, d)
        if sh is None:
            return "empty", None, None
        r = ip.solve(sh, stop=stop, on_candidate=on_candidate, **kwargs)
        if r is None:
            return "none", None, None
        pl = plan_for(run_lengths(r))
        return ("ok" if pl else "no"), r, pl

    lo, d, hit = 0, 1, None
    while True:  # exponential probe
        status, r, pl = attempt(d, fast)
        if status == "empty":
            return None
        if status == "ok":
            hit = (d, r, pl)
            break
        lo, d = d, d * 2
    hi_d = hit[0]
    while hi_d - lo > 1:  # bisect
        mid = (lo + hi_d) // 2
        status, r, pl = attempt(mid, fast)
        if status == "ok":
            hi_d, hit = mid, (mid, r, pl)
        else:
            lo = mid
    status, r, pl = attempt(hit[0], kw)  # full-quality re-solve at the winner
    if status == "ok" and r.area > hit[1].area:
        hit = (hit[0], r, pl)
    return hit[1], hit[2]


def _search(
    poly,
    stock,
    cells,
    must_use,
    bonus_weight,
    angle_stock=None,
    *,
    gate_stock=None,
    stop=None,
    evaluate=None,
    **kw,
):
    stock = clean_stock(stock)
    cands = []
    plans = {}
    seen = set()

    def plan_for(runs):
        key = tuple(sorted(int(r) for r in runs))
        if key not in plans:
            if gate_stock is not None:
                plans[key] = assign_inventory(
                    key, stock, gate_stock, angle_stock, stop=stop
                )
            else:
                plans[key] = buildable(key, stock, must_use, angle_stock, stop=stop)
        return plans[key]

    def remember(r, how):
        ip.check_stop(stop)
        key = r.wkb
        if key in seen:
            return
        pl = plan_for(run_lengths(r))
        if pl is None:
            return
        # Publish immediately, including a feasible assignment retained when
        # the optional two-gate search reaches its deadline.
        score = evaluate(r, pl) if evaluate is not None else None
        if evaluate is not None and score is None:
            return
        if evaluate is None:
            bonus = (
                sum(
                    c
                    for items in pl.values()
                    for pieces, c in items
                    if any(kind == 0 for kind, _ in pieces)
                )
                >= 2
                if gate_stock is not None
                else bonus_earned(pl, must_use)
            )
            score = r.area + bonus_weight * bonus
        seen.add(key)
        cands.append((score, r, pl, how))

    old_cell, old_run = ip.CELL, ip.MAX_RUN
    try:
        for cell in cells:
            ip.check_stop(stop)
            ip.CELL, ip.MAX_RUN = cell, 30 // cell

            def publish(r, cell=cell):
                remember(r, f"grid(cell={cell})")

            r = ip.solve(poly, stop=stop, on_candidate=publish, **kw)
            if r is not None:
                pl = plan_for(run_lengths(r))
                if pl:
                    remember(r, f"unconstrained(cell={cell})")
                    continue
            sres = shrink_search(
                poly,
                stock,
                must_use,
                angle_stock,
                stop=stop,
                plan_for=plan_for,
                on_candidate=publish,
                **kw,
            )
            if sres is not None:
                remember(sres[0], f"shrunk(cell={cell})")
        ip.CELL, ip.MAX_RUN = old_cell, old_run
        best_rectangle(
            poly,
            stock,
            must_use,
            angle_stock,
            stop=stop,
            plan_for=plan_for,
            on_candidate=lambda r: remember(r, "rectangle"),
        )
    except ip.SearchStopped:
        pass
    finally:
        ip.CELL, ip.MAX_RUN = old_cell, old_run

    if not cands:
        return None, None, "infeasible"
    _, r, pl, how = max(cands, key=lambda c: c[0])
    return r, pl, how


def solve_with_stock(
    poly,
    stock,
    cells=(5, 10, 15),
    must_use=None,
    bonus_weight=0.0,
    on_class_impossible="fallback",
    angle_stock=None,
    gate_stock=None,
    stop=None,
    evaluate=None,
    **kw,
):
    """
    Returns (polygon, piece_plan, how) or (None, None, 'infeasible').

    cells        : grid sizes tried (bigger cells => fewer short walls).
    must_use     : ONE class of piece lengths, e.g. [9, 13].
                   HARD : at least one RUN contains a class piece.
                   BONUS: a flat, one-time reward when at least two separate
                          runs contain one (pieces in the same run count once,
                          and a third instance adds nothing).
    bonus_weight : area units the one-time bonus is worth when ranking
                   candidate shapes (0 = bonus only breaks ties).
    on_class_impossible : "fallback" -> return the best shape WITHOUT the class
                   and say why in `how`; "fail" -> return infeasible.
    angle_stock  : angle-piece inventory, e.g. {"straight": 20, "right": 20,
                   "diagonal": 20}; a missing key means unlimited.
                   right    : one per corner (90 or 270 degrees) = one per run.
                   straight : one per 180-degree joint between collinear pieces.
                   diagonal : not used yet (shapes are rectilinear).

    gate_stock   : when supplied, stock means wall counts and gate_stock holds
                   separate gate counts. Plans then contain typed (kind, length)
                   pieces, with kind 0 for gates and 1 for walls.
    stop         : cooperative CPU deadline callback, checked inside searches.
    evaluate     : optional (polygon, plan) -> real score, or None to reject.
                   Called immediately for each buildable candidate, before any
                   area-only ranking. A timeout preserves the best published one.

    Legacy length-class handling, cheapest test first:
      1. class_usable      : can the pieces balance (right==left, up==down)?
                             If not, no polygon can use a class piece at all.
      2. class_fits_in_run : can a class piece sit inside a run that is a
                             multiple of the cell size? Then NO area is lost.
                             If not, walls would have to be shifted by whole
                             units (small area loss; not implemented).
    """
    # Typed callers pass wall counts as stock and separate gate counts. Their
    # callback validates/scores every buildable outline before area filtering.
    if gate_stock is not None:
        return _search(
            poly,
            stock,
            cells,
            None,
            bonus_weight,
            angle_stock,
            gate_stock=gate_stock,
            stop=stop,
            evaluate=evaluate,
            **kw,
        )
    if stop is not None and stop():
        return None, None, "time budget reached"
    stock = clean_stock(stock)
    ang = clean_angle_stock(angle_stock)
    if "right" in ang and ang["right"] < 4:
        return None, None, "infeasible (a rectilinear polygon needs >= 4 right angles)"
    if must_use:
        why = None
        try:
            if not class_usable(stock, must_use, stop=stop):
                why = "balance-impossible"
            elif not any(
                class_fits_in_run(stock, must_use, c, stop=stop) for c in cells
            ):
                why = "needs-wall-shifting"
        except ip.SearchStopped:
            return None, None, "time budget reached"
        if why is None:
            r = _search(
                poly,
                stock,
                cells,
                must_use,
                bonus_weight,
                angle_stock,
                stop=stop,
                evaluate=evaluate,
                **kw,
            )
            if r[0] is not None:
                return r
            why = "class-unbuildable-with-this-stock"
        if on_class_impossible == "fail":
            return None, None, f"infeasible ({why})"
        r = _search(
            poly,
            stock,
            cells,
            None,
            0.0,
            angle_stock,
            stop=stop,
            evaluate=evaluate,
            **kw,
        )
        if r[0] is None:
            return r
        return r[0], r[1], f"{r[2]} [class unused: {why}]"
    return _search(
        poly,
        stock,
        cells,
        None,
        bonus_weight,
        angle_stock,
        stop=stop,
        evaluate=evaluate,
        **kw,
    )


def diagnose_class(poly, stock, must_use, cells=(5, 10, 15), **kw):
    """Print which stage rejects (or accepts) the must-use class, and why."""
    cleaned = clean_stock(stock)
    dropped = sorted(
        length for length, c in stock.items() if int(length) not in cleaned
    )
    print(f"stock lengths ignored (outside 5..30 or count 0): {dropped or 'none'}")
    missing = [p for p in must_use if p not in cleaned]
    print(f"class lengths missing from usable stock: {missing or 'none'}")
    print(f"balance test (class_usable): {class_usable(cleaned, must_use)}")
    old_cell, old_run = ip.CELL, ip.MAX_RUN
    try:
        for c in cells:
            ip.CELL, ip.MAX_RUN = c, 30 // c
            hosts = sorted(
                L
                for L in range(c, 31, c)
                if any(
                    any(q in must_use for q in comp)
                    and all(Counter(comp)[q] <= cleaned.get(q, 0) for q in comp)
                    for comp in compositions(L, sorted(cleaned))
                )
            )
            sh = ip.solve(poly, **kw)
            if sh is None:
                print(f"cell {c}: no inscribed shape at all")
                continue
            rl = run_lengths(sh)
            have = Counter(rl.tolist())
            n_host = sum(have[L] for L in hosts)
            print(
                f"cell {c}: run totals that can host a class piece = {hosts or 'none'}; "
                f"shape has {n_host} such runs; max run in shape = {int(rl.max())}; "
                f"buildable without class: {bool(assign_pieces(rl, cleaned))}, "
                f"with class: {bool(assign_pieces(rl, cleaned, must_use))}"
            )
    finally:
        ip.CELL, ip.MAX_RUN = old_cell, old_run


if __name__ == "__main__":
    L_shape = Polygon([(0, 0), (200, 0), (200, 80), (90, 80), (90, 210), (0, 210)])
    stock = {5: 60, 10: 40, 15: 30, 20: 15, 30: 8}
    for name, ang in [
        ("unlimited angles", None),
        ("right=60", {"right": 60}),
        ("right=20", {"right": 20}),
        ("straight=0", {"straight": 0}),
        ("straight=10, right=40", {"straight": 10, "right": 40}),
    ]:
        r, plan, how = solve_with_stock(L_shape, stock, angle_stock=ang)
        if r is None:
            print(f"{name:24s} -> {how}")
            continue
        print(f"{name:24s} -> area {r.area:6.0f}  {how:24s} uses {angle_usage(plan)}")
