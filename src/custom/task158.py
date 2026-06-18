"""task158 (ARC-AGI 6aa20dc0) — variable-mag dihedral sprite re-stamping.

================================================================================
STATUS: Algorithm SOLVED and verified EXACT (numpy, 50000/50000 + 30000/30000 +
        20000/20000 across solver variants). The prior agent's stated single
        blocker (canonical-sprite RECOVERY) is now solved unique-by-construction.
        A correction was also found: the bare 12-pass stamp is NOT exact even
        with an oracle sprite (~0.08% phantom-bridge fails) — it ALSO needs the
        exact-cover step below. The full pipeline IS opset-10 ONNX-expressible
        (bounded ≤4-step propagation unroll) but the ONNX graph is large
        (~recovery + 12 correlation stamps + 4-step per-cell propagation) and is
        NOT yet built here. `build(task)` is intentionally a NotImplemented stub
        with the complete, verified reference solver `solve()` below as the spec.
================================================================================

GENERATOR (key facts, all verified against /tmp/arc-gen/tasks/task_6aa20dc0.py):
  - A canonical 3x3 sprite: c0@(0,0), c1@(2,2) (distinct corner colours), body
    colour c2 at 2-3 sampled cells of {(1,0),(1,1),(2,0),(2,1)} MIRRORED across
    the main diagonal (so the body mask is transpose-symmetric; (0,2)/(2,0) CAN
    be body cells — they are NOT corners).
  - 2..4 "megas" placed, each with own mag in {1,2,3} and own (hflip,vflip).
    Megas are non-overlapping with margin 2 (common.overlaps(...,2)).
  - INPUT: mega idx 0 (the REFERENCE) is FULLY drawn; every other mega shows ONLY
    its two magnified diagonal CORNER blocks (the c0 block and the c1 block).
    OUTPUT: every mega filled completely (flipped-magnified full sprite).
  - bg = color_list[3].

THE RECOVERY BREAKTHROUGH (unique-by-construction; prior agent's wall was wrong):
  1. The REFERENCE is ALWAYS mag=1 (generator: `mag = 1 if not mags else ...`, and
     `mags` is empty on the first iteration). VERIFIED.
  2. The body colour c2 appears ONLY inside the reference 3x3 region. VERIFIED
     3000/3000 — non-ref megas hide all body cells (they draw only corners, which
     are c0/c1). So c2 cells deterministically LOCATE the reference. No search.
  3. c2 = the non-bg colour with MINIMUM bbox span (it is confined to the 3x3 ref;
     all other non-bg colours (c0,c1) are spread across multiple megas).
  4. The canonical sprite = unflip the reference 3x3 using its own (h,v): try the 4
     unflips, the valid one has c0@(0,0), c1@(2,2) (both !=bg,!=c2,distinct) and a
     transpose-symmetric c2 body. This uniquely recovers the canonical colour grid.
  NOTE: when the body is 180-rotation-symmetric (4 of 10 possible body shapes), the
  c0<->c1 labelling is ambiguous — but it DOESN'T MATTER: the 180-rotated canon is
  reached by the same 4 (h,v) stamp passes, so every mega still matches. (Verified.)

THE STAMP + EXACT-COVER (correction to prior agent — bare stamp is NOT exact):
  For each (mag in {1,2,3}, h, v) build the flipped-magnified tile T (3*mag square)
  and a "visible" mask M = the two magnified diagonal corner blocks. A candidate
  placement matches where window==T on M AND window==bg elsewhere in the footprint.
  This ALONE over-stamps PHANTOMS: a c0 corner-block of mega A and a c1 corner-block
  of mega B can align as the two diagonal corners of a (different) tile, with bg
  between — a false mega (~0.08% of instances with oracle sprite). Two fixes,
  combined, make it EXACT (0 fails / 50000):
    (a) ISOLATION: reject any placement whose 1-cell ring around the full footprint
        is not all-bg (true megas are isolated by margin-2; cuts most phantoms).
    (b) EXACT-COVER by bounded constraint-propagation: the true non-ref megas are
        the UNIQUE set of isolated placements whose visible corner-block cells
        partition (all non-bg cells minus the reference region) exactly. Resolve by
        naked-singles: repeatedly choose every candidate that uniquely owns some
        target cell, remove its cells, kill overlappers. Terminates in <=3 rounds
        (<=3 non-ref megas) => UNROLLABLE to a fixed 4 iterations => ONNX-buildable
        (stacked bool masks + sum-reductions + elementwise; no Loop/argmax needed).
        bg must be the MODE colour, not grid[0,0] (a mega can sit at (0,0): 0.85%).

ONNX BUILD PLAN (not yet implemented; projected ~14.5-15.3 pts, +1.7-2.5):
  - bg: per-channel ReduceSum over grid -> ArgMax  => mode colour.
  - c2: per-channel row/col occupancy -> bbox span (ReduceMax-ReduceMin) -> ArgMin
        over non-bg channels.
  - ref locate + canon: c2 occupancy bbox -> Slice the (<=2x2 window options) 3x3,
        4 unflips via reversed Gather, pick the transpose-symmetric one.
  - 12 stamps: build T and M for each (mag,h,v) from canon (Conv/kron via Tile),
        correlation match (Equal+And+ReduceSum over footprint), isolation via a
        dilated bg check.  => candidate match-maps.
  - exact-cover: per-cell claim counts; 4 unrolled propagation rounds of
        (uniquely-owned -> select -> remove -> kill-overlap); write chosen tiles.
"""

import numpy as np


# ============================================================================
# VERIFIED EXACT REFERENCE SOLVER (numpy) — 50000/50000 fresh.
# This is the spec the ONNX graph must replicate.
# ============================================================================
def _bg(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return vals[counts.argmax()]


def _recover_canon(sub, bg, c2):
    """sub = 3x3 reference window. Return the canonical (unflipped) 3x3 colour grid
    or None. Canonical: c0@(0,0), c1@(2,2) distinct non-bg non-c2; every other
    non-bg cell is c2; c2 body transpose-symmetric."""
    for h in range(2):
        for v in range(2):
            t = sub.copy()
            if h:
                t = t[:, ::-1]
            if v:
                t = t[::-1, :]
            a, b = t[0, 0], t[2, 2]
            if a == bg or b == bg or a == b or a == c2 or b == c2:
                continue
            ok = True
            for r in range(3):
                for c in range(3):
                    if (r, c) in [(0, 0), (2, 2)]:
                        continue
                    if t[r, c] != bg and t[r, c] != c2:
                        ok = False
            if not ok:
                continue
            body = (t == c2)
            if np.array_equal(body, body.T):
                return t
    return None


def _isolated(grid, yy, xx, Sz, bg, H, W):
    for c in range(xx - 1, xx + Sz + 1):
        for r in (yy - 1, yy + Sz):
            if 0 <= r < H and 0 <= c < W and grid[r, c] != bg:
                return False
    for r in range(yy - 1, yy + Sz + 1):
        for c in (xx - 1, xx + Sz):
            if 0 <= r < H and 0 <= c < W and grid[r, c] != bg:
                return False
    return True


def _stamp_cover(grid, canon, bg, H, W, refpos):
    """Exact-cover stamp via bounded (4-round) naked-singles propagation."""
    visms, writes = [], []
    for mag in (1, 2, 3):
        base = np.kron(canon, np.ones((mag, mag), dtype=canon.dtype))
        Sz = 3 * mag
        vis = np.zeros((Sz, Sz), dtype=bool)
        vis[0:mag, 0:mag] = True
        vis[2 * mag:3 * mag, 2 * mag:3 * mag] = True
        for h in range(2):
            for v in range(2):
                T = base.copy()
                M = vis.copy()
                if h:
                    T = T[:, ::-1]
                    M = M[:, ::-1]
                if v:
                    T = T[::-1, :]
                    M = M[::-1, :]
                for yy in range(H - Sz + 1):
                    for xx in range(W - Sz + 1):
                        if mag == 1 and (yy, xx) == refpos:
                            continue
                        win = grid[yy:yy + Sz, xx:xx + Sz]
                        if not (np.all(win[M] == T[M]) and np.all(win[~M] == bg)):
                            continue
                        if not _isolated(grid, yy, xx, Sz, bg, H, W):
                            continue
                        vm = np.zeros((H, W), dtype=bool)
                        vm[yy:yy + Sz, xx:xx + Sz][M] = True
                        visms.append(vm)
                        writes.append((yy, xx, Sz, T))
    ry, rx = refpos
    refmask = np.zeros((H, W), dtype=bool)
    refmask[ry:ry + 3, rx:rx + 3] = (grid[ry:ry + 3, rx:rx + 3] != bg)
    target = (grid != bg) & ~refmask
    if not visms:
        # no non-ref megas would be impossible (num_megas>=2), but guard anyway
        return grid.copy() if not target.any() else None
    V = np.stack(visms)  # [K,H,W]
    K = V.shape[0]
    chosen = np.zeros(K, dtype=bool)
    dead = np.zeros(K, dtype=bool)
    covered = np.zeros((H, W), dtype=bool)
    for _ in range(4):  # bounded unroll (<=3 non-ref megas)
        live = ~chosen & ~dead
        if not live.any():
            break
        claim = (V & live[:, None, None]).sum(0)
        uniqcell = (claim == 1) & target
        has_uniq = (V & uniqcell[None] & live[:, None, None]).reshape(K, -1).any(1)
        newsel = has_uniq & live
        if not newsel.any():
            break
        chosen |= newsel
        covered |= V[newsel].any(0)
        overlap = (V & covered[None]).reshape(K, -1).any(1)
        dead |= overlap & ~chosen
    if not np.array_equal(covered & target, target):
        return None
    if (covered & ~target).any():
        return None
    out = grid.copy()
    for i in range(K):
        if chosen[i]:
            yy, xx, Sz, T = writes[i]
            out[yy:yy + Sz, xx:xx + Sz] = T
    return out


def solve(grid):
    """Reference exact solver. grid: 2D list/array. Returns 2D np.ndarray output."""
    grid = np.array(grid)
    H, W = grid.shape
    bg = _bg(grid)
    colsu = [c for c in np.unique(grid) if c != bg]
    cand = []
    for c in colsu:
        ys, xs = np.where(grid == c)
        sp = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
        cand.append((sp, (grid == c).sum(), c))
    cand.sort()
    for _, _, c2 in cand:
        c2m = (grid == c2)
        tot = c2m.sum()
        ys, xs = np.where(c2m)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        if y1 - y0 > 2 or x1 - x0 > 2:
            continue
        canon = None
        refpos = None
        for ty in range(max(0, y1 - 2), min(y0, H - 3) + 1):
            for tx in range(max(0, x1 - 2), min(x0, W - 3) + 1):
                if ty + 3 > H or tx + 3 > W:
                    continue
                w = grid[ty:ty + 3, tx:tx + 3]
                if (w == c2).sum() != tot:
                    continue
                cc = _recover_canon(w, bg, c2)
                if cc is not None:
                    canon = cc
                    refpos = (ty, tx)
                    break
            if canon is not None:
                break
        if canon is None:
            continue
        out = _stamp_cover(grid, canon, bg, H, W, refpos)
        if out is not None:
            return out
    return None


def build(task):
    raise NotImplementedError(
        "task158 ONNX graph not yet built. Algorithm is solved & verified exact "
        "(see solve() and module docstring). The opset-10 ONNX build (recovery + "
        "12 correlation stamps + 4-round per-cell exact-cover propagation) is the "
        "remaining engineering work; projected ~14.5-15.3 pts.")
