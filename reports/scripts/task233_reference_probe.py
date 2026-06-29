#!/usr/bin/env python3
"""Task233 semantic reference probe.

This is research scaffolding, not an ONNX candidate.  It tests the discovered
rotation-only component-template rule against stored and fresh task233 examples.
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np

from src.genverify import load_gen
from src.harness import load_task


def block(cnt: np.ndarray, thr: int = 4) -> tuple[int, int]:
    best = None
    i = 0
    while i < len(cnt):
        if cnt[i] >= thr:
            j = i
            while j + 1 < len(cnt) and cnt[j + 1] >= thr:
                j += 1
            score = (j - i + 1, int(cnt[i : j + 1].sum()))
            if best is None or score > best[2]:
                best = (i, j, score)
            i = j + 1
        else:
            i += 1
    if best is None:
        return 0, len(cnt) - 1
    return best[0], best[1]


def span(cnt: np.ndarray, thr: int = 4) -> tuple[int, int]:
    """Return first..last positions above threshold, allowing holes inside.

    The red rectangle can contain 3x3 black components that create low-count
    rows/cols inside the true box.  After one axis has been localized, the robust
    bbox is the span of strong rows/cols, not necessarily the longest contiguous
    block.
    """
    idx = np.flatnonzero(cnt >= thr)
    if len(idx) == 0:
        return 0, len(cnt) - 1
    return int(idx[0]), int(idx[-1])


def red_bbox(g: np.ndarray, mode: str) -> tuple[int, int, int, int]:
    red = g == 2
    if mode == "simple":
        r0, r1 = block(red.sum(1), 4)
        c0, c1 = block(red.sum(0), 4)
        return r0, r1, c0, c1
    if mode == "row_first":
        r0, r1 = block(red.sum(1), 4)
        c0, c1 = block(red[r0 : r1 + 1, :].sum(0), 4)
        return r0, r1, c0, c1
    if mode == "iter":
        r0, r1 = block(red.sum(1), 4)
        c0, c1 = block(red[r0 : r1 + 1, :].sum(0), 4)
        r0, r1 = block(red[:, c0 : c1 + 1].sum(1), 4)
        c0, c1 = block(red[r0 : r1 + 1, :].sum(0), 4)
        return r0, r1, c0, c1
    if mode == "iter_span":
        r0, r1 = block(red.sum(1), 4)
        c0, c1 = block(red[r0 : r1 + 1, :].sum(0), 4)
        r0, r1 = span(red[:, c0 : c1 + 1].sum(1), 4)
        c0, c1 = span(red[r0 : r1 + 1, :].sum(0), 4)
        return r0, r1, c0, c1
    if mode == "iter_bounded_span":
        r0a, r1a = block(red.sum(1), 4)
        c0a, c1a = block(red[r0a : r1a + 1, :].sum(0), 4)
        rr0, rr1 = span(red[r0a : r1a + 1, c0a : c1a + 1].sum(1), 4)
        r0, r1 = r0a + rr0, r0a + rr1
        cc0, cc1 = span(red[r0 : r1 + 1, c0a : c1a + 1].sum(0), 4)
        c0, c1 = c0a + cc0, c0a + cc1
        return r0, r1, c0, c1
    raise ValueError(mode)


def rotations(mask: np.ndarray) -> list[np.ndarray]:
    out = []
    for k in range(4):
        m = np.rot90(mask, k)
        if not any(np.array_equal(m, x) for x in out):
            out.append(m)
    return out


def components(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    if conn8:
        neigh = [(dr, dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if not (dr == 0 and dc == 0)]
    else:
        neigh = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in neigh:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def outside_sprites(g: np.ndarray, bbox: tuple[int, int, int, int]) -> list[tuple[int, np.ndarray]]:
    r0, r1, c0, c1 = bbox
    outside = g != 0
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    out = []
    for comp in components(outside, conn8=False):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        br0, br1, bc0, bc1 = min(rs), max(rs), min(cs), max(cs)
        if br1 - br0 + 1 != 3 or bc1 - bc0 + 1 != 3:
            continue
        patch = g[br0 : br1 + 1, bc0 : bc1 + 1]
        colors = [int(v) for v in set(patch.ravel()) if v not in (0, 2)]
        if len(colors) == 1:
            out.append((colors[0], patch == 2))
    return out


def solve(grid: list[list[int]], bbox_mode: str = "simple", conn8: bool = True) -> np.ndarray:
    g = np.asarray(grid, dtype=np.int64)
    bbox = red_bbox(g, bbox_mode)
    r0, r1, c0, c1 = bbox
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    black = crop == 0
    out = np.full(crop.shape, 2, dtype=np.int64)
    sprites = outside_sprites(g, bbox)
    for comp in components(black, conn8=conn8):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        br0, br1, bc0, bc1 = min(rs), max(rs), min(cs), max(cs)
        if br1 - br0 + 1 > 3 or bc1 - bc0 + 1 > 3:
            continue
        done = False
        for color, shape in sprites:
            for m in rotations(shape):
                # Candidate anchors that could cover this component bbox.
                for ar in range(br1 - 2, br0 + 1):
                    for ac in range(bc1 - 2, bc0 + 1):
                        if (
                            0 <= ar
                            and 0 <= ac
                            and ar + 3 <= black.shape[0]
                            and ac + 3 <= black.shape[1]
                            and np.array_equal(black[ar : ar + 3, ac : ac + 3], m)
                        ):
                            out[ar : ar + 3, ac : ac + 3] = color
                            out[ar : ar + 3, ac : ac + 3][m] = 2
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
            if done:
                break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", type=int, default=200)
    parser.add_argument(
        "--bbox",
        choices=["simple", "row_first", "iter", "iter_span", "iter_bounded_span"],
        default="simple",
    )
    parser.add_argument("--conn4", action="store_true")
    args = parser.parse_args()
    task = load_task(233)
    for split in ("train", "test"):
        for i, ex in enumerate(task[split]):
            pred = solve(ex["input"], args.bbox, conn8=not args.conn4)
            ok = np.array_equal(pred, np.asarray(ex["output"]))
            print(f"{split} {i}: {ok} pred={pred.shape} target={np.asarray(ex['output']).shape}")
            if not ok:
                return
    gen = load_gen(233)
    ok = run = tries = 0
    while run < args.fresh and tries < args.fresh * 20:
        tries += 1
        ex = gen.generate()
        if max(len(ex["input"]), len(ex["input"][0]), len(ex["output"]), len(ex["output"][0])) > 30:
            continue
        run += 1
        pred = solve(ex["input"], args.bbox, conn8=not args.conn4)
        if not np.array_equal(pred, np.asarray(ex["output"])):
            print(f"fresh fail at {run}: pred={pred.shape} target={np.asarray(ex['output']).shape}")
            break
        ok += 1
    print(f"fresh {ok}/{run} tries={tries} bbox={args.bbox} conn8={not args.conn4}")


if __name__ == "__main__":
    main()
