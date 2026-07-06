#!/usr/bin/env python3
"""Behavioral semantic probes for the task index. Each probe: (samples) -> (value, confidence).
samples = list of (input_grid, output_grid) numpy int arrays. Add a probe here + bump
PROBE_VERSION to give the matching engine a new feature to slice on."""
from __future__ import annotations
import numpy as np

PROBE_VERSION = 1


def _agree(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def probe_shape_relation(samples):
    rels = []
    for i, o in samples:
        i = np.asarray(i); o = np.asarray(o)
        if i.shape == o.shape:
            rels.append("equal")
        elif o.shape[0] % max(i.shape[0], 1) == 0 and o.shape[1] % max(i.shape[1], 1) == 0 \
                and o.shape[0] // i.shape[0] == o.shape[1] // i.shape[1] and o.shape[0] > i.shape[0]:
            rels.append(f"scaled:{o.shape[0] // i.shape[0]}x")
        elif o.shape[0] <= i.shape[0] and o.shape[1] <= i.shape[1]:
            rels.append("crop")
        else:
            rels.append("other")
    from collections import Counter
    val, cnt = Counter(rels).most_common(1)[0]
    return val, cnt / len(rels)


def probe_delta(samples):
    copy_fracs, changed = [], []
    for i, o in samples:
        i = np.asarray(i); o = np.asarray(o)
        if i.shape != o.shape:
            copy_fracs.append(0.0); changed.append(int(o.size)); continue
        same = (i == o)
        copy_fracs.append(float(same.mean()))
        changed.append(int((~same).sum()))
    return {"copy_frac": round(float(np.mean(copy_fracs)), 4),
            "changed_cells": int(np.mean(changed))}, 1.0


def probe_color_source(samples):
    # FIXED_DELTA: the set of colours on changed cells is constant across examples.
    # SMALL_K: <=2 distinct changed-colours per example. COPY: changed colours vary and come
    # from input. RESHAPE: shapes differ.
    delta_palettes = []
    small = True
    for i, o in samples:
        i = np.asarray(i); o = np.asarray(o)
        if i.shape != o.shape:
            return "RESHAPE", 1.0
        ch = o[i != o]
        pal = frozenset(int(c) for c in np.unique(ch))
        delta_palettes.append(pal)
        if len(pal) > 2:
            small = False
    nonempty = [p for p in delta_palettes if p]
    if nonempty and all(p == nonempty[0] for p in nonempty):
        return "FIXED_DELTA", _agree([p == nonempty[0] for p in nonempty])
    if small:
        return "SMALL_K", 1.0
    return "COPY", 1.0


def probe_d4_transform_of_input(samples):
    def d4(g):
        yield g; yield np.rot90(g, 1); yield np.rot90(g, 2); yield np.rot90(g, 3)
        yield np.fliplr(g); yield np.flipud(g); yield g.T; yield np.fliplr(np.rot90(g, 1))
    flags = []
    for i, o in samples:
        i = np.asarray(i); o = np.asarray(o)
        flags.append(any(t.shape == o.shape and np.array_equal(t, o) for t in d4(i)))
    return _agree(flags) > 0.9, _agree(flags)


def _solid_rects_count(mask):
    # crude: number of connected axis-aligned true-runs; caps at 99. mask is 2D bool.
    from itertools import product
    seen = np.zeros_like(mask, dtype=bool); rects = 0
    for r, c in product(range(mask.shape[0]), range(mask.shape[1])):
        if mask[r, c] and not seen[r, c]:
            r2 = r
            while r2 + 1 < mask.shape[0] and mask[r2 + 1, c]:
                r2 += 1
            c2 = c
            while c2 + 1 < mask.shape[1] and mask[r, c2:c2 + 2].all():
                c2 += 1
            seen[r:r2 + 1, c:c2 + 1] = True; rects += 1
            if rects > 99:
                break
    return rects


def probe_separable_rect_output(samples):
    counts = []
    for i, o in samples:
        i = np.asarray(i); o = np.asarray(o)
        if i.shape != o.shape:
            counts.append(99); continue
        counts.append(_solid_rects_count(i != o))
    med = int(np.median(counts))
    return {"is": med <= 4, "n_rects": med}, 1.0


def probe_locality_radius(samples):
    # smallest radius r in {0,1,2} s.t. every changed cell's new colour is determined by the
    # input (2r+1)^2 window (no collisions across samples); else None.
    for r in (0, 1, 2):
        table = {}
        ok = True
        for i, o in samples:
            i = np.asarray(i); o = np.asarray(o)
            if i.shape != o.shape:
                ok = False; break
            pad = np.pad(i, r, constant_values=-1)
            for rr in range(i.shape[0]):
                for cc in range(i.shape[1]):
                    win = tuple(pad[rr:rr + 2 * r + 1, cc:cc + 2 * r + 1].ravel())
                    v = int(o[rr, cc])
                    if table.get(win, v) != v:
                        ok = False; break
                    table[win] = v
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return r, 1.0
    return None, 1.0


def probe_flood_ccl(samples):
    # heuristic: output changes cluster into few large connected components AND fill background
    fracs = []
    for i, o in samples:
        i = np.asarray(i); o = np.asarray(o)
        if i.shape != o.shape:
            fracs.append(0.0); continue
        changed = (i != o)
        filled_bg = ((i == 0) & changed).sum()
        fracs.append(filled_bg / max(changed.sum(), 1))
    return float(np.mean(fracs)) > 0.6, 1.0


def probe_periodicity(samples):
    periods = []
    for i, o in samples:
        o = np.asarray(o)
        p = 0
        for cand in range(1, max(2, o.shape[1] // 2 + 1)):
            if o.shape[1] % cand == 0 and all(
                    np.array_equal(o[:, :cand], o[:, k:k + cand]) for k in range(0, o.shape[1], cand)):
                p = cand; break
        periods.append(p)
    from collections import Counter
    val, cnt = Counter(periods).most_common(1)[0]
    return int(val), cnt / len(periods)


def probe_n_objects_est(samples):
    counts = []
    for i, o in samples:
        i = np.asarray(i)
        counts.append(int(len(np.unique(i)) - (1 if 0 in i else 0)))
    return int(np.median(counts)), 1.0


def probe_output_colors(samples):
    cols = set()
    for i, o in samples:
        cols |= set(int(c) for c in np.unique(o))
    return sorted(cols), 1.0


_PROBES = {
    "shape_relation": probe_shape_relation, "delta": probe_delta,
    "color_source": probe_color_source, "d4_transform_of_input": probe_d4_transform_of_input,
    "separable_rect_output": probe_separable_rect_output, "locality_radius": probe_locality_radius,
    "flood_ccl": probe_flood_ccl, "periodicity": probe_periodicity,
    "n_objects": probe_n_objects_est, "output_colors": probe_output_colors,
}


def run_probes(samples) -> dict:
    out = {}
    for name, fn in _PROBES.items():
        try:
            value, conf = fn(samples)
        except Exception as e:  # a probe must never crash the whole build
            value, conf = None, 0.0
        out[name] = {"value": value, "confidence": round(float(conf), 3)}
    return out
