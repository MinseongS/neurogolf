"""Scan 400 tasks for SPARSE-EDIT class: output = input with few cells changed,
same grid shape. Such tasks admit ScatterND-into-free-input (mem~=0 scatter; cost
dominated by a small [K,4] int64 index, K ~ 2*max_changed_cells).

Rank by: current points ASC (overpay headroom) among tasks whose worst-case edit is
small enough that a scatter index beats the incumbent.

Usage: uv run python -m reports.scripts.sparse_edit_scan
"""
import json, glob, os, math
import numpy as np

DATA = 'data'


def load_cost():
    cost = {}
    p = 'reports/SCOREBOARD.md'
    if os.path.exists(p):
        for line in open(p):
            parts = [x.strip() for x in line.split('|')]
            if len(parts) >= 6 and parts[1].isdigit():
                try:
                    t = int(parts[1]); mem = int(parts[3]); par = int(parts[4]); pts = float(parts[5])
                    cost[t] = (mem, par, pts)
                except Exception:
                    pass
    return cost


def main():
    cost = load_cost()
    rows = []
    for path in sorted(glob.glob(f'{DATA}/task*.json')):
        t = int(os.path.basename(path)[4:7])
        d = json.load(open(path))
        examples = d.get('train', []) + d.get('test', []) + d.get('arc-gen', [])
        if not examples:
            continue
        max_edit = 0
        same_shape = True
        trivial = True
        n = 0
        for ex in examples:
            ig = np.array(ex['input']); og = np.array(ex['output'])
            if ig.shape != og.shape:
                same_shape = False
                break
            n += 1
            diff = int((ig != og).sum())
            if diff > 0:
                trivial = False
            if diff > max_edit:
                max_edit = diff
        if not same_shape or trivial:
            continue
        mem, par, pts = cost.get(t, (None, None, None))
        # scatter cost estimate: index [~2*max_edit, 4] int64 = 2*max_edit*4*8 bytes + ~detection
        Kbytes = 2 * max_edit * 4 * 8
        est = Kbytes + 800  # rough: index + modest detection/build
        est_pts = 25 - math.log(max(1, est))
        gain = (est_pts - pts) if pts is not None else None
        rows.append((t, max_edit, mem, par, pts, est, est_pts, gain))
    # candidates: gain > 0 (scatter estimate beats incumbent), rank by gain desc
    cand = [r for r in rows if r[7] is not None and r[7] > 0.3]
    cand.sort(key=lambda r: -r[7])
    print(f'{len(rows)} non-trivial same-shape tasks; {len(cand)} with est scatter gain >0.3\n')
    print(f'{"task":>4} {"maxEdit":>7} {"curMem":>7} {"curPts":>7} {"estCost":>7} {"estPts":>7} {"gain":>6}')
    for t, me, mem, par, pts, est, ep, g in cand[:40]:
        print(f'{t:4d} {me:7d} {str(mem):>7} {pts:7.2f} {est:7d} {ep:7.2f} {g:+6.2f}')
    print(f'\ntop-{min(40,len(cand))} est total gain: +{sum(r[7] for r in cand[:40]):.1f}')


if __name__ == '__main__':
    main()
