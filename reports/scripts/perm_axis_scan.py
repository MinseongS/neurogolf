"""Scan all 400 tasks for the SINGLE-AXIS PERMUTATION class:
output is a pure row-only (or col-only) reordering of input rows (cols),
holding across ALL examples. Such tasks admit output = Gather(input, perm, axis)
— one node, output free, perm is a [N] index → mem 0, cost ~= detection only.

Usage: uv run python -m reports.scripts.perm_axis_scan
"""
import json, glob, os
import numpy as np

DATA = 'data'


def row_gather_feasible(ig, og):
    """output = Gather(input, idx, axis=rows) feasible iff every output row
    equals some input row (each out row is selectable from in rows)."""
    inrows = set(tuple(r) for r in ig.tolist())
    return all(tuple(r) in inrows for r in og.tolist())


def col_gather_feasible(ig, og):
    incols = set(tuple(c) for c in ig.T.tolist())
    return all(tuple(c) in incols for c in og.T.tolist())


def classify(examples):
    """Return (row_ok_all, col_ok_all, nontrivial, n): whether a single-axis
    Gather along rows (or cols) can realise the output across ALL examples."""
    row_ok = True
    col_ok = True
    nontrivial = False
    n = 0
    for ex in examples:
        ig = np.array(ex['input']); og = np.array(ex['output'])
        if ig.shape != og.shape:
            return (False, False, False, 0)
        n += 1
        if not np.array_equal(ig, og):
            nontrivial = True
        if row_ok and not row_gather_feasible(ig, og):
            row_ok = False
        if col_ok and not col_gather_feasible(ig, og):
            col_ok = False
        if not row_ok and not col_ok:
            break
    return (row_ok, col_ok, nontrivial, n)


def load_cost():
    """current mem/params/points per task from SCOREBOARD.md if present."""
    cost = {}
    p = 'reports/SCOREBOARD.md'
    if os.path.exists(p):
        for line in open(p):
            parts = [x.strip() for x in line.split('|')]
            if len(parts) >= 5 and parts[1].isdigit():
                try:
                    t = int(parts[1]); mem = int(parts[3]); par = int(parts[4]); pts = float(parts[5])
                    cost[t] = (mem, par, pts)
                except Exception:
                    pass
    return cost


def main():
    cost = load_cost()
    hits = []
    for path in sorted(glob.glob(f'{DATA}/task*.json')):
        t = int(os.path.basename(path)[4:7])
        d = json.load(open(path))
        examples = d.get('train', []) + d.get('test', []) + d.get('arc-gen', [])
        if not examples:
            continue
        row_ok, col_ok, nontrivial, n = classify(examples)
        if nontrivial and (row_ok or col_ok):
            axis = 'ROW' if row_ok else 'COL'
            if row_ok and col_ok:
                axis = 'BOTH'
            mem, par, pts = cost.get(t, (None, None, None))
            hits.append((t, axis, n, mem, par, pts))
    # rank by current points ASC (lowest points = most headroom) then mem DESC
    hits.sort(key=lambda x: (x[5] if x[5] is not None else 99, -(x[3] or 0)))
    print(f'{len(hits)} single-axis permutation tasks found\n')
    print(f'{"task":>4} {"axis":>4} {"n":>4} {"mem":>6} {"par":>4} {"pts":>7}  potential(21.5-pts)')
    for t, axis, n, mem, par, pts in hits:
        gain = (21.5 - pts) if pts is not None else None
        print(f'{t:4d} {axis:>4} {n:4d} {str(mem):>6} {str(par):>4} {str(pts):>7}  {f"{gain:+.2f}" if gain is not None else "?"}')
    # total potential
    tot = sum((21.5 - pts) for _,_,_,_,_,pts in hits if pts is not None)
    print(f'\nrough total headroom if all reach ~21.5: +{tot:.1f}')


if __name__ == '__main__':
    main()
