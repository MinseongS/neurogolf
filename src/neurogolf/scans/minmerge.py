"""Overfit public-dump min-merge (S18 lever, LB-proven +4.46 in one pass from urad 7242.52).

Baselines against `submission/overfit_nets/` (the ACTIVE best submission). In overfit mode
the gate is bundled fail=0 ONLY (constant dataset => permanent), so every cheaper
bundled-fail=0 net is pure, permanent upside. STATIC-cost prefilter first, then a full
evaluate on plausibly-cheaper candidates.

ROUTINE
  1. Pull new higher-scoring uploaders (kaggle kernels output <ref> -p DIR; unzip submission.zip).
  2. Min-merge:  ng mine-public mine/urad_nets mine/poby_nets ...
  3. Adopt:      ng mine-public ... --apply   (each winner routed through the single adopt() gate)

Ported (Task 13): --apply no longer copies files itself; every winner goes through
neurogolf.adoption.adopt() (the single catastrophe-prevention gate). min-merge adoption is
always paired with insight reverse-engineering (deep lane), so a pointer line is printed after.
"""
import glob
import os

import numpy as np
import onnx
from onnx import shape_inference

from neurogolf.paths import OVERFIT_NETS
from neurogolf.scoring import load_task, evaluate

DT = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2}


def static_cost(p):
    """approx grader (memory+params): sum counted node-output bytes + initializer element count."""
    try:
        m = onnx.load(str(p))
        m = shape_inference.infer_shapes(m)
    except Exception:
        return None
    g = m.graph
    vi = {}
    for v in list(g.value_info) + list(g.output) + list(g.input):
        dims = tuple(d.dim_value if d.HasField("dim_value") else 0 for d in v.type.tensor_type.shape.dim)
        vi[v.name] = (v.type.tensor_type.elem_type, dims)
    mem = 0
    for n in g.node:
        for o in n.output:
            if o in ("input", "output"):
                continue
            et, dims = vi.get(o, (None, None))
            if not dims or any(d == 0 for d in dims):
                continue
            c = 1
            for d in dims:
                c *= d
            mem += c * DT.get(et, 4)
    params = sum(int(np.prod(i.dims)) for i in g.initializer)
    return mem + params


def find_net(dumpdir, t):
    for pat in (f"{dumpdir}/task{t:03d}.onnx", f"{dumpdir}/**/task{t:03d}.onnx"):
        g = glob.glob(pat, recursive=True)
        if g:
            return g[0]
    return None


def mine(dumps: list, margin: int = 0, apply: bool = False) -> list[dict]:
    dumps = [str(d) for d in dumps]
    wins = []  # dicts: {delta_pts, task, dump, path, memory, params}
    for t in range(1, 401):
        op = OVERFIT_NETS / f"task{t:03d}.onnx"
        if not op.exists():
            continue
        oc = static_cost(op)
        if oc is None:
            continue
        # collect statically-cheaper candidates across dumps
        cands = []
        for dd in dumps:
            cp = find_net(dd, t)
            if not cp:
                continue
            cc = static_cost(cp)
            if cc is not None and cc < oc - margin:
                cands.append((cc, dd, cp))
        if not cands:
            continue
        # isolated exact eval: ours + each cheaper candidate
        our = evaluate(str(op), load_task(t))
        best = None
        for cc, dd, cp in sorted(cands):
            r = evaluate(cp, load_task(t))
            if r["ok"] and r["fail"] == 0 and r["points"] > our["points"] + 1e-6:
                if best is None or r["points"] > best[0]:
                    best = (r["points"], dd, cp, r["memory"], r["params"])
        if best:
            wins.append({"delta_pts": round(best[0] - our["points"], 4), "task": t,
                         "dump": os.path.basename(best[1]), "path": best[2],
                         "memory": best[3], "params": best[4]})

    wins.sort(key=lambda w: -w["delta_pts"])
    total = 0.0
    print(f"{'dP':>7} {'task':>4} {'dump':>28} {'mem':>6} {'par':>5}")
    for w in wins:
        total += w["delta_pts"]
        print(f"{w['delta_pts']:7.4f} {w['task']:4d} {w['dump']:>28} {w['memory']:6d} {w['params']:5d}")
    print(f"\nTOTAL adoptable: +{round(total, 4)} across {len(wins)} tasks")

    if apply and wins:
        adopted = []
        from neurogolf.adoption import adopt
        for w in wins:
            try:
                adopt(w["path"], w["task"], note=f"min-merge from {w['dump']}")
                adopted.append(w["task"])
            except SystemExit as e:
                print(f"  gate REJECT task{w['task']:03d}: {e}")
        print(f"\nADOPTED {len(adopted)} nets via adopt() gate: "
              f"{', '.join(f'task{t:03d}' for t in adopted)}")
        # deep-lane pointer (min-merge adoption is always paired with insight reverse-engineering)
        print("다음: ng scan public_autopsy 후 playbook/public-insight.md 딥레인 실행 "
              f"(채택 태스크: {', '.join(f'task{t:03d}' for t in adopted)})")
    return wins
