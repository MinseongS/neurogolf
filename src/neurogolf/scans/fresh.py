"""Fresh-generalization gate for a NeuroGolf candidate (diagnostic).

Builds incumbent (src.custom.taskNNN) and optional candidate, generates N fresh
instances via the arc-gen generator, requires candidate fail=0 vs ground-truth.

Ported (Task 13): exposed as fresh_check(task_num, candidate=None, n=1500) -> (passes, runs);
cache dir -> CANDIDATES / "fresh_cache".
"""
import importlib
import json
import os
import sys

import numpy as np
import onnxruntime as ort

from neurogolf.paths import ROOT, STATE, CANDIDATES
from neurogolf.scoring import load_task, sanitize_model

# repo root on path so `import src.custom.*` works.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# appended (NOT insert-0): arc-gen has its own src/ that would shadow ours.
_ARCGEN = str(ROOT / "arc-gen")
if not os.path.isdir(_ARCGEN):
    raise FileNotFoundError(f"repo-local arc-gen not found: {_ARCGEN}")
if _ARCGEN not in sys.path:
    sys.path.append(_ARCGEN)


def to_onehot(grid):
    g = np.asarray(grid)
    H, W = g.shape
    if max(H, W) > 30:
        return None
    x = np.zeros((1, 10, 30, 30), dtype=np.float32)
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x[0, g.astype(np.int64), rr, cc] = 1.0
    return x


def _cached_examples(task_num, n):
    path = CANDIDATES / "fresh_cache" / f"task{task_num:03d}.npz"
    z = np.load(path)
    ins, outs, hs, ws = z["inputs"], z["outputs"], z["heights"], z["widths"]
    ohs = z["oheights"] if "oheights" in z else hs
    ows = z["owidths"] if "owidths" in z else ws
    if len(ins) < n:
        print(f"  (cache has only {len(ins)} < {n})")
    for i in range(min(n, len(ins))):
        h, w = int(hs[i]), int(ws[i])
        oh, ow = int(ohs[i]), int(ows[i])
        yield ins[i][:h, :w], outs[i][:oh, :ow]


def _sess(mod_build, task):
    m = mod_build(task)
    s = sanitize_model(m)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(s.SerializeToString(), so, providers=['CPUExecutionProvider'])


def _run(s, x):
    return (s.run(['output'], {'input': x})[0] > 0.0).astype(np.float32)


def fresh_check(task_num: int, candidate=None, n: int = 1500, use_cache: bool = False) -> tuple:
    """Generate N fresh arc instances; report (passes, runs) for the subject net
    (candidate if given, else incumbent). Also prints incumbent/candidate fail counts."""
    mapping = json.load(open(STATE / "arc_mapping.json"))
    arc = mapping[str(task_num)]['arc_id']
    gen = importlib.import_module(f'tasks.task_{arc}')

    task = load_task(task_num)
    inc_mod = importlib.import_module(f'src.custom.task{task_num:03d}')
    importlib.reload(inc_mod)
    inc = _sess(inc_mod.build, task)

    cand = None
    if candidate:
        candidate = str(candidate)
        import importlib.util as _iu
        if candidate.endswith('.py'):
            # exec as if inside src.custom package (for relative ._exact import)
            spec = _iu.spec_from_file_location('src.custom._candmod', candidate)
            cm = _iu.module_from_spec(spec)
            cm.__package__ = 'src.custom'
            sys.modules['src.custom._candmod'] = cm
            spec.loader.exec_module(cm)
        else:
            cm = importlib.import_module(candidate)
            importlib.reload(cm)
        cand = _sess(cm.build, task)

    def _examples():
        if use_cache:
            yield from _cached_examples(task_num, n)
            return
        for _ in range(n):
            try:
                ex = gen.generate()
                yield ex["input"], ex["output"]
            except Exception:
                continue

    inc_fail = cand_fail = cand_vs_inc = 0
    n_ok = 0
    for gi, go in _examples():
        x = to_onehot(gi)
        if x is None:
            continue
        y = to_onehot(go)
        if y is None:
            continue
        ybool = (y > 0.0).astype(np.float32)
        n_ok += 1
        # inference errors count as fails (some incumbents hard-error ORT on rare inputs, task101)
        try:
            oi = _run(inc, x)
        except Exception:
            oi = None
            inc_fail += 1
        if oi is not None and not np.array_equal(oi, ybool):
            inc_fail += 1
        if cand is not None:
            try:
                oc = _run(cand, x)
            except Exception:
                oc = None
                cand_fail += 1
            if oc is not None and not np.array_equal(oc, ybool):
                cand_fail += 1
            if (oc is None) != (oi is None) or (oc is not None and oi is not None and not np.array_equal(oc, oi)):
                cand_vs_inc += 1

    print(f"task{task_num} arc={arc} fresh_instances={n_ok}/{n}")
    print(f"  incumbent fail = {inc_fail}")
    subject_fail = inc_fail
    if cand is not None:
        print(f"  candidate fail = {cand_fail}")
        print(f"  candidate != incumbent = {cand_vs_inc}")
        subject_fail = cand_fail
    return n_ok - subject_fail, n_ok
