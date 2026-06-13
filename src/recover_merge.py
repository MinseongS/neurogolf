"""Rebuild the submission gated on GENERALIZATION (fresh arc-gen instances),
not just stored-example local score. For each task, gather candidate nets from
the public-kernel sources + our current network, accept only those that pass
BOTH the stored examples (harness.evaluate ok) AND a batch of freshly generated
arc-gen instances, then keep the highest-scoring generalizing candidate.

This fixes the keep-best bug where exact-match memorizer nets (which score ~0 on
Kaggle's held-out arc-gen) displaced generalizing public/custom nets.
"""
import importlib.util, json, sys, os, multiprocessing, shutil
import numpy as np
import onnxruntime as ort
from src.harness import convert_to_numpy, load_task, evaluate

MAPPING = json.load(open("reports/arc_mapping.json"))
sys.path.insert(0, "/tmp/arc-gen")

SOURCES = [
    ("ours", "networks"),
    ("thbdh6332", "/tmp/ng_public2/thbdh6332/unpacked"),
    ("vyank6322", "/tmp/ng_public2/vyank6322/unpacked"),
    ("wguesdon6315", "/tmp/ng_public2/wguesdon6315/unpacked"),
    ("seddik", "/tmp/ng_public2/seddik/submission"),
    ("galaxy", "/tmp/ng_public2/galaxy/unpacked"),
    ("biohack_new", "/tmp/ng_public2/biohack_new/submission"),
]
NFRESH = 60

def load_gen(num):
    path = MAPPING[str(num)]["generator"]
    spec = importlib.util.spec_from_file_location(f"gen{num}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def fresh_ok(path, num, gen, n=NFRESH):
    """True iff net passes n fresh instances (or no generator available)."""
    if gen is None:
        return True  # can't generate -> trust stored eval
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        sess = ort.InferenceSession(path, so)
    except Exception:
        return False
    run = tries = 0
    while run < n and tries < n*5:
        tries += 1
        try:
            ex = gen.generate()
        except Exception:
            continue
        if max(len(ex["input"]), len(ex["input"][0]), len(ex["output"]), len(ex["output"][0])) > 30:
            continue
        bm = convert_to_numpy(ex)
        try:
            out = sess.run(None, {"input": bm["input"].astype(np.float32)})[0]
        except Exception:
            return False
        if not ((out[0] > 0).astype(np.int8) == bm["output"][0].astype(np.int8)).all():
            return False
        run += 1
    return True

def _worker(num):
    task = load_task(num)
    try:
        gen = load_gen(num)
    except Exception:
        gen = None
    best = None  # (points, label, path)
    for label, d in SOURCES:
        p = os.path.join(d, f"task{num:03d}.onnx")
        if not os.path.exists(p):
            continue
        try:
            ev = evaluate(p, task)
        except Exception:
            continue
        if not ev["ok"]:
            continue
        if best is not None and ev["points"] <= best[0]:
            continue  # can't win even before the (expensive) fresh check
        if not fresh_ok(p, num, gen):
            continue
        best = (ev["points"], label, p)
    return num, best

def main():
    manifest = json.load(open("reports/manifest.json"))["tasks"]
    chosen = {}
    total = 0.0
    bylabel = {}
    with multiprocessing.Pool(8, maxtasksperchild=1) as pool:
        for num, best in pool.imap_unordered(_worker, range(1,401)):
            chosen[num] = best
            if best:
                total += best[0]
                bylabel.setdefault(best[1],[0,0.0]); bylabel[best[1]][0]+=1; bylabel[best[1]][1]+=best[0]
    # write the winning nets into networks/
    changed = 0
    new_manifest = {}
    for num in range(1,401):
        best = chosen[num]
        dst = f"networks/task{num:03d}.onnx"
        if best is None:
            new_manifest[str(num)] = None
            continue
        pts, label, src = best
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy(src, dst); changed += 1
        ev = evaluate(dst, load_task(num))
        old = manifest.get(str(num),{})
        meth = old.get("method","?") if label=="ours" else f"gen:{label}"
        new_manifest[str(num)] = {"points":ev["points"],"memory":ev["memory"],"params":ev["params"],"method":meth}
    json.dump({"tasks":new_manifest}, open("reports/manifest.json","w"), indent=1)
    print(f"GENERALIZING submission rebuilt: est real LB ~ {total:.1f} ({changed} nets swapped)")
    for lbl,(n,p) in sorted(bylabel.items(),key=lambda x:-x[1][1]):
        print(f"   {lbl:14s} {n:3d} tasks  {p:.1f} pts")
    unsolved = sum(1 for n in range(1,401) if chosen[n] is None)
    print(f"   unsolved (no generalizing net): {unsolved}")

if __name__=="__main__":
    main()
