"""Tier-S hunt: find tasks whose output is a FIXED content-independent transform of input
(=> single Gather/Transpose/Tile into the free output, mem ~0, 18-25 pts). The 7500 lever.

For each task, generate N instances and test a library of fixed transforms; a task is a Tier-S
candidate if ONE transform reproduces the output EXACTLY on all N instances. Reports candidates
with current points (headroom). Run: PYTHONPATH=. .venv/bin/python reports/tier_s_scan.py
"""
import json, sys
import numpy as np
sys.path.insert(0, "/tmp/arc-gen")
import importlib.util

MAP = json.load(open("reports/arc_mapping.json"))
MAN = json.load(open("reports/manifest.json"))["tasks"]

def load_gen(num):
    path = MAP[str(num)]["generator"]
    spec = importlib.util.spec_from_file_location(f"gen{num}", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def tile(a, ry, rx): return np.tile(a, (ry, rx))
def kron_self(a):  # fractal: each cell -> full grid if nonzero (kron(a>0, a))
    return np.kron((a > 0).astype(a.dtype), a)

# transform library: name -> fn(input_grid ndarray) -> predicted output ndarray
TFS = {
    "identity": lambda a: a,
    "fliplr": np.fliplr, "flipud": np.flipud,
    "rot90": lambda a: np.rot90(a, 1), "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3),
    "transpose": lambda a: a.T, "anti_transpose": lambda a: np.rot90(a, 2).T,
    "tile2x2": lambda a: tile(a, 2, 2), "tile3x3": lambda a: tile(a, 3, 3),
    "tile1x2": lambda a: tile(a, 1, 2), "tile2x1": lambda a: tile(a, 2, 1),
    "tile1x3": lambda a: tile(a, 1, 3), "tile3x1": lambda a: tile(a, 3, 1),
    "kron_self": kron_self,
    # mirror-concat (common ARC): output = [a | fliplr(a)] etc.
    "concat_lr_mirror": lambda a: np.concatenate([a, np.fliplr(a)], axis=1),
    "concat_ud_mirror": lambda a: np.concatenate([a, np.flipud(a)], axis=0),
    "concat_lr": lambda a: np.concatenate([a, a], axis=1),
    "concat_ud": lambda a: np.concatenate([a, a], axis=0),
}

def scan_task(num, N=4):
    try:
        gen = load_gen(num)
    except Exception:
        return None
    inst = []
    tries = 0
    while len(inst) < N and tries < N * 6:
        tries += 1
        try:
            ex = gen.generate()
        except Exception:
            continue
        a = np.array(ex["input"]); o = np.array(ex["output"])
        if max(a.shape) > 30 or max(o.shape) > 30:
            continue
        inst.append((a, o))
    if len(inst) < 2:
        return None
    hits = []
    for name, fn in TFS.items():
        ok = True
        for a, o in inst:
            try:
                p = fn(a)
            except Exception:
                ok = False; break
            if p.shape != o.shape or not np.array_equal(p, o):
                ok = False; break
        if ok:
            hits.append(name)
    return hits

def main():
    cands = []
    for num in range(1, 401):
        hits = scan_task(num)
        if hits:
            pts = MAN[str(num)]["points"]
            cands.append((num, pts, MAN[str(num)].get("method", "?"), hits))
    cands.sort(key=lambda t: t[1])  # lowest points = most headroom
    print(f"TIER-S CANDIDATES (output = fixed transform of input): {len(cands)}")
    print(f"{'task':>4} {'pts':>6} {'method':22} transforms")
    for num, pts, meth, hits in cands:
        flag = "  <-- HEADROOM" if pts < 18 else ""
        print(f"{num:4d} {pts:6.2f} {meth[:22]:22} {','.join(hits)}{flag}")
    json.dump([{"task": n, "points": p, "method": m, "transforms": h} for n, p, m, h in cands],
              open("reports/tier_s_candidates.json", "w"), indent=1)
    head = [c for c in cands if c[1] < 18]
    print(f"\n{len(head)} with headroom (<18pt). Re-encoding each Tier-S (single Gather, mem~0) "
          f"=> ~18-25pt. Est +{sum(min(20,25)-c[1] for c in head):.0f} if all reach ~20.")

if __name__ == "__main__":
    main()
