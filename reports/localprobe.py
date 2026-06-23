"""Test same-shape 'diff' tasks for a FIXED LOCAL RULE:
out[r,c] is a consistent function of the KxK input neighborhood around (r,c).
If consistent across many fresh examples, the task is a fixed-conv golf target.
"""
import json, sys, importlib.util
import numpy as np
sys.path.insert(0, "/tmp/arc-gen")
MAPPING = json.load(open("reports/arc_mapping.json"))
MAN = json.load(open("reports/manifest.json"))["tasks"]
REL = json.load(open("reports/relprobe.json"))

def load_gen(num):
    path = MAPPING[str(num)]["generator"]
    spec = importlib.util.spec_from_file_location(f"gen{num}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def fresh(num, n=20):
    gen = load_gen(num)
    exs = []; t = 0
    while len(exs) < n and t < n * 6:
        t += 1
        try: ex = gen.generate()
        except Exception: continue
        i = np.array(ex["input"], np.int64); o = np.array(ex["output"], np.int64)
        if i.shape != o.shape or max(i.shape) > 30: continue
        exs.append((i, o))
    return exs

def local_consistency(num, K=3):
    """Return fraction of neighborhoods whose -> output is consistent, and #conflicts."""
    exs = fresh(num)
    if len(exs) < 6: return None
    pad = K // 2
    table = {}
    conflicts = 0; total = 0
    for i, o in exs:
        ip = np.pad(i, pad, constant_values=-1)  # -1 = off-grid sentinel
        H, W = i.shape
        for r in range(H):
            for c in range(W):
                patch = ip[r:r+K, c:c+K].tobytes()
                out = int(o[r, c])
                total += 1
                if patch in table:
                    if table[patch] != out: conflicts += 1
                else:
                    table[patch] = out
    return conflicts, total

def main():
    # same-shape diff tasks with mem >= 1500
    cands = []
    for k, v in REL.items():
        if v[0] == "diff" and MAN[k]["memory"] >= 1500:
            cands.append((MAN[k]["memory"], int(k)))
    cands.sort(reverse=True)
    print("=== LOCAL-RULE test (3x3) on same-shape diff tasks, mem>=1500 ===")
    print("mem      task    conflicts/total   verdict")
    for mem, num in cands:
        try:
            res = local_consistency(num, 3)
        except Exception as e:
            print(f"{mem:>7} task{num:03d}  ERR {str(e)[:30]}"); continue
        if res is None:
            print(f"{mem:>7} task{num:03d}  (few examples)"); continue
        conf, tot = res
        frac = conf / tot if tot else 1
        verdict = "*** LOCAL-3x3 EXACT ***" if conf == 0 else ("near" if frac < 0.005 else "no")
        print(f"{mem:>7} task{num:03d}  {conf:>5}/{tot:<7} {frac:.4f}  {verdict}")

if __name__ == "__main__":
    main()
