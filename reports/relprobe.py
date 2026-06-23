"""Programmatic input->output relationship probe.

For each task, generate fresh examples and test whether the input->output map is a
SIMPLE closed form (identity / fixed D4 / per-pixel color LUT / crop / tile / scale).
Cross-reference with net memory to find golf targets: simple rule + bloated net.
"""
import json, sys, importlib.util
import numpy as np

sys.path.insert(0, "/tmp/arc-gen")
MAPPING = json.load(open("reports/arc_mapping.json"))
MAN = json.load(open("reports/manifest.json"))["tasks"]

def load_gen(num):
    path = MAPPING[str(num)]["generator"]
    spec = importlib.util.spec_from_file_location(f"gen{num}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def d4_variants(a):
    outs = {}
    outs["id"] = a
    outs["rot90"] = np.rot90(a, 1)
    outs["rot180"] = np.rot90(a, 2)
    outs["rot270"] = np.rot90(a, 3)
    outs["flipud"] = np.flipud(a)
    outs["fliplr"] = np.fliplr(a)
    outs["transpose"] = a.T
    outs["antitranspose"] = np.rot90(a, 1)[::-1]  # anti-diagonal
    return outs

def classify(num, n=24):
    try:
        gen = load_gen(num)
    except Exception as e:
        return ("nogen", str(e)[:40])
    exs = []
    tries = 0
    while len(exs) < n and tries < n * 5:
        tries += 1
        try:
            ex = gen.generate()
        except Exception:
            continue
        i = np.array(ex["input"], dtype=np.int64)
        o = np.array(ex["output"], dtype=np.int64)
        if max(i.shape + o.shape) > 30:
            continue
        exs.append((i, o))
    if len(exs) < 8:
        return ("few", f"{len(exs)} ex")

    # shape relationship
    same_shape = all(i.shape == o.shape for i, o in exs)
    if not same_shape:
        # crop? output smaller. tile/scale? output larger by integer factor.
        ratios = set()
        for i, o in exs:
            if i.shape[0] and i.shape[1]:
                ratios.add((o.shape[0] / i.shape[0], o.shape[1] / i.shape[1]))
        intscale = all(abs(r0 - round(r0)) < 1e-9 and abs(r1 - round(r1)) < 1e-9 and r0 >= 1 and r1 >= 1 for r0, r1 in ratios)
        if intscale and len(ratios) <= 3:
            return ("scale?", f"ratios={ratios}")
        smaller = all(o.shape[0] <= i.shape[0] and o.shape[1] <= i.shape[1] for i, o in exs)
        if smaller:
            return ("crop?", "out<=in shape")
        return ("shape", f"varied shapes")

    # identity
    if all(np.array_equal(i, o) for i, o in exs):
        return ("identity", "")

    # fixed D4 (same transform across all)
    for name in ["rot90", "rot180", "rot270", "flipud", "fliplr", "transpose", "antitranspose"]:
        ok = True
        for i, o in exs:
            v = d4_variants(i)[name]
            if v.shape != o.shape or not np.array_equal(v, o):
                ok = False; break
        if ok:
            return ("D4:" + name, "")

    # per-pixel color LUT (consistent map color->color across all pixels & examples)
    lut = {}
    consistent = True
    for i, o in exs:
        for a, b in zip(i.ravel(), o.ravel()):
            a, b = int(a), int(b)
            if a in lut:
                if lut[a] != b:
                    consistent = False; break
            else:
                lut[a] = b
        if not consistent:
            break
    if consistent and any(k != v for k, v in lut.items()):
        return ("colorLUT", f"{lut}")

    # otherwise: same shape, measure pixel diff fraction
    diffs = []
    for i, o in exs:
        diffs.append(float((i != o).mean()))
    return ("diff", f"meandiff={np.mean(diffs):.3f}")

def main():
    targets = [int(t) for t in MAN.keys()]
    results = {}
    for num in sorted(targets):
        mem = MAN[str(num)]["memory"]
        try:
            cls, info = classify(num)
        except Exception as e:
            cls, info = "err", str(e)[:40]
        results[num] = (cls, mem, info)
    # print simple-relationship tasks with notable memory, sorted by memory desc
    SIMPLE = {"identity", "colorLUT"}
    simple_pref = lambda c: c in SIMPLE or c.startswith("D4:") or c in {"scale?", "crop?"}
    rows = [(mem, num, cls, info) for num, (cls, mem, info) in results.items() if simple_pref(cls)]
    rows.sort(reverse=True)
    print("=== SIMPLE-RELATIONSHIP TASKS (by net memory desc) ===")
    for mem, num, cls, info in rows:
        print(f"{mem:>8}  task{num:03d}  {cls:14} {info[:60]}")
    json.dump({str(k): v for k, v in results.items()}, open("reports/relprobe.json", "w"))
    print(f"\nwrote reports/relprobe.json ({len(results)} tasks)")

if __name__ == "__main__":
    main()
