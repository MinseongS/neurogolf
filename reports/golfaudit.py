"""Helper for auditing whether a task's input->output rule is exactly expressible.

Usage:
    import reports.golfaudit as G
    exs = G.fresh(173, 200)            # list of (input_np, output_np) int grids
    G.verify(173, my_rule_fn, n=300)   # returns (mismatches, total); 0 == EXACT
where my_rule_fn(input_np) -> predicted output_np (int grid, same convention).
"""
import json, sys, importlib.util
import numpy as np
sys.path.insert(0, "/tmp/arc-gen")
MAPPING = json.load(open("reports/arc_mapping.json"))

def load_gen(num):
    path = MAPPING[str(num)]["generator"]
    spec = importlib.util.spec_from_file_location(f"gen{num}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

def fresh(num, n=200, maxdim=30):
    gen = load_gen(num)
    exs = []; t = 0
    while len(exs) < n and t < n * 8:
        t += 1
        try:
            ex = gen.generate()
        except Exception:
            continue
        i = np.array(ex["input"], np.int64); o = np.array(ex["output"], np.int64)
        if max(i.shape + o.shape) > maxdim:
            continue
        exs.append((i, o))
    return exs

def verify(num, fn, n=300):
    exs = fresh(num, n)
    bad = 0
    for i, o in exs:
        try:
            p = fn(i)
        except Exception:
            bad += 1; continue
        p = np.asarray(p)
        if p.shape != o.shape or not np.array_equal(p, o):
            bad += 1
    return bad, len(exs)
