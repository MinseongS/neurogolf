"""residual_spatialop oracle (2026-07-08, insight `residual_spatialop_to_free_einsum_collapse`).

INDEPENDENT floor probe (NOT self-referential): for each of OUR active overfit nets, sum the counted
grader bytes produced by residual SPATIAL ops (Conv/GridSample/TopK/MaxPool/GatherND/GatherElements)
plus WIDE index/one-hot planes. High score => the net still materializes planes the public frontier
collapses into free-input Einsum contractions => hand-rebuild candidate. Low score => genuinely near an
Einsum floor. This is the oracle the reformed SKILL requires before writing any "floor" verdict.

Usage: PYTHONPATH=. uv run python reports/scripts/residual_spatialop_scan.py [netdir]
       (default netdir = submission/overfit_nets)
"""
import glob, os, sys, collections
import numpy as np
import onnx
from onnx import shape_inference

DT = {1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2}
SPATIAL = {"Conv", "ConvTranspose", "GridSample", "TopK", "MaxPool", "AveragePool",
           "GatherND", "GatherElements", "ConvInteger", "QLinearConv"}
WIDE_MIN = 900          # a single intermediate >= this many bytes is a "wide plane"
NETDIR = sys.argv[1] if len(sys.argv) > 1 else "submission/overfit_nets"


def bytes_of(et, dims):
    if not dims or any(d == 0 for d in dims):
        return 0
    c = 1
    for d in dims:
        c *= d
    return c * DT.get(et, 4)


def scan(p):
    try:
        m = shape_inference.infer_shapes(onnx.load(p))
    except Exception:
        return None
    g = m.graph
    vi = {}
    for v in list(g.value_info) + list(g.output) + list(g.input):
        dims = tuple(d.dim_value if d.HasField("dim_value") else 0 for d in v.type.tensor_type.shape.dim)
        vi[v.name] = (v.type.tensor_type.elem_type, dims)
    spatial_bytes = 0
    spatial_ops = collections.Counter()
    wide_bytes = 0
    total_mem = 0
    for n in g.node:
        for o in n.output:
            if o in ("input", "output"):
                continue
            et, dims = vi.get(o, (None, None))
            b = bytes_of(et, dims)
            total_mem += b
            if n.op_type in SPATIAL:
                spatial_bytes += b
                if b:
                    spatial_ops[n.op_type] += 1
            if b >= WIDE_MIN:
                wide_bytes += b
    params = sum(int(np.prod(i.dims)) for i in g.initializer)
    return dict(mem=total_mem, params=params, spatial_bytes=spatial_bytes,
                wide_bytes=wide_bytes, spatial_ops=dict(spatial_ops))


rows = []
for p in sorted(glob.glob(f"{NETDIR}/task*.onnx")):
    t = int(os.path.basename(p)[4:7])
    r = scan(p)
    if r is None:
        continue
    # headroom proxy: bytes tied up in collapsible spatial ops + wide planes, on non-trivial nets
    headroom = r["spatial_bytes"] + 0.3 * r["wide_bytes"]
    if headroom > 0:
        rows.append((headroom, t, r))

rows.sort(reverse=True)
print(f"{'rank':>4} {'task':>4} {'mem':>7} {'par':>6} {'spatialB':>9} {'wideB':>7}  spatial_ops")
for i, (h, t, r) in enumerate(rows[:40], 1):
    print(f"{i:4d} {t:4d} {r['mem']:7d} {r['params']:6d} {r['spatial_bytes']:9d} {r['wide_bytes']:7d}  {r['spatial_ops']}")
print(f"\n{len(rows)} nets carry residual spatial-op / wide-plane bytes (collapse candidates).")
print(f"Total spatial_bytes across all: {sum(r['spatial_bytes'] for _,_,r in rows)}")
