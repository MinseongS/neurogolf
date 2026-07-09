"""Regime-vein recon: find DEPLOYED nets dominated by a large >=20x20 node-output plane
(the output-welded routing mask = the 900B floor). These are regime-crack candidates
(fold the routing into a free-output Einsum; see memory neurogolf-regime-crack-freeoutput-einsum).

PRE-FILTER when picking targets: skip POSITIONED-CONTENT masks (arbitrary content at a
data-dependent position -> floor, e.g. task112/163/099); take global-state / structured
(ring/block/periodic/diagonal/threshold-run/count).

Output ranked by bytes*fraction.
"""
import numpy as np

import onnx
from onnx import shape_inference

from neurogolf.paths import OVERFIT_NETS

NETS = OVERFIT_NETS

ITEMSIZE = {onnx.TensorProto.FLOAT: 4, onnx.TensorProto.FLOAT16: 2,
            onnx.TensorProto.UINT8: 1, onnx.TensorProto.INT8: 1,
            onnx.TensorProto.BOOL: 1, onnx.TensorProto.INT32: 4,
            onnx.TensorProto.INT64: 8, onnx.TensorProto.DOUBLE: 8}
TNAME = {v: k for k, v in onnx.TensorProto.DataType.items()}


def scan(task_num):
    path = NETS / f"task{task_num:03d}.onnx"
    if not path.exists():
        return None
    m = onnx.load(str(path))
    try:
        g = shape_inference.infer_shapes(m, strict_mode=True).graph
    except Exception:
        return None
    inits = {i.name for i in g.initializer}
    innames = {i.name for i in g.input}
    node_outs = {o for n in g.node for o in n.output if o}
    vi = {v.name: v for v in list(g.value_info) + list(g.output)}
    prod = {o: n.op_type for n in g.node for o in n.output if o}
    total = 0
    biggest = None
    for name, v in vi.items():
        if name in inits or name in innames or name == "output" or name not in node_outs:
            continue
        tt = v.type.tensor_type
        if not tt.HasField("shape"):
            continue
        dims, ok = [], True
        for d in tt.shape.dim:
            if not d.HasField("dim_value") or d.dim_value <= 0:
                ok = False; break
            dims.append(d.dim_value)
        if not ok:
            continue
        nb = int(np.prod(dims)) * ITEMSIZE.get(tt.elem_type, 4)
        total += nb
        spatial = len(dims) >= 2 and dims[-1] >= 20 and dims[-2] >= 20
        if spatial and (biggest is None or nb > biggest[0]):
            biggest = (nb, name, dims, TNAME.get(tt.elem_type, "?"), prod.get(name, "?"))
    if not biggest or total == 0:
        return None
    frac = biggest[0] / total
    return {"task": task_num, "mask_bytes": biggest[0], "mask_name": biggest[1],
            "mask_dims": biggest[2], "mask_dtype": biggest[3], "producer": biggest[4],
            "total_mem": total, "frac": round(frac, 3), "score": round(biggest[0] * frac, 1)}


def scan_all(tasks: list[int] | None = None, min_frac: float = 0.30) -> dict:
    task_range = tasks if tasks else range(1, 401)
    rows = [r for t in task_range if (r := scan(t)) and r["frac"] >= min_frac]
    rows.sort(key=lambda x: -x["score"])
    items = []
    for r in rows:
        # expected_gain = saved_bytes / cost (ranking only; gate decides real points)
        eg = r["mask_bytes"] / r["total_mem"] if r["total_mem"] else 0.0
        items.append({**r, "expected_gain": round(eg, 4)})
    return {"items": items}
