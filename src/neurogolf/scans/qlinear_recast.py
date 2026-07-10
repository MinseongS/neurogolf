"""QLinearConv-recast lever (detection-floor attack, 2026-07-10 R&D).

A fp32 detection `Conv -> fp32 [1,Cout,H,W] (counted 4B/elem) -> Cast->uint8` can be
replaced, BIT-IDENTICALLY when ARC colours 0-9 quantize losslessly (scale=1, zp=0) and
the existing Cast already proves the result fits uint8, by:

    QuantizeLinear(free input) -> QLinearConv(s) -> uint8   (one shared quantized input)

CORRECT per-net byte model (the win is per-net, NOT per-conv): you cannot isolate a single
input channel without a counted fp32 Slice, so the honest recast quantizes the WHOLE input
ONCE (IHW * Cin_total bytes, uint8) and all convs share it. Then each fp32 conv output
(Cout*OHW*4) becomes uint8 (Cout*OHW*1).

    net_win = Σ_conv(Cout*OHW*4)  -  [ IHW*Cin_total  +  Σ_conv(Cout*OHW*1) ]

    => WINS iff  Σ_conv(Cout*OHW*3)  >  IHW*Cin_total.

For a 30x30x10 input, IHW*Cin_total = 9000, so a net needs >3000 elements of fp32 conv
output to win — only a multi-channel conv BANK (Cout>=4) reaches that. Single-channel
detection convs (Cout=1) never do.

Board sweep 2026-07-10: **0 deployed nets pass** — heaviest fp32-conv net is task233 at
6736B (< 9000B shared quant). Multi-channel banks were already converted to QLinearConv
(task367 v_main was the last). This scanner is the LIVE reopen-trigger detector: re-run
after any new Conv net (public dump / rewrite) — a positive row is a buildable bit-identical
win. See memory neurogolf-detection-floor-costmodel-proof.
"""
import numpy as np

import onnx
from onnx import shape_inference

from neurogolf.paths import OVERFIT_NETS

NETS = OVERFIT_NETS
FP32 = onnx.TensorProto.FLOAT


def scan(task_num: int):
    path = NETS / f"task{task_num:03d}.onnx"
    if not path.exists():
        return None
    m = onnx.load(str(path))
    try:
        g = shape_inference.infer_shapes(m, strict_mode=True).graph
    except Exception:
        g = m.graph
    innames = {i.name for i in g.input}
    free = innames | {o.name for o in g.output}
    vi = {}
    for v in list(g.value_info) + list(g.input):
        tt = v.type.tensor_type
        if not tt.HasField("shape"):
            continue
        dims, ok = [], True
        for d in tt.shape.dim:
            if not d.HasField("dim_value") or d.dim_value <= 0:
                ok = False; break
            dims.append(d.dim_value)
        if ok:
            vi[v.name] = (tt.elem_type, dims)
    fp32_conv = 0
    u8_out = 0
    convs = []
    for n in g.node:
        if n.op_type != "Conv":
            continue
        out = n.output[0]
        if out in free:
            continue
        et, dims = vi.get(out, (None, None))
        if et != FP32 or not dims:
            continue
        ne = int(np.prod(dims))
        fp32_conv += ne * 4
        u8_out += ne
        convs.append(out)
    if not convs:
        return None
    inp = next((vi[x] for x in innames if x in vi), None)
    if inp is None:
        return None
    quant = int(np.prod(inp[1]))  # uint8 full-input quantize, shared
    win = fp32_conv - (quant + u8_out)
    if win <= 200:
        return None
    return {"task": task_num, "nconv": len(convs), "fp32_conv_bytes": fp32_conv,
            "shared_quant_bytes": quant, "uint8_out_bytes": u8_out, "win": win,
            "convs": convs}


def scan_all(tasks: list[int] | None = None) -> dict:
    task_range = tasks if tasks else range(1, 401)
    items = [r for t in task_range if (r := scan(t))]
    items.sort(key=lambda x: -x["win"])
    return {"items": items}
