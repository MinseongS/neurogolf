"""Task 205 (confettibox / ARC 8731374e).

Rule: a solid wide x tall rectangle of `boxcolor` sits in a noise grid, with
1-3 interior pixels of a second `color`.  Output = the box (wide x tall, aligned
top-left), with each special pixel's whole row and whole column flooded with
`color`:
    out[i][j] = color  if (i is a cross-row OR j is a cross-col) else boxcolor
cells outside the wide x tall box are off (all channels false).

Memory floor-break.  The old net materialised an `inp16` [1,10,30,30] cast
(18KB) plus ~24 fp16 [1,1,30,30] planes and a 9KB `out_b` -> 106354 bytes.

Two changes:
  * Convs read the fp32 `input` directly (fp32 weights), so the 18KB 10-channel
    cast is gone.
  * The whole tail is replaced by a SEPARABLE uint8 label map.  Once the box is
    detected, the per-cell output is separable: it depends only on a 1-D
    cross-row vector cr[i], a 1-D cross-col vector cc[j], the corner rectangle
    (i<h, j<w), and the two scalars boxcolor/color.  We build a single uint8
    label L[1,1,30,30] = corner ? ((cr[i]|cc[j]) ? color : boxcolor) : 10 and
    emit `output = Equal(L, arange[1,10,1,1])` (opset 11, BOOL output).  No
    10-channel intermediate is ever materialised.

Box detection (unchanged in spirit, kept fp16 and reduced to bool early): the
solid box is the region whose cells equal both a length-L=6 horizontal AND a
length-6 vertical neighbour-run (the box side is always >=6; a chance 6x6 noise
block is negligible).  Outside-grid cells get a unique negative sentinel so no
equal-run ever forms there.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
I32 = TensorProto.INT32
L = 6  # box side is always >= 6, so a length-6 run is box-only


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # shared scalars / index planes
    init("Half", np.array(0.5, np.float16), np.float16)
    init("One", np.array(1.0, np.float16), np.float16)
    init("C30", np.array(30.0, np.float16), np.float16)
    init("NegHalf", np.array(-0.5, np.float16), np.float16)
    init("Icol", np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1), np.float16)
    init("IcolH", (30.0 - np.arange(30)).astype(np.float16).reshape(1, 1, 30, 1), np.float16)
    init("Irow", np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30), np.float16)
    init("IrowH", (30.0 - np.arange(30)).astype(np.float16).reshape(1, 1, 1, 30), np.float16)

    # ---- integer grid G (fp16) and grid mask gm (read fp32 input directly) ----
    Wg = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("Wg", Wg, np.float32)
    n("Conv", ["input", "Wg"], "Gf")                # [1,1,30,30] f32 colour
    n("Cast", ["Gf"], "G", to=F16)
    Wm = np.ones((1, 10, 1, 1), np.float32)
    init("Wm", Wm, np.float32)
    n("Conv", ["input", "Wm"], "gm_ff")             # f32 1 inside grid
    n("Cast", ["gm_ff"], "gm_f", to=F16)
    n("Greater", ["gm_f", "Half"], "gm_b")          # bool inside grid
    # Gms: G inside grid, unique negative sentinel outside (kills exterior runs;
    # interior background-0 blocks of 6x6 are ~1e-35 so only the exterior matters)
    init("IDX", (np.arange(900).reshape(1, 1, 30, 30) + 1).astype(np.float16), np.float16)
    n("Sub", ["gm_f", "One"], "gm1")                # 0 inside, -1 outside
    n("Mul", ["gm1", "IDX"], "Gm")                  # 0 inside, -idx outside
    n("Add", ["G", "Gm"], "Gms")

    # ---- equal-to-neighbour maps via neighbour-difference Conv (==0) ----
    init("Wdh", np.array([1.0, -1.0], np.float16).reshape(1, 1, 1, 2), np.float16)
    n("Conv", ["Gms", "Wdh"], "dh")
    n("Abs", ["dh"], "dha"); n("Greater", ["dha", "Half"], "hne_b"); n("Not", ["hne_b"], "Eh_b")
    n("Cast", ["Eh_b"], "Eh", to=F16)
    init("Wdv", np.array([1.0, -1.0], np.float16).reshape(1, 1, 2, 1), np.float16)
    n("Conv", ["Gms", "Wdv"], "dv")
    n("Abs", ["dv"], "dva"); n("Greater", ["dva", "Half"], "vne_b"); n("Not", ["vne_b"], "Ev_b")
    n("Cast", ["Ev_b"], "Ev", to=F16)

    # ---- runs of L-1 consecutive equals, then dilate to cover L cells ----
    Lm = L - 1
    init("WhR", np.ones((1, 1, 1, Lm), np.float16), np.float16)
    n("Conv", ["Eh", "WhR"], "hrun")
    init("LmThr", np.array(Lm - 0.5, np.float16), np.float16)
    n("Greater", ["hrun", "LmThr"], "hs_b"); n("Cast", ["hs_b"], "hs", to=F16)
    init("WhD", np.ones((1, 1, 1, L), np.float16), np.float16)
    n("Conv", ["hs", "WhD"], "hcov_s", pads=[0, L - 1, 0, L - 1])
    n("Greater", ["hcov_s", "Half"], "hcov_b")
    init("WvR", np.ones((1, 1, Lm, 1), np.float16), np.float16)
    n("Conv", ["Ev", "WvR"], "vrun")
    n("Greater", ["vrun", "LmThr"], "vs_b"); n("Cast", ["vs_b"], "vs", to=F16)
    init("WvD", np.ones((1, 1, L, 1), np.float16), np.float16)
    n("Conv", ["vs", "WvD"], "vcov_s", pads=[L - 1, 0, L - 1, 0])
    n("Greater", ["vcov_s", "Half"], "vcov_b")
    n("And", ["hcov_b", "vcov_b"], "solid_hv")
    n("And", ["solid_hv", "gm_b"], "solid_b")
    n("Cast", ["solid_b"], "solid", to=F16)         # [1,1,30,30]

    # ---- box bounds r0,r1,c0,c1 (scalars) ----
    n("ReduceMax", ["solid"], "rowocc", axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["solid"], "colocc", axes=[2], keepdims=1)  # [1,1,1,30]
    n("Mul", ["rowocc", "IcolH"], "r0a"); n("ReduceMax", ["r0a"], "r0m", keepdims=0)
    n("Sub", ["C30", "r0m"], "r0")
    n("Mul", ["rowocc", "Icol"], "r1a"); n("ReduceMax", ["r1a"], "r1", keepdims=0)
    n("Mul", ["colocc", "IrowH"], "c0a"); n("ReduceMax", ["c0a"], "c0m", keepdims=0)
    n("Sub", ["C30", "c0m"], "c0")
    n("Mul", ["colocc", "Irow"], "c1a"); n("ReduceMax", ["c1a"], "c1", keepdims=0)

    # ---- inbox plane (absolute coords) just to locate the cross cells/colour ----
    n("Sub", ["Icol", "r0"], "ir_lo"); n("Greater", ["ir_lo", "NegHalf"], "ir1")
    n("Sub", ["r1", "Icol"], "ir_hi"); n("Greater", ["ir_hi", "NegHalf"], "ir2")
    n("And", ["ir1", "ir2"], "inr_b")               # [1,1,30,1]
    n("Sub", ["Irow", "c0"], "ic_lo"); n("Greater", ["ic_lo", "NegHalf"], "ic1")
    n("Sub", ["c1", "Irow"], "ic_hi"); n("Greater", ["ic_hi", "NegHalf"], "ic2")
    n("And", ["ic1", "ic2"], "inc_b")               # [1,1,1,30]
    n("And", ["inr_b", "inc_b"], "inbox_b")

    # ---- boxcolor / color scalars ----
    n("Mul", ["G", "solid"], "Gs"); n("ReduceMax", ["Gs"], "boxcolor", keepdims=1)  # [1,1,1,1]
    n("Sub", ["G", "boxcolor"], "gd"); n("Abs", ["gd"], "gda")
    n("Greater", ["gda", "Half"], "notbp_b")        # NOT boxcolor
    n("And", ["inbox_b", "notbp_b"], "crosscell_b")
    n("Cast", ["crosscell_b"], "crosscell", to=F16)
    n("Mul", ["G", "crosscell"], "Gc"); n("ReduceMax", ["Gc"], "color", keepdims=1)

    # 1-D cross-row / cross-col occupancy (absolute), then shift to corner
    n("ReduceMax", ["crosscell"], "crossrow", axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["crosscell"], "crosscol", axes=[2], keepdims=1)  # [1,1,1,30]

    # shift matrices Sr/Pc move absolute-coord occupancy to corner coords
    Dr = (np.arange(30)[None, :] - np.arange(30)[:, None]).astype(np.int32)  # k-i
    Dc = (np.arange(30)[:, None] - np.arange(30)[None, :]).astype(np.int32)  # a-j
    init("Dr_i", Dr, np.int32)
    init("Dc_i", Dc, np.int32)
    n("Cast", ["r0"], "r0_i", to=I32)
    n("Cast", ["c0"], "c0_i", to=I32)
    n("Equal", ["Dr_i", "r0_i"], "Sr_b"); n("Cast", ["Sr_b"], "Sr", to=F16)  # [30,30]
    n("Equal", ["Dc_i", "c0_i"], "Pc_b"); n("Cast", ["Pc_b"], "Pc", to=F16)
    n("MatMul", ["Sr", "crossrow"], "crossrow_c")   # [1,1,30,1] corner-coord 1-D
    n("MatMul", ["crosscol", "Pc"], "crosscol_c")   # [1,1,1,30]
    n("Greater", ["crossrow_c", "Half"], "cr_row_b")   # [1,1,30,1] bool cross-row
    n("Greater", ["crosscol_c", "Half"], "cc_col_b")   # [1,1,1,30] bool cross-col

    # corner rectangle: rows < h, cols < w  (separable 1-D bounds)
    n("Sub", ["r1", "r0"], "hm1"); n("Add", ["hm1", "One"], "h")
    n("Sub", ["c1", "c0"], "wm1"); n("Add", ["wm1", "One"], "w")
    n("Sub", ["h", "Icol"], "h_gt"); n("Greater", ["h_gt", "Half"], "rin_b")  # [1,1,30,1]
    n("Sub", ["w", "Irow"], "w_gt"); n("Greater", ["w_gt", "Half"], "cin_b")  # [1,1,1,30]

    # ---- separable uint8 label map ----
    # boxcolor / color as uint8 scalars (broadcast)
    n("Cast", ["boxcolor"], "boxc_u", to=TensorProto.UINT8)   # [1,1,1,1]
    n("Cast", ["color"], "color_u", to=TensorProto.UINT8)
    init("v10u", np.array(10, np.uint8), np.uint8)            # off-box sentinel
    # cross = cross-row[i] OR cross-col[j]  (broadcast [1,1,30,1] | [1,1,1,30])
    n("Or", ["cr_row_b", "cc_col_b"], "cross_b")             # [1,1,30,30]
    # in-corner = rin[i] AND cin[j]
    n("And", ["rin_b", "cin_b"], "corner_b")                 # [1,1,30,30]
    n("Where", ["cross_b", "color_u", "boxc_u"], "Lin")      # colour vs box
    n("Where", ["corner_b", "Lin", "v10u"], "L")            # off-box -> 10
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                      # -> BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
