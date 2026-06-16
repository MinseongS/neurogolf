"""Task 205 (confettibox / ARC 8731374e).

Rule: a solid wide x tall rectangle of `boxcolor` sits in a noise grid, with
1-3 interior pixels of a second `color`.  Output = the box (wide x tall, aligned
top-left), with each special pixel's whole row and whole column flooded with
`color`:
    out[i][j] = color  if (i is a cross-row OR j is a cross-col) else boxcolor
cells outside the wide x tall box are off (all channels false).

The per-cell output is SEPARABLE once the box is found: it depends only on a
1-D corner cross-row vector cr[i], a 1-D corner cross-col vector cc[j], the
corner rectangle (i<h, j<w), and two scalars boxcolor/color.  We build a single
uint8 label L[1,1,30,30] = corner ? ((cr[i]|cc[j]) ? color : boxcolor) : 10 and
emit `output = Equal(L, arange[1,10,1,1])` (opset 11, BOOL output) -- no
10-channel intermediate is ever materialised.

Box detection (colour-agnostic, run based): the solid box is the region whose
cells equal both a length>=6 horizontal AND a length>=6 vertical neighbour-run.
The box side is always >=6, so a chance 6x6 noise block is negligible.
Off-grid cells get a unique per-cell coordinate value so no spurious equal-run
forms across / outside the grid boundary.

Memory/param shrink vs the previous encoding (69514B / 2882 params):
  * the off-grid sentinel uses a coordinate plane built from two tiny 1-D ramps
    (60 params) instead of a 900-element IDX initializer;
  * the absolute->corner cross-occupancy shift uses a data-dependent Gather with
    a static [30] index vector instead of two 30x30 MatMul shift matrices
    (1800 params -> 0);
  * detection intermediates are kept bool wherever possible.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
I32 = TensorProto.INT32
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
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

    # ---- shared scalars / 1-D index ramps ----
    init("Half", np.array(0.5, np.float16), np.float16)
    init("One", np.array(1.0, np.float16), np.float16)
    init("C30", np.array(30.0, np.float16), np.float16)
    init("NegHalf", np.array(-0.5, np.float16), np.float16)
    init("Icol", np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1), np.float16)
    init("IcolH", (30.0 - np.arange(30)).astype(np.float16).reshape(1, 1, 30, 1), np.float16)
    init("Irow", np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30), np.float16)
    init("IrowH", (30.0 - np.arange(30)).astype(np.float16).reshape(1, 1, 1, 30), np.float16)

    # ---- integer colour grid G (fp16): G = colour_index + 1 in [1..10].
    # The +1 bias (a free Conv bias) makes every in-grid cell value >=1, so a
    # single Where(cross, G, 0) plane later carries BOTH the cross-cell colour
    # AND a positive occupancy marker even when the special colour is 0.
    # Neighbour differences are unaffected (the +1 cancels).  ----
    Wg = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("Wg", Wg, np.float32)
    init("Bias1", np.array([1.0], np.float32), np.float32)
    n("Conv", ["input", "Wg", "Bias1"], "Gf")       # [1,1,30,30] f32 colour+1
    n("Cast", ["Gf"], "G", to=F16)
    # in-grid mask is SEPARABLE: the ARC grid is the top-left [0,H)x[0,W)
    # rectangle and every in-grid cell is one-hot, so per-row/col occupancy is a
    # 1-D reduction straight off the FREE input -- no full occupancy Conv plane.
    n("ReduceMax", ["input"], "rowany_f", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["input"], "colany_f", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    init("Half32", np.array(0.5, np.float32), np.float32)
    n("Greater", ["rowany_f", "Half32"], "rowany_b")  # [1,1,30,1] bool
    n("Greater", ["colany_f", "Half32"], "colany_b")  # [1,1,1,30] bool
    n("And", ["rowany_b", "colany_b"], "gm_b")      # [1,1,30,30] bool in-grid

    # off-grid cells -> a unique per-cell coordinate so no equal-run forms there.
    # coord = -(col*30 + row + 1)  (broadcast of two 1-D ramps, 60 params); the
    # negatives are disjoint from in-grid colours 0..9, so neighbour-equality
    # never fires in / across the exterior.  Gms = Where(in_grid, G, coord).
    init("C30c", (-30.0 * np.arange(30) - 1.0).astype(np.float16).reshape(1, 1, 30, 1), np.float16)
    n("Sub", ["C30c", "Irow"], "coord")             # [1,1,30,30] unique negatives
    n("Where", ["gm_b", "G", "coord"], "Gms")

    # ---- equal-to-neighbour maps via neighbour-difference Conv == 0.
    # Gms values are integers (colours 1..10, sentinels <=-1) exact in fp16, so
    # an exact Equal(diff, 0) needs no Abs/threshold pair. ----
    init("Zero16", np.array(0.0, np.float16), np.float16)
    init("Wdh", np.array([1.0, -1.0], np.float16).reshape(1, 1, 1, 2), np.float16)
    n("Conv", ["Gms", "Wdh"], "dh")
    n("Equal", ["dh", "Zero16"], "Eh_b"); n("Cast", ["Eh_b"], "Eh", to=F16)
    init("Wdv", np.array([1.0, -1.0], np.float16).reshape(1, 1, 2, 1), np.float16)
    n("Conv", ["Gms", "Wdv"], "dv")
    n("Equal", ["dv", "Zero16"], "Ev_b"); n("Cast", ["Ev_b"], "Ev", to=F16)

    # ---- runs of L-1 consecutive equals, then dilate to cover L cells ----
    # A spurious >=6 run can occur in noise (~1e-5 per position), so robust box
    # detection requires the 2-D coincidence solid = hcov AND vcov (cell lies in
    # BOTH a horizontal AND a vertical >=6 run): p ~ 1e-10, negligible.
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
    n("And", ["hcov_b", "vcov_b"], "solid_b")
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

    # ---- separable 1-D in-box bounds (absolute coords) ----
    n("Sub", ["Icol", "r0"], "ir_lo"); n("Greater", ["ir_lo", "NegHalf"], "ir1")
    n("Sub", ["r1", "Icol"], "ir_hi"); n("Greater", ["ir_hi", "NegHalf"], "ir2")
    n("And", ["ir1", "ir2"], "inr_b")               # [1,1,30,1]
    n("Sub", ["Irow", "c0"], "ic_lo"); n("Greater", ["ic_lo", "NegHalf"], "ic1")
    n("Sub", ["c1", "Irow"], "ic_hi"); n("Greater", ["ic_hi", "NegHalf"], "ic2")
    n("And", ["ic1", "ic2"], "inc_b")               # [1,1,1,30]

    # ---- boxcolor: the box top-left corner (r0,c0) is always pure boxcolour
    # (special pixels are strictly interior), so read it with two scalar Gathers
    # -- no full G*solid plane. ----
    n("Cast", ["r0"], "r0i_f", to=I32); n("Reshape", ["r0i_f", "shp1"], "r0i")
    n("Cast", ["c0"], "c0i_f", to=I32); n("Reshape", ["c0i_f", "shp1"], "c0i")
    init("shp1", np.array([1], np.int64), np.int64)
    n("Gather", ["G", "r0i"], "Grow0", axis=2)      # [1,1,1,30]
    n("Gather", ["Grow0", "c0i"], "boxcolorp1", axis=3)  # [1,1,1,1] boxcolour+1
    # NOT boxcolour: G holds integers in [1..10] so an exact fp16 Equal is safe
    # and avoids the two full Sub/Abs difference planes.
    n("Equal", ["G", "boxcolorp1"], "isbp_b"); n("Not", ["isbp_b"], "notbp_b")
    # crosscell = inr AND inc AND notbox, associated so no full inbox plane forms
    n("And", ["inc_b", "notbp_b"], "cc_tmp")        # [1,1,30,30]
    n("And", ["inr_b", "cc_tmp"], "crosscell_b")
    # ONE value-carrying plane: (colour+1) at cross cells, 0 elsewhere.  Because
    # G is offset by +1, every real cross cell is >=1, so the SAME plane gives the
    # colour scalar (global ReduceMax) AND the cross-row/col occupancy profiles
    # (max>0.5), even when the special colour is 0.
    n("Where", ["crosscell_b", "G", "Zero16"], "Gc")     # [1,1,30,30]
    n("ReduceMax", ["Gc"], "colorp1", keepdims=1)
    n("ReduceMax", ["Gc"], "crossrow", axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["Gc"], "crosscol", axes=[2], keepdims=1)  # [1,1,1,30]

    # shift absolute occupancy to corner coords with a data-dependent Gather:
    # cr_corner[i] = crossrow[r0+i], cc_corner[j] = crosscol[c0+j]. The index
    # vectors are static-shape [30] so memory stays measurable; out-of-corner
    # positions are masked away later by the corner rectangle.
    init("Iarange32", np.arange(30, dtype=np.float32).reshape(30), np.float32)
    init("shp30", np.array([30], np.int64), np.int64)
    init("C29f", np.array(29.0, np.float32), np.float32)
    init("C0f", np.array(0.0, np.float32), np.float32)
    n("Cast", ["r0"], "r0s", to=F32)                # [1,1,1,1] f32
    n("Add", ["Iarange32", "r0s"], "row_idx_f")     # [1,1,1,30] (broadcast)
    n("Reshape", ["row_idx_f", "shp30"], "row_idx_r")
    n("Max", ["row_idx_r", "C0f"], "row_idx_c0")
    n("Min", ["row_idx_c0", "C29f"], "row_idx_cl")
    n("Cast", ["row_idx_cl"], "row_idx", to=I32)    # [30]
    n("Cast", ["c0"], "c0s", to=F32)
    n("Add", ["Iarange32", "c0s"], "col_idx_f")
    n("Reshape", ["col_idx_f", "shp30"], "col_idx_r")
    n("Max", ["col_idx_r", "C0f"], "col_idx_c0")
    n("Min", ["col_idx_c0", "C29f"], "col_idx_cl")
    n("Cast", ["col_idx_cl"], "col_idx", to=I32)    # [30]
    n("Gather", ["crossrow", "row_idx"], "crossrow_c", axis=2)   # [1,1,30,1]
    n("Gather", ["crosscol", "col_idx"], "crosscol_c", axis=3)   # [1,1,1,30]
    n("Greater", ["crossrow_c", "Half"], "cr_row_b")   # [1,1,30,1] bool cross-row
    n("Greater", ["crosscol_c", "Half"], "cc_col_b")   # [1,1,1,30] bool cross-col

    # corner rectangle: rows < h, cols < w  (separable 1-D bounds)
    n("Sub", ["r1", "r0"], "hm1"); n("Add", ["hm1", "One"], "h")
    n("Sub", ["c1", "c0"], "wm1"); n("Add", ["wm1", "One"], "w")
    n("Sub", ["h", "Icol"], "h_gt"); n("Greater", ["h_gt", "Half"], "rin_b")  # [1,1,30,1]
    n("Sub", ["w", "Irow"], "w_gt"); n("Greater", ["w_gt", "Half"], "cin_b")  # [1,1,1,30]

    # ---- separable uint8 label map (undo the +1 colour offset) ----
    n("Sub", ["boxcolorp1", "One"], "boxcolor")     # [1,1,1,1]
    n("Sub", ["colorp1", "One"], "color")
    n("Cast", ["boxcolor"], "boxc_u", to=U8)
    n("Cast", ["color"], "color_u", to=U8)
    init("v10u", np.array(10, np.uint8), np.uint8)  # off-box sentinel
    n("Or", ["cr_row_b", "cc_col_b"], "cross_b")    # [1,1,30,30]
    n("And", ["rin_b", "cin_b"], "corner_b")        # [1,1,30,30]
    n("Where", ["cross_b", "color_u", "boxc_u"], "Lin")
    n("Where", ["corner_b", "Lin", "v10u"], "Lab")
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["Lab", "chan"], "output")           # -> BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
