"""task132 (ARC-AGI 56ff96f3) — fill each box from its two diagonal corners.

Rule (from the generator): the grid holds 1..2 axis-aligned rectangular "boxes".
Each box has a UNIQUE colour and is drawn in the input as exactly its two
DIAGONAL corner cells (top-left+bottom-right, or top-right+bottom-left depending
on a flip).  Boxes never overlap (overlaps() with margin 1).  The output FILLS
each box's full bounding rectangle with that box's colour; every other in-grid
cell is background (colour 0).

Because each colour's two marked cells are the two diagonal corners of its box,
the bounding box of that colour's pixels (min/max row, min/max col) is exactly
the box rectangle:

  output[c, r, c'] = present_c AND rmin_c<=r<=rmax_c AND cmin_c<=c'<=cmax_c  (c>=1)
  output[0, r, c'] = in-grid AND not covered by any box   (background channel)

This reduces to a per-cell colour-index label map L (0 = background) followed by
output = Equal(L, arange[0..9]) -> free BOOL one-hot.

L is built WITHOUT ever materializing a [1,10,H,W] plane.  Boxes are disjoint, so
with rowcond_c[r] (channel c covers row r) and colcond_c[c'] (covers col c'),
    L[r,c'] = sum_c (c * rowcond_c[r]) * colcond_c[c']
            = MatMul( A[r,c]=c*rowcond_c[r] , B[c,c']=colcond_c[c'] ).
Background channel (c=0) carries weight 0; disjointness means no cell is stamped
twice, so in-grid background cells get L=0 automatically.  Off-grid padding cells
get a sentinel (10) so no channel — not even ch0 — lights up there.

The generator grid is at most 15x15 (width,height in 6..15) and is placed at the
top-left of the 30x30 canvas, so ALL active cells live in a WORK=16 window; the
label map is built at 16x16 and Pad'ed (sentinel 10) to 30x30 before the Equal.

Dominant intermediates: the two fp32 occupancy reductions [1,10,30,1]+[1,10,1,30]
(1200B each, irreducible — ReduceMax of the fp32 input emits fp32) and the padded
label map [1,1,30,30] uint8 (900B).  Everything else is fp16/bool at <=16 extent.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
B = TensorProto.BOOL

WORK = 15  # grid is at most 15x15 (width,height in 6..15) at the top-left


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    BIG = 100.0

    # ---- per-channel occupancy over the FREE input (fp32 reductions) ---------
    n("ReduceMax", ["input"], "rowocc", axes=[3], keepdims=1)  # [1,10,30,1] f32
    n("ReduceMax", ["input"], "colocc", axes=[2], keepdims=1)  # [1,10,1,30] f32

    init("half", np.array(0.5, np.float32), np.float32)
    # booleanise the FULL reductions (bool plane 300B << fp32 crop), then crop.
    init("s0", np.array([0], np.int64), np.int64)
    init("sW", np.array([WORK], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    n("Greater", ["rowocc", "half"], "rowb_full")             # [1,10,30,1] bool
    n("Greater", ["colocc", "half"], "colb_full")             # [1,10,1,30] bool
    n("Slice", ["rowb_full", "s0", "sW", "ax2"], "rowb")      # [1,10,WORK,1] bool
    n("Slice", ["colb_full", "s0", "sW", "ax3"], "colb")      # [1,10,1,WORK] bool

    # row / col index ramps (fp16: values <16 and +/-100 exact)
    rr = np.arange(WORK, dtype=np.float16).reshape(1, 1, WORK, 1)
    rc = np.arange(WORK, dtype=np.float16).reshape(1, 1, 1, WORK)
    init("rr", rr, np.float16)
    init("rc", rc, np.float16)
    init("PBIG", np.array(BIG, np.float16), np.float16)
    init("NBIG", np.array(-BIG, np.float16), np.float16)

    # per-channel bbox bounds (fp16). absent channel => rmin=+BIG,rmax=-BIG => empty
    n("Where", ["rowb", "rr", "PBIG"], "rmin_src")             # [1,10,WORK,1] f16
    n("ReduceMin", ["rmin_src"], "rmin", axes=[2], keepdims=1)  # [1,10,1,1] f16
    n("Where", ["rowb", "rr", "NBIG"], "rmax_src")
    n("ReduceMax", ["rmax_src"], "rmax", axes=[2], keepdims=1)
    n("Where", ["colb", "rc", "PBIG"], "cmin_src")             # [1,10,1,WORK]
    n("ReduceMin", ["cmin_src"], "cmin", axes=[3], keepdims=1)
    n("Where", ["colb", "rc", "NBIG"], "cmax_src")
    n("ReduceMax", ["cmax_src"], "cmax", axes=[3], keepdims=1)

    # rowcond_c[r] = rmin_c <= r <= rmax_c = NOT( r<rmin  OR  rmax<r )
    n("Less", ["rr", "rmin"], "r_lt_min")
    n("Less", ["rmax", "rr"], "r_gt_max")
    n("Or", ["r_lt_min", "r_gt_max"], "r_out")
    n("Not", ["r_out"], "rowcond")                            # [1,10,WORK,1] bool

    n("Less", ["rc", "cmin"], "c_lt_min")
    n("Less", ["cmax", "rc"], "c_gt_max")
    n("Or", ["c_lt_min", "c_gt_max"], "c_out")
    n("Not", ["c_out"], "colcond")                            # [1,10,1,WORK] bool

    # ---- weighted contraction over the channel axis via MatMul --------------
    # A[r,c]=c*rowcond_c[r] ; B[c,c']=colcond_c[c'] ; L = A @ B
    # wrow = channel-weight where row in band, else 0 (one Where, no extra cast).
    chvec = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("chvec", chvec, np.float16)
    init("zero16", np.array(0.0, np.float16), np.float16)
    n("Where", ["rowcond", "chvec", "zero16"], "wrow")        # [1,10,WORK,1] f16
    n("Cast", ["colcond"], "colcond_f", to=F16)               # [1,10,1,WORK]
    n("Transpose", ["wrow"], "wrow_t", perm=[0, 3, 2, 1])     # [1,1,WORK,10]
    n("Transpose", ["colcond_f"], "col_t", perm=[0, 2, 1, 3])  # [1,1,10,WORK]
    n("MatMul", ["wrow_t", "col_t"], "Lf")                    # [1,1,WORK,WORK] f16

    # ---- restrict to the in-grid region (off-grid cells -> sentinel 10) ------
    # The grid is a solid HxW rectangle at the top-left; background channel 0
    # fills every in-grid cell, so a row is in-grid iff ANY channel occupies it.
    # in-grid = rowany[r] AND colany[c']  (separable, no [1,1,30,30] plane).
    # rowany[r] = ANY channel occupies row r.  Reduce the fp32 occupancy over the
    # channel axis (tiny [1,1,30,1] output), then crop+booleanise.
    n("ReduceMax", ["rowocc"], "rowany30", axes=[1], keepdims=1)  # [1,1,30,1] f32
    n("ReduceMax", ["colocc"], "colany30", axes=[1], keepdims=1)  # [1,1,1,30] f32
    n("Slice", ["rowany30", "s0", "sW", "ax2"], "rowany")        # [1,1,WORK,1]
    n("Slice", ["colany30", "s0", "sW", "ax3"], "colany")        # [1,1,1,WORK]
    n("Greater", ["rowany", "half"], "rowany_b")
    n("Greater", ["colany", "half"], "colany_b")
    n("And", ["rowany_b", "colany_b"], "ingrid_b")            # [1,1,WORK,WORK] bool
    # cast colour index to uint8 first, then mask off-grid to sentinel 10 (uint8
    # plane is half the fp16 one).
    n("Cast", ["Lf"], "Lf_u8", to=U8)                         # [1,1,WORK,WORK] u8
    n("Where", ["ingrid_b", "Lf_u8", "u10"], "Lw")            # [1,1,WORK,WORK] u8
    # pad WORK window -> 30x30 with sentinel 10 (off-grid)
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Pad", ["Lw", "padpads", "u10"], "L", mode="constant")  # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                       # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task132", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
