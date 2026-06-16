"""Task 161 (6cdd2623): laser rows/cols.

Rule (from ARC-GEN generator): an H x W grid (W in 15..25, H in 10..20) holds
scattered pixels in two palette colours plus "laser" markers in a third colour
(the megacolor).  A laser ROW r (1 <= r <= H-2) has the megacolor at grid[r][0]
and grid[r][W-1]; a laser COL c (1 <= c <= W-2) has it at grid[0][c] and
grid[H-1][c].  The megacolor appears ONLY at those border endpoints, never in the
interior.  The output paints the ENTIRE laser row / laser column with the
megacolor; every other in-grid cell is background (colour 0); off-grid is empty.

Reconstruction:
  L[r,c] = megacolor   if (row r is a laser row OR col c is a laser col) and in-grid
         = 0           if in-grid but not painted (background channel)
         = 10          if off-grid (sentinel: matches no channel)
  output = Equal(L, arange[0..9])           (10-way expansion in the FREE output)
A laser row is exactly  grid[r][0] == megacolor  (megacolor in col 0 occurs only
at laser-row left ends); a laser col is  grid[0][c] == megacolor.

Recovering the megacolor scalar (robust, 0 err / 12000 fresh instances, and the
ONNX net passes 500/500 fresh):
  megacolor = the unique colour k that is BOTH
     (a) present at both ends of a line:  (in left col AND in right col)
                                       OR (in top row AND in bottom row), and
     (b) NOT present anywhere strictly inside the grid.
Condition (a) alone occasionally admits a scattered colour that repeats at both
ends of one line; condition (b) (border-only) removes it -- the megacolor is the
unique border-only colour appearing at both ends of some line.  Presence is a
ReduceMax over each border line ([1,10], tiny); interior presence is counted as
total[k] - border_ring_count[k] (>0) with the 4 corners corrected for double
counting.  No per-row matched-pair products are needed (presence suffices).

All border lines are taken with Gather (static output shapes, so the harness can
measure memory -- runtime-tensor Slice would leave symbolic dims and trip the
"performance could not be measured" trap).  The laser masks come from Gathering
the megacolor channel out of the col-0 / row-0 one-hot lines directly (no colour
Conv / k-weighted reduce).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---------- constants ----------
    init("k10u", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("ar_r", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1), np.float32)
    init("ar_c", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30), np.float32)
    init("idx0_1d", np.array([0], np.int64), np.int64)     # [1] -> squeeze to scalar
    init("one_f", np.array(1.0, np.float32), np.float32)
    init("eps", np.array(0.5, np.float32), np.float32)
    init("mc_off", np.array(0, np.uint8), np.uint8)        # placeholder bg colour
    init("sent", np.array(10, np.uint8), np.uint8)         # off-grid sentinel

    n("Squeeze", ["idx0_1d"], "idx0")                            # 0-d scalar 0

    # ---------- in-grid extent H, W (scalars) ----------
    n("ReduceMax", ["input"], "occ_r", axes=[1, 3], keepdims=1)   # [1,1,30,1]
    n("ReduceMax", ["input"], "occ_c", axes=[1, 2], keepdims=1)   # [1,1,1,30]
    n("ReduceSum", ["occ_r"], "Hf", keepdims=0)                   # scalar float H
    n("ReduceSum", ["occ_c"], "Wf", keepdims=0)                   # scalar float W
    n("Sub", ["Hf", "one_f"], "Hm1f")                            # H-1 float
    n("Sub", ["Wf", "one_f"], "Wm1f")                            # W-1 float
    n("Cast", ["Hm1f"], "Hm1c", to=TensorProto.INT64)
    n("Cast", ["Wm1f"], "Wm1c", to=TensorProto.INT64)
    # Squeeze to a 0-d scalar so every Gather drops its axis identically (rank-3
    # lines); idx0 is already a 0-d scalar init.
    n("Squeeze", ["Hm1c"], "Hm1")                               # scalar int H-1
    n("Squeeze", ["Wm1c"], "Wm1")                               # scalar int W-1

    # ---------- border one-hot lines via Gather (fp32, rank-3) ----------
    # left  col : input[:,:,:,0]      right col : input[:,:,:,W-1]
    n("Gather", ["input", "idx0"], "left_h", axis=3)            # [1,10,30] fp32
    n("Gather", ["input", "Wm1"], "right_h", axis=3)           # [1,10,30] fp32
    # top   row : input[:,:,0,:]      bottom    : input[:,:,H-1,:]
    n("Gather", ["input", "idx0"], "top_h", axis=2)           # [1,10,30] fp32
    n("Gather", ["input", "Hm1"], "bot_h", axis=2)           # [1,10,30] fp32

    # ---------- per-colour line presence ([1,10], all tiny) ----------
    # A laser colour is present at BOTH ends of a line: in (left AND right) columns
    # or in (top AND bottom) rows.  Presence (ReduceMax) over each border line.
    n("ReduceMax", ["left_h"], "pL", axes=[2], keepdims=0)    # [1,10] 0/1
    n("ReduceMax", ["right_h"], "pR", axes=[2], keepdims=0)
    n("ReduceMax", ["top_h"], "pT", axes=[2], keepdims=0)
    n("ReduceMax", ["bot_h"], "pB", axes=[2], keepdims=0)
    n("Greater", ["pL", "eps"], "bL")                         # [1,10] bool
    n("Greater", ["pR", "eps"], "bR")
    n("Greater", ["pT", "eps"], "bT")
    n("Greater", ["pB", "eps"], "bB")
    n("And", ["bL", "bR"], "row_ends")                        # colour at both col ends
    n("And", ["bT", "bB"], "col_ends")                        # colour at both row ends
    n("Or", ["row_ends", "col_ends"], "both_ends")            # [1,10] bool

    # ---------- interior presence per colour ([1,10]) ----------
    # total count vs border-ring count: a colour with interior cells has
    # interior_cnt > 0.  Ring count = 4 side lines minus the 4 doubly-counted corners.
    n("ReduceSum", ["input"], "total", axes=[2, 3], keepdims=0)  # [1,10] fp32
    n("ReduceSum", ["left_h"], "cL", axes=[2], keepdims=0)    # [1,10]
    n("ReduceSum", ["right_h"], "cR", axes=[2], keepdims=0)
    n("ReduceSum", ["top_h"], "cT", axes=[2], keepdims=0)
    n("ReduceSum", ["bot_h"], "cB", axes=[2], keepdims=0)
    n("Add", ["cL", "cR"], "cLR")
    n("Add", ["cT", "cB"], "cTB")
    n("Add", ["cLR", "cTB"], "csum")                          # [1,10] (corners x2)
    n("Gather", ["left_h", "idx0"], "c00", axis=2)            # [1,10] corner (0,0)
    n("Gather", ["left_h", "Hm1"], "c10", axis=2)
    n("Gather", ["right_h", "idx0"], "c01", axis=2)
    n("Gather", ["right_h", "Hm1"], "c11", axis=2)
    n("Add", ["c00", "c10"], "ca")
    n("Add", ["c01", "c11"], "cb")
    n("Add", ["ca", "cb"], "corners")                         # [1,10]
    n("Sub", ["csum", "corners"], "border_cnt")               # [1,10] ring count
    n("Sub", ["total", "border_cnt"], "interior_cnt")         # [1,10]
    n("Greater", ["interior_cnt", "eps"], "is_int")           # [1,10] bool interior
    n("Not", ["is_int"], "not_int")                           # [1,10] bool border-only
    n("And", ["both_ends", "not_int"], "is_mega")             # [1,10] bool (unique True)

    # ---------- megacolor index = argmax over colours ----------
    n("Cast", ["is_mega"], "is_mega_f", to=TensorProto.FLOAT)  # [1,10]
    n("ArgMax", ["is_mega_f"], "mc_idx", axis=1, keepdims=1)   # [1,1] int64
    n("Cast", ["mc_idx"], "mc_u8", to=TensorProto.UINT8)       # [1,1] uint8 megacolor

    # ---------- laser-row / laser-col masks ----------
    # colour index of the left column and top row (Conv-free channel reduce)
    # laser row r  <=>  col-0 one-hot has the megacolor channel set at row r.
    # Gather the megacolor channel directly from the line one-hots (no k-reduce).
    n("Squeeze", ["mc_idx"], "mc_scalar")                     # 0-d scalar megacolor idx
    n("Gather", ["left_h", "mc_scalar"], "row_pres", axis=1)  # [1,30] fp16 0/1
    n("Gather", ["top_h", "mc_scalar"], "col_pres", axis=1)   # [1,30] fp16 0/1
    init("half", np.array(0.5, np.float32), np.float32)
    init("sh_r", np.array([1, 1, 30, 1], np.int64), np.int64)
    init("sh_c", np.array([1, 1, 1, 30], np.int64), np.int64)
    n("Reshape", ["row_pres", "sh_r"], "row_pres4")          # [1,1,30,1]
    n("Reshape", ["col_pres", "sh_c"], "col_pres4")          # [1,1,1,30]
    n("Greater", ["row_pres4", "half"], "rowmask")           # [1,1,30,1] bool
    n("Greater", ["col_pres4", "half"], "colmask")           # [1,1,1,30] bool

    # ---------- in-grid bounds (1-D) ----------
    n("Less", ["ar_r", "Hf"], "r_in")                          # [1,1,30,1] bool
    n("Less", ["ar_c", "Wf"], "c_in")                          # [1,1,1,30] bool

    # ---------- label map L ----------
    # paint = laser AND in-grid; lasers fill only the in-grid extent of their row/
    # col, so the row/col mask must be clipped to (r<H, c<W) before painting.
    n("Or", ["rowmask", "colmask"], "linemask")                # [1,1,30,30] bool
    n("And", ["r_in", "c_in"], "ingrid")                       # [1,1,30,30] bool
    n("Where", ["linemask", "mc_u8", "mc_off"], "inner")       # mc on laser else 0
    n("Where", ["ingrid", "inner", "sent"], "L")               # off-grid -> 10
    n("Equal", ["L", "k10u"], "output")                        # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task161", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
