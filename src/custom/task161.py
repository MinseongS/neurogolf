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

Megacolor scalar (0 err / 30000 fresh instances bar the ~6/30000 generator
ambiguities the deployed net also hits): the unique colour k that is BOTH
  (a) present at both ends of a line (left&right cols OR top&bottom rows), and
  (b) NOT present strictly inside the grid (border-only).
Both are derived from the four border-line one-hots (Gather of col 0 / col W-1 /
row 0 / row H-1).  Per-side presence reuses the same border-line COUNTS that the
interior test needs (count>0 == present), so no separate ReduceMax presence
planes are built.  border_ring = cL+cR+cT+cB - corners (4 corners double-counted).

Border masks for the two laser directions come from Gathering the megacolor
channel out of the col-0 / row-0 one-hots.  The label collapses to ONE
[1,1,30,30] carrier built with broadcast-condition Wheres (no separate in-grid
And plane), then expands 10-ways in the FREE Equal output.

All data-dependent border reads use Gather with squeezed SCALAR indices (static
output shapes); a runtime Slice would leave symbolic dims and trip the
"performance could not be measured" trap.
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
    init("bg0", np.array(0, np.uint8), np.uint8)           # background colour 0
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
    n("Squeeze", ["Hm1c"], "Hm1")                               # scalar int H-1
    n("Squeeze", ["Wm1c"], "Wm1")                               # scalar int W-1

    # ---------- border one-hot lines via Gather (fp32, rank-3) ----------
    n("Gather", ["input", "idx0"], "left_h", axis=3)            # col 0   [1,10,30]
    n("Gather", ["input", "Wm1"], "right_h", axis=3)           # col W-1 [1,10,30]
    n("Gather", ["input", "idx0"], "top_h", axis=2)            # row 0   [1,10,30]
    n("Gather", ["input", "Hm1"], "bot_h", axis=2)             # row H-1 [1,10,30]

    # ---------- per-colour side COUNTS ([1,10]) ----------
    n("ReduceSum", ["left_h"], "cL", axes=[2], keepdims=0)
    n("ReduceSum", ["right_h"], "cR", axes=[2], keepdims=0)
    n("ReduceSum", ["top_h"], "cT", axes=[2], keepdims=0)
    n("ReduceSum", ["bot_h"], "cB", axes=[2], keepdims=0)

    # both-ends presence (count>0) per colour -- reuses the counts, no ReduceMax
    n("Greater", ["cL", "eps"], "bL")
    n("Greater", ["cR", "eps"], "bR")
    n("Greater", ["cT", "eps"], "bT")
    n("Greater", ["cB", "eps"], "bB")
    n("And", ["bL", "bR"], "row_ends")
    n("And", ["bT", "bB"], "col_ends")
    n("Or", ["row_ends", "col_ends"], "both_ends")               # [1,10] bool

    # ---------- interior presence per colour ----------
    # interior_cnt = total - (cL+cR+cT+cB - corners)
    n("ReduceSum", ["input"], "total", axes=[2, 3], keepdims=0)  # [1,10]
    n("Add", ["cL", "cR"], "cLR")
    n("Add", ["cT", "cB"], "cTB")
    n("Add", ["cLR", "cTB"], "csum")                             # corners x2
    n("Gather", ["left_h", "idx0"], "c00", axis=2)              # corner (0,0)
    n("Gather", ["left_h", "Hm1"], "c10", axis=2)              # corner (H-1,0)
    n("Gather", ["right_h", "idx0"], "c01", axis=2)           # corner (0,W-1)
    n("Gather", ["right_h", "Hm1"], "c11", axis=2)           # corner (H-1,W-1)
    n("Add", ["c00", "c10"], "ca")
    n("Add", ["c01", "c11"], "cb")
    n("Add", ["ca", "cb"], "corners")                           # [1,10]
    n("Sub", ["csum", "corners"], "border_cnt")                 # ring count
    n("Sub", ["total", "border_cnt"], "interior_cnt")
    n("Greater", ["interior_cnt", "eps"], "is_int")
    n("Not", ["is_int"], "not_int")                             # border-only
    n("And", ["both_ends", "not_int"], "is_mega")               # [1,10] unique True

    # ---------- megacolor index ----------
    n("Cast", ["is_mega"], "is_mega_f", to=TensorProto.FLOAT)
    n("ArgMax", ["is_mega_f"], "mc_idx", axis=1, keepdims=1)    # [1,1] int64
    n("Cast", ["mc_idx"], "mc_u8", to=TensorProto.UINT8)        # [1,1] uint8
    n("Squeeze", ["mc_idx"], "mc_scalar")                       # 0-d scalar

    # ---------- laser-row / laser-col masks ----------
    n("Gather", ["left_h", "mc_scalar"], "row_pres", axis=1)   # [1,30] fp32
    n("Gather", ["top_h", "mc_scalar"], "col_pres", axis=1)    # [1,30] fp32
    # threshold to bool FIRST (30B planes) then reshape the bool (cheaper than
    # reshaping the fp32 [1,30] up to a [1,1,30,1]/[1,1,1,30] fp32 plane).
    n("Greater", ["row_pres", "eps"], "row_b")                 # [1,30] bool
    n("Greater", ["col_pres", "eps"], "col_b")                 # [1,30] bool
    init("sh_r", np.array([1, 1, 30, 1], np.int64), np.int64)
    init("sh_c", np.array([1, 1, 1, 30], np.int64), np.int64)
    n("Reshape", ["row_b", "sh_r"], "rowmask")                 # [1,1,30,1] bool
    n("Reshape", ["col_b", "sh_c"], "colmask")                 # [1,1,1,30] bool

    # ---------- in-grid bounds (1-D) ----------
    n("Less", ["ar_r", "Hf"], "r_in")                          # [1,1,30,1] bool
    n("Less", ["ar_c", "Wf"], "c_in")                          # [1,1,1,30] bool

    # ---------- label map L (single [1,1,30,30] carrier) ----------
    # The row/col laser Or is folded INTO the nested Where so no [1,1,30,30]
    # linemask plane materialises: paint_r is the tiny [1,1,30,1] "mc where the
    # row is a laser else bg", and L0 paints mc where the col is a laser ELSE
    # falls back to paint_r -> (rowmask OR colmask) ? mc : bg in ONE plane.
    # Then two broadcast-condition Wheres stamp the off-grid sentinel without a
    # separate in-grid And plane: tmp = c_in ? L0 : 10 ; L = r_in ? tmp : 10.
    n("Where", ["rowmask", "mc_u8", "bg0"], "paint_r")         # [1,1,30,1] uint8
    n("Where", ["colmask", "mc_u8", "paint_r"], "L0")          # [1,1,30,30] laser?mc:bg
    n("Where", ["c_in", "L0", "sent"], "tmp")                  # cols off-grid -> 10
    n("Where", ["r_in", "tmp", "sent"], "L")                   # rows off-grid -> 10
    n("Equal", ["L", "k10u"], "output")                        # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task161", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
