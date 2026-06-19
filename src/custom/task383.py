"""task383 (ARC-AGI f1cefba8) — barnacle stripes through a coloured box.

Rule: ONE axis-aligned box at (top,left)..(bottom,right) with a 2-px OUTER ring
in colour C0 and a SOLID inner block in colour C1.  A few "barnacle" markers are
single C1 pixels sitting on the inner ring line (rows top+1 / bottom-1, cols
left+1 / right-1).  Each marker projects a STRIPE perpendicular to the ring:
  * a marker on the TOP/BOTTOM ring -> a full COLUMN stripe in its column;
  * a marker on the LEFT/RIGHT ring -> a full ROW stripe in its row.
Output = the clean box, plus every stripe line:
  * INSIDE the box the crossed cell -> C0;
  * OUTSIDE the box but IN-GRID -> C1 (off-grid stays background).

FRONT-END (reused, kojimar7113): a 1x1-effective dilated 2x2 Conv emits the
colour-index plane `color` cropped to the 24x24 active region (bg->10,
off-grid->11, colour c->c) as uint8; bbox top/bottom/left/right come from ArgMax
of per-row/col occupancy; C0/C1 from corner reads; the marker indicators
row_select_u8[1,24] / col_select_u8[1,24] mark which rows/cols carry a barnacle.

BACK-END (this rewrite, plane-lean): everything separable into 1-D vectors so the
old TopK + ScatterND + GatherElements/ScatterElements machinery (3 sequential
24x24 colour planes + index planes) collapses to just 2 extra 24x24 planes:
    paint = (rowmark[r] & ingrid_col[c]) | (colmark[c] & ingrid_row[r])
    final = Where(paint, Where(in_row & in_col, C0, C1), color)
then Pad to 30x30 and Equal(channel_values) into the FREE bool output.

Dominant intermediate: the forced fp32 colour-index entry plane color_f
[1,1,24,24]=2304B (10->1 Conv inherits fp32; input is fp32 so it cannot be born
fp16 without an 18000B input cast).  Everything else is uint8 24x24 or 1-D.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
B = TensorProto.BOOL
I64 = TensorProto.INT64

WORK = 24  # active-region crop (grid width/height <= 16+8 = 24)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour-index entry plane (kojimar dilated-conv trick) --------------
    # color = 11 + sum_k w_k*input_k  with w=[-1,-10,..,-2] -> bg->10, c->c,
    # off-grid (all-zero channels)->11.  Dilated 2x2 kernel (only [0,0] nonzero)
    # crops the output to 24x24 = the max active grid.
    cw = np.zeros((1, 10, 2, 2), np.float32)
    cw[0, 0, 0, 0] = -1.0
    for c in range(1, 10):
        cw[0, c, 0, 0] = -(11 - c)
    init("color_weights", cw, np.float32)
    init("color_bias", np.array([11.0], np.float32), np.float32)
    n("Conv", ["input", "color_weights", "color_bias"], "color_f",
      dilations=[6, 6], kernel_shape=[2, 2])              # [1,1,24,24] f32 (entry)
    n("Cast", ["color_f"], "color", to=U8)                 # [1,1,24,24] u8 (576B)

    init("ten_u8", np.array([10], np.uint8), np.uint8)
    init("eleven_u8", np.array([11], np.uint8), np.uint8)
    init("rev_work", np.arange(WORK - 1, -1, -1, np.int64), np.int64)
    init("last_work_idx", np.array([WORK - 1], np.int64), np.int64)
    init("one", np.array([1], np.int64), np.int64)
    init("two", np.array([2], np.int64), np.int64)

    # ---- in-grid extents (per-row / per-col min < 11 -> row/col touches grid) -
    n("ReduceMin", ["color"], "row_min", axes=[1, 3], keepdims=0)  # [1,24]
    n("ReduceMin", ["color"], "col_min", axes=[1, 2], keepdims=0)  # [1,24]
    n("Less", ["row_min", "eleven_u8"], "ingr_b")          # [1,24] bool (in-grid row)
    n("Less", ["col_min", "eleven_u8"], "ingc_b")          # [1,24] bool (in-grid col)
    # box occupancy (min < 10 -> row/col has a coloured/box cell)
    n("Less", ["row_min", "ten_u8"], "row_any_b")          # [1,24] bool
    n("Less", ["col_min", "ten_u8"], "col_any_b")          # [1,24] bool
    n("Cast", ["row_any_b"], "row_any", to=U8)
    n("Cast", ["col_any_b"], "col_any", to=U8)

    # ---- bbox scalars via ArgMax of the occupancy vectors -------------------
    n("ArgMax", ["row_any"], "top", axis=1, keepdims=0)    # [1] i64
    n("Gather", ["row_any", "rev_work"], "row_any_rev", axis=1)
    n("ArgMax", ["row_any_rev"], "bottom_rev", axis=1, keepdims=0)
    n("Sub", ["last_work_idx", "bottom_rev"], "bottom")
    n("ArgMax", ["col_any"], "left", axis=1, keepdims=0)
    n("Gather", ["col_any", "rev_work"], "col_any_rev", axis=1)
    n("ArgMax", ["col_any_rev"], "right_rev", axis=1, keepdims=0)
    n("Sub", ["last_work_idx", "right_rev"], "right")
    n("Add", ["top", "one"], "top1")                       # inner ring rows
    n("Sub", ["bottom", "one"], "bottom1")
    n("Add", ["left", "one"], "left1")                     # inner ring cols
    n("Sub", ["right", "one"], "right1")
    n("Add", ["top", "two"], "top2")                       # inner block corner
    n("Add", ["left", "two"], "left2")

    # ---- C0 (outer/ring) and C1 (inner) colours -----------------------------
    n("Gather", ["color", "top"], "outer_row", axis=2)
    n("Gather", ["outer_row", "left"], "outer_color", axis=3)   # [1,1] u8 -> C0
    n("Gather", ["color", "top2"], "inner_row", axis=2)
    n("Gather", ["inner_row", "left2"], "inner_color", axis=3)  # [1,1] u8 -> C1

    # ---- marker indicators (which cols carry a top/bottom barnacle, etc.) ----
    n("Concat", ["top1", "bottom1"], "marker_rows_idx", axis=0)
    n("Gather", ["color", "marker_rows_idx"], "marker_rows_color", axis=2)
    n("Equal", ["marker_rows_color", "inner_color"], "marker_rows_b")
    n("Cast", ["marker_rows_b"], "marker_rows_u8", to=U8)
    n("ReduceMax", ["marker_rows_u8"], "col_select_u8", axes=[1, 2], keepdims=0)  # [1,24]
    n("Concat", ["left1", "right1"], "marker_cols_idx", axis=0)
    n("Gather", ["color", "marker_cols_idx"], "marker_cols_color", axis=3)
    n("Equal", ["marker_cols_color", "inner_color"], "marker_cols_b")
    n("Cast", ["marker_cols_b"], "marker_cols_u8", to=U8)
    n("ReduceMax", ["marker_cols_u8"], "row_select_u8", axes=[1, 3], keepdims=0)  # [1,24]

    # ---- reshape the 1-D vectors to broadcastable [1,1,24,1] / [1,1,1,24] -----
    # in_row / in_col come FREE from the box-occupancy vectors (row_any_b is
    # exactly "row r contains a box cell" = top<=r<=bottom since the box is
    # contiguous) -> no ramp + compare machinery needed.
    init("shp_row", np.array([1, 1, WORK, 1], np.int64), np.int64)
    init("shp_col", np.array([1, 1, 1, WORK], np.int64), np.int64)
    n("Reshape", ["col_select_u8", "shp_col"], "colmark_u8")   # marker COLUMN flag
    n("Reshape", ["row_select_u8", "shp_row"], "rowmark_u8")   # marker ROW flag
    n("Cast", ["colmark_u8"], "colmark", to=B)                 # [1,1,1,24]
    n("Cast", ["rowmark_u8"], "rowmark", to=B)                 # [1,1,24,1]
    n("Reshape", ["ingr_b", "shp_row"], "ingr")                # [1,1,24,1]
    n("Reshape", ["ingc_b", "shp_col"], "ingc")                # [1,1,1,24]
    n("Reshape", ["row_any_b", "shp_row"], "in_row")           # [1,1,24,1] bool
    n("Reshape", ["col_any_b", "shp_col"], "in_col")           # [1,1,1,24] bool

    # ---- per-axis stripe colour VECTORS (SMALL 1-D, no full mask plane) ------
    # A horizontal stripe sits on a marker ROW; the value at col c is:
    #   off-grid col          -> 11 (sentinel; pads to background in output)
    #   in-grid, inside box   -> C0
    #   in-grid, outside box  -> C1
    # Folding the in-grid gate INTO the value vector removes the full And mask:
    # the stripe mask is then just rowmark (a 1-D broadcast), no 24x24 plane.
    n("Where", ["in_col", "outer_color", "inner_color"], "rowval_ig")  # [1,1,1,24]
    n("Where", ["ingc", "rowval_ig", "eleven_u8"], "rowval")           # [1,1,1,24]
    # A vertical stripe sits on a marker COL; value at row r (same structure):
    n("Where", ["in_row", "outer_color", "inner_color"], "colval_ig")  # [1,1,24,1]
    n("Where", ["ingr", "colval_ig", "eleven_u8"], "colval")           # [1,1,24,1]

    # ---- final colour plane (only 2 extra 24x24 planes) ---------------------
    # vertical stripes first, horizontal stripes win at crossings.
    n("Where", ["colmark", "colval", "color"], "final0")      # [1,1,24,24] u8 (576)
    n("Where", ["rowmark", "rowval", "final0"], "final")      # [1,1,24,24] u8 (576)

    # ---- pad to 30x30 (off-grid -> 11 sentinel, never equals a channel) ------
    init("pad_to_30", np.array([0, 0, 0, 0, 0, 0, 6, 6], np.int64), np.int64)
    n("Pad", ["final", "pad_to_30", "eleven_u8"], "padded")   # [1,1,30,30] u8 (900)
    init("channel_values",
         np.array([10, 1, 2, 3, 4, 5, 6, 7, 8, 9], np.uint8).reshape(1, 10, 1, 1),
         np.uint8)
    n("Equal", ["padded", "channel_values"], "output")        # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task383", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 12)])
