"""Task 039 (2013d3e2): output = the 3x3 top-left crop of the pinwheel.

Rule (from ARC-GEN generator, verified fresh):
  The 10x10 input holds a "pinwheel": 4 rotated copies of a 3x3 colour pattern,
  placed so the UPPER-LEFT 3x3 block at (row, col) equals the 3x3 output exactly
  (output[r][c] = grid[row+r][col+c], r,c in 0..2).  row, col in {1,2,3}.  The
  full figure spans rows row..row+5, cols col..col+5, i.e. within rows 1..8 and
  cols 1..8 of the 10x10 grid.  The 3x3 crop therefore starts at the bounding-box
  minimum (min nonzero row, min nonzero col) -- verified equal to the output on
  thousands of fresh instances.

  => Pure spatial COPY (Tier S): find (first_row, first_col) = top-left nonzero
  cell, then gather the 3x3 window of the FREE input directly, pad to 30x30.

Encoding:
  Because min_row and min_col are both in {1,2,3}, the FIRST present row/col is
  the global minimum.  So presence only needs to be scanned over the small
  candidate windows:
    - row presence: slice input[ch 1..9, rows 1..3, cols 1..8] -> [1,9,3,8] (864B),
      ReduceMax over channel+col -> [1,3] presence of rows 1,2,3; ArgMax -> +1.
    - col presence: slice input[ch 1..9, rows 1..8, cols 1..3] -> [1,9,8,3] (864B),
      ReduceMax over channel+row -> [1,3] presence of cols 1,2,3; ArgMax -> +1.
  (Channel 0 = background is excluded by slicing channels 1..9; cols/rows 1..8
   cover the entire possible pinwheel extent.)
  Then row_indices = first_row + [0,1,2], col_indices = first_col + [0,1,2];
  Gather(input, row_indices, axis=2) [1,10,3,30], Gather(.,col_indices,axis=3)
  [1,10,3,3], Pad to [1,10,30,30] (zeros) -> fp32 output (a copy of input cells).
"""

import numpy as np
import onnx
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

    # ---- candidate-window slices for presence (channels 1..9 exclude bg) ----
    # rows 1..3, cols 1..8  -> find first present row
    init("rs_st", np.array([1, 1, 1], np.int64), np.int64)
    init("rs_en", np.array([10, 4, 9], np.int64), np.int64)
    init("rs_ax", np.array([1, 2, 3], np.int64), np.int64)
    # cols 1..3, rows 1..8  -> find first present col
    init("cs_st", np.array([1, 1, 1], np.int64), np.int64)
    init("cs_en", np.array([10, 9, 4], np.int64), np.int64)
    init("cs_ax", np.array([1, 2, 3], np.int64), np.int64)

    init("offsets", np.array([0, 1, 2], np.int64), np.int64)
    # active region (rows 1..8, cols 1..8, all channels) for the crop gather
    init("act_st", np.array([1, 1], np.int64), np.int64)
    init("act_en", np.array([9, 9], np.int64), np.int64)
    init("act_ax", np.array([2, 3], np.int64), np.int64)
    # pad the [1,10,3,3] crop to [1,10,30,30] (zeros) at the top-left corner
    init("pads", np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64), np.int64)
    init("padval", np.array(0.0, np.float32), np.float32)

    # ---- first_row (min nonzero row, in {1,2,3}); local index in {0,1,2} ----
    n("Slice", ["input", "rs_st", "rs_en", "rs_ax"], "row_win")   # [1,9,3,8] f32
    n("ReduceMax", ["row_win"], "row_pres", axes=[1, 3], keepdims=0)  # [1,3]
    n("ArgMax", ["row_pres"], "fr_local", axis=1, keepdims=0)     # [1]  (0..2)

    # ---- first_col (min nonzero col, in {1,2,3}); local index in {0,1,2} ----
    n("Slice", ["input", "cs_st", "cs_en", "cs_ax"], "col_win")   # [1,9,8,3] f32
    n("ReduceMax", ["col_win"], "col_pres", axes=[1, 2], keepdims=0)  # [1,3]
    n("ArgMax", ["col_pres"], "fc_local", axis=1, keepdims=0)     # [1]

    # ---- gather the 3x3 window from the small active region (copy) ----
    # act = input[rows 1..8, cols 1..8] -> local indices = fr_local + [0,1,2]
    n("Slice", ["input", "act_st", "act_en", "act_ax"], "act")    # [1,10,8,8] f32
    n("Add", ["fr_local", "offsets"], "row_indices")              # [3] in 0..4
    n("Add", ["fc_local", "offsets"], "col_indices")              # [3] in 0..4
    n("Gather", ["act", "row_indices"], "row_crop", axis=2)       # [1,10,3,8]
    n("Gather", ["row_crop", "col_indices"], "small", axis=3)     # [1,10,3,3]
    n("Pad", ["small", "pads", "padval"], "output", mode="constant")  # [1,10,30,30]

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
