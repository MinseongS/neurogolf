"""task351 (ARC-AGI dc0a314f) — fill the green hole from 4-fold symmetry (Tier-S copy).

Rule (verified 500/500 fresh on the diagonal-mirror idea):
  The 16x16 grid is built from an 8x8 quadrant mirrored both horizontally and
  vertically, so it has full D2 symmetry: value(r,c) == value(15-r,c) ==
  value(r,15-c) == value(15-r,15-c).  A 5x5 cutout at (row,col) with
  row,col in {0,1,2,3} (always in the top-left quadrant) is erased to GREEN(=3);
  green appears NOWHERE else (random_color excludes green and the pattern fills
  every cell).  The 5x5 output is the original cutout content, which the diagonal
  mirror reconstructs exactly:

      output[i][j] = grid[15-row-i][15-col-j]     (always intact, in bottom-right)

  This is a pure SPATIAL COPY of input cells -> Tier-S.  The output one-hot at
  (i,j) equals the input one-hot at the mirrored source cell, so we just GATHER
  the FREE 10-channel input at the reversed mirror indices, then zero-pad the
  5x5 block to the [1,10,30,30] canvas (cells outside the 5x5 output are all-zero
  in the target encoding).

Recovery of (row,col): green channel (index 3) is 1 exactly on the 25 hole cells;
  row = first occupied row, col = first occupied col (ReduceMin of an index ramp).

Pipeline (ONNX opset 11):
  1. green = Gather(input, [3], axis=1)               [1,1,30,30]
  2. rowhas = ReduceMax(green, axes=[3]); row = ReduceMin(where(has, ramp, BIG))
     colhas = ReduceMax(green, axes=[2]); col similarly.
  3. ridx = 15 - row - arange(5)  (int64);  cidx = 15 - col - arange(5)
  4. Vr = Gather(input, ridx, axis=2)   [1,10,5,30]
     Vs = Gather(Vr,   cidx, axis=3)    [1,10,5,5]
  5. output = Pad(Vs, zeros, bottom/right to 30x30)   [1,10,30,30] f32 (>0 == one-hot)
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
I64 = TensorProto.INT64
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    WORK = 5

    # ---- recover hole top-left (row, col) as scalars -------------------------
    # green channel (3) marks the hole; reduce per-axis WITHOUT a 30x30 plane,
    # then slice channel 3 from the tiny [1,10,30,1]/[1,10,1,30] profiles.
    n("ReduceMax", ["input"], "rowprof", axes=[3], keepdims=1)  # [1,10,30,1]
    n("ReduceMax", ["input"], "colprof", axes=[2], keepdims=1)  # [1,10,1,30]
    init("g3", np.array([3], np.int64), np.int64)
    n("Gather", ["rowprof", "g3"], "rowhas", axis=1)          # [1,1,30,1]
    n("Gather", ["colprof", "g3"], "colhas", axis=1)          # [1,1,1,30]
    # green runs over 5 consecutive rows/cols; ArgMax(first-occurrence) of the
    # {0,1} occupancy gives the FIRST hole row/col = (row, col).  No ramp needed.
    n("ArgMax", ["rowhas"], "row_i", axis=2, keepdims=1)      # [1,1,1,1] int64
    n("ArgMax", ["colhas"], "col_i", axis=3, keepdims=1)      # [1,1,1,1] int64

    # ---- mirror source indices  idx = 15 - off - arange(WORK) ----------------
    init("shp1", np.array([1], np.int64), np.int64)
    n("Reshape", ["row_i", "shp1"], "row_s")                 # [1] int64
    n("Reshape", ["col_i", "shp1"], "col_s")                 # [1] int64
    # base = 15 - arange(WORK)  ->  [15,14,13,12,11]
    base = (15 - np.arange(WORK)).astype(np.int64)
    init("base", base, np.int64)                             # [WORK] int64
    n("Sub", ["base", "row_s"], "ridx")                      # [WORK] int64
    n("Sub", ["base", "col_s"], "cidx")                      # [WORK] int64

    # ---- gather the mirrored WORK x WORK one-hot window from FREE input -------
    n("Gather", ["input", "ridx"], "Vr", axis=2)             # [1,10,WORK,30] f32
    n("Gather", ["Vr", "cidx"], "Vs", axis=3)                # [1,10,WORK,WORK] f32

    # ---- zero-pad the 5x5 block to the 30x30 canvas (rest all-channels-off) ---
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("zero", np.array(0.0, np.float32), np.float32)
    n("Pad", ["Vs", "padpads", "zero"], "output", mode="constant")  # [1,10,30,30]

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task351", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
