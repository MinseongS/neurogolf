"""task214 (ARC-AGI 8e5a5113) — symmetrise a size x size colour block into 3
copies across a (3*size+2) wide grid, separated by gray columns.

Rule (size=3, grid 3x11, verified fresh):
  Block1 (cols 0-2)  = S            (identity copy of input cols 0-2)
  gray separators at output cols 3, 7 (channel 5), copied from input
  Block2 (cols 4-6)  = output[c][6-r] = S[r][c]   (a rotation/transpose of S)
  Block3 (cols 8-10) = output[2-r][10-c] = S[r][c]
  Everything else is background.  Output is a FIXED-geometry spatial remap of
  the input one-hot planes -> pure copy/relabel, scored as (value > 0).

Encoding (uint8 whole-pipeline; output is a one-hot copy):
  Every output cell in the 3x11 region copies a fixed input cell.  The source
  cols used are {0,1,2,3,7}; col 7 is gray and identical to col 3, so remap 7->3
  and the entire source fits in input cols 0-3 ([1,10,3,4]).  Flatten that region
  (row-major, 12 cells) and one Gather by a fixed 33-length flat index reproduces
  the whole 3x11 output region; Pad to 30x30 (background channel 0 fills off-grid
  -> handled by declaring the bg plane via pad of channel-0 = 1).

  All working planes are uint8 (itemsize 1): Slice forces fp32 (input dtype) so
  pay ONE fp32 slice of the tiny 3x4 corner, Cast to uint8, gather + pad in uint8.
  Output declared uint8; harness scores (out > 0) identically to the one-hot.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []
    seen = set()

    def init(name, arr, dt):
        if name in seen:
            return name
        seen.add(name)
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8

    # ---- source region: input cols 0-2, rows 0-2 -> [1,10,3,3] -------------
    # Only the coloured block (cols 0-2) needs sampling; the gray separators are
    # CONSTANT (channel 5 at output cols 3,7), so we append ONE synthetic gray
    # source cell instead of widening the fp32 slice to cols 0-3.
    init("s_start", np.array([0, 0], np.int64), np.int64)
    init("s_end", np.array([3, 3], np.int64), np.int64)
    init("s_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "s_start", "s_end", "s_ax"], "sl")     # [1,10,3,3] fp32
    n("Cast", ["sl"], "slu", to=U8)                             # [1,10,3,3] uint8

    # flatten the 3x3 coloured region row-major -> [1,10,9]
    init("shp_flat", np.array([1, 10, 9], np.int64), np.int64)
    n("Reshape", ["slu", "shp_flat"], "src9")                   # [1,10,9] uint8

    # append a gray source cell (channel 5 = 1) as flat index 9 -> [1,10,10]
    gray = np.zeros((1, 10, 1), np.uint8)
    gray[0, 5, 0] = 1
    init("gray_cell", gray, np.uint8)
    n("Concat", ["src9", "gray_cell"], "src", axis=2)           # [1,10,10] uint8

    # one Gather rebuilds the whole 3x11 region directly.  A 2-D index [3,11] on
    # rank-3 data [1,10,10] yields rank-4 [1,10,3,11] (data_rank-1 + idx_rank),
    # so NO second reshape is needed.  Index 9 = the appended gray cell.
    flat = np.array(
        [[0, 1, 2, 9, 6, 3, 0, 9, 8, 7, 6],
         [3, 4, 5, 9, 7, 4, 1, 9, 5, 4, 3],
         [6, 7, 8, 9, 8, 5, 2, 9, 2, 1, 0]], np.int64)
    init("gidx", flat, np.int64)
    n("Gather", ["src", "gidx"], "reg", axis=2)                 # [1,10,3,11] uint8

    # pad to 30x30.  The active grid is exactly the 3x11 region (every cell is
    # coloured / gray; channel 0 is never set in-region).  Off-grid cells are
    # all-zero in the target one-hot (verified), so a constant-0 pad is exact.
    init("pad30", np.array([0, 0, 0, 0, 0, 0, 27, 19], np.int64), np.int64)
    init("pv", np.array(0, np.uint8), np.uint8)
    n("Pad", ["reg", "pad30", "pv"], "output")                  # [1,10,30,30] uint8

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task214", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
