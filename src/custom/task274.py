"""task274 (ARC-AGI b0c4d837) — read the empty-cup "space" height into a 3x3 tally.

Rule (from the generator b0c4d837):
  The input draws a "cup": gray (5) side-walls at columns col_gap and
  col_gap+base-1 spanning the cup body, a gray base row at the bottom, and cyan
  (8) "water" filling the bottom `water` rows of the cup interior.  Above the
  water sit `space` (1..4) rows of EMPTY cup (gray walls, no cyan inside).  The
  3x3 output encodes ONLY the scalar `space`:
      out[0][0]=cyan if space>0   out[0][1]=cyan if space>1
      out[0][2]=cyan if space>2   out[1][2]=cyan if space>3
  every other output cell is background (0).  (space>=1 always, so out[0][0] is
  always cyan.)

  `space` is recovered as a scalar with NO per-cell plane:
      cyanrows = #rows containing any cyan  (== water)
      grayrows = #rows containing any gray  (== space + water + 1, walls+base)
      space    = grayrows - cyanrows - 1.
  Both come from ONE fp32 row-presence reduction rowmax = ReduceMax(input,[3])
  ([1,10,30,1], 1200B — the dominant intermediate, irreducible since ReduceMax
  emits fp32), then a channel slice (cyan ch8 / gray ch5) + ReduceSum over rows.

  Output:  cyan_mask[i,j] = space >= T[i,j] on the 3x3 tally, with
      T = [[1,2,3],[99,99,4],[99,99,99]]   (99 = never-cyan positions),
  L = Where(cyan_mask, 8, 0) -> pad to 30x30 with sentinel 10 (off-grid: no
  channel lights up) -> output = Equal(L, arange[0..9]) free BOOL one-hot.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
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

    # ---- per-row presence over the FREE input (one fp32 reduction) ----------
    n("ReduceMax", ["input"], "rowmax", axes=[3], keepdims=1)  # [1,10,30,1] f32

    # slice cyan (ch8) and gray (ch5) per-row presence
    init("c8a", np.array([8], np.int64), np.int64)
    init("c8b", np.array([9], np.int64), np.int64)
    init("c5a", np.array([5], np.int64), np.int64)
    init("c5b", np.array([6], np.int64), np.int64)
    init("ax1", np.array([1], np.int64), np.int64)
    n("Slice", ["rowmax", "c8a", "c8b", "ax1"], "cyanrow")   # [1,1,30,1] f32
    n("Slice", ["rowmax", "c5a", "c5b", "ax1"], "grayrow")   # [1,1,30,1] f32

    # count rows with cyan / gray (rowmax is exactly 0/1, ReduceSum over rows)
    n("ReduceSum", ["cyanrow"], "cyancnt", axes=[2], keepdims=1)  # [1,1,1,1]
    n("ReduceSum", ["grayrow"], "graycnt", axes=[2], keepdims=1)  # [1,1,1,1]

    # space = grayrows - cyanrows - 1   (scalar fp32)
    n("Sub", ["graycnt", "cyancnt"], "gminc")
    init("one", np.array(1.0, np.float32).reshape(1, 1, 1, 1), np.float32)
    n("Sub", ["gminc", "one"], "space")                      # [1,1,1,1] f32

    # ---- 3x3 cyan tally: cyan iff space >= T[i,j] ---------------------------
    # cyan_mask = NOT(space < T)  (opset-11 has no GreaterOrEqual)
    T = np.array([[1, 2, 3], [99, 99, 4], [99, 99, 99]], np.float32).reshape(1, 1, 3, 3)
    init("T", T, np.float32)
    n("Less", ["space", "T"], "lt")                          # [1,1,3,3] bool
    n("Not", ["lt"], "cyan_mask")                            # [1,1,3,3] bool

    # L = Where(cyan_mask, 8, 0) on the 3x3 tally (uint8 index plane)
    init("u8val", np.array(8, np.uint8), np.uint8)
    init("u0val", np.array(0, np.uint8), np.uint8)
    n("Where", ["cyan_mask", "u8val", "u0val"], "L3")        # [1,1,3,3] u8

    # pad 3x3 -> 30x30 with sentinel 10 (off-grid: no channel lights up)
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - 3, 30 - 3], np.int64), np.int64)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L3", "padpads", "u10"], "L", mode="constant")  # [1,1,30,30] u8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                       # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task274", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
