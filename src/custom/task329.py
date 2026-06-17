"""Task 329 (d23f8c26): keep only the middle column of a square grid.

Rule (from ARC-GEN generator, verified):
  Input is a square SIZE x SIZE grid (SIZE odd, in {3,5,7,9}) anchored at the
  top-left corner of the 30x30 canvas; every in-grid cell is either background
  (0) or a random colour.  Output is the same grid with EVERY cell zeroed
  EXCEPT the middle column (c == SIZE//2), which copies the input verbatim.
  Off-grid cells are background (all-channels-off in the one-hot embedding).

Key structural facts:
  * In the 10-channel one-hot embedding an OFF-GRID cell has ALL channels 0
    (the harness only sets cells inside the grid).  Every IN-GRID cell sets
    exactly one channel (ch0 for background).  Therefore:
        in-grid column  <=>  some channel is on in that column
        in-grid row     <=>  some channel is on in that row
    and SIZE = (number of in-grid columns).  Verified 0 mismatches / 60000.
  * mid = floor(SIZE / 2).

Encoding (spatial COPY routed into the FREE Where output, no colour plane):
  colany[c] = ReduceMax(input, axes=[1,2])  -> [1,1,1,30]  (1 iff col in-grid)
  rowany[r] = ReduceMax(input, axes=[1,3])  -> [1,1,30,1]  (1 iff row in-grid)
  SIZE      = ReduceSum(colany, axes=[3])   -> [1,1,1,1]
  mid       = floor(SIZE / 2)               -> [1,1,1,1]
  fill[r,c] = rowany[r] AND (c != mid) AND colany[c]   -> [1,1,30,30] bool
  output    = Where(fill, bg_onehot[1,10,1,1], input)
              at non-mid in-grid cells -> background one-hot (ch0=1)
              at mid col (fill False, in-grid) -> input copied verbatim
              off-grid (fill False) -> input == all-zero -> correct.

  The only ~900B intermediate is the bool fill plane; the rest is 1-D (<=120B)
  or scalar.  No [1,10,30,30] colour plane is ever materialised.
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

    F = TensorProto.FLOAT
    B = TensorProto.BOOL

    init("colramp", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30),
         np.float32)
    init("half", np.array(0.5, np.float32), np.float32)
    init("bg_onehot",
         np.array([1] + [0] * 9, dtype=np.float32).reshape(1, 10, 1, 1),
         np.float32)

    # per-axis in-grid occupancy (1 iff that row/col lies inside the grid)
    n("ReduceMax", ["input"], "colany", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    n("ReduceMax", ["input"], "rowany", axes=[1, 3], keepdims=1)  # [1,1,30,1]

    # SIZE and middle-column index
    n("ReduceSum", ["colany"], "size", axes=[3], keepdims=1)      # [1,1,1,1]
    n("Mul", ["size", "half"], "size_h")
    n("Floor", ["size_h"], "mid")                                 # floor(SIZE/2)

    # fill condition = non-mid AND in-grid (row and col both in-grid)
    n("Equal", ["colramp", "mid"], "is_mid")                      # [1,1,1,30] bool
    n("Not", ["is_mid"], "not_mid")
    n("Greater", ["colany", "half"], "colin")                     # [1,1,1,30] bool
    n("Greater", ["rowany", "half"], "rowin")                     # [1,1,30,1] bool
    n("And", ["not_mid", "colin"], "fillcol")                     # [1,1,1,30] bool
    n("And", ["rowin", "fillcol"], "fill")                        # [1,1,30,30] bool

    # route the 10-channel expansion into the FREE output
    n("Where", ["fill", "bg_onehot", "input"], "output")          # [1,10,30,30] f32

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task329", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
