"""Task 155 (68b16354): vertical flip of the grid about its own centre.

Rule (from ARC-GEN generator):
  The grid occupies the top-left size x size corner (size in 4..8).  The output
  is the input with its ROWS reversed within the grid:
      output[size-1-r][c] = input[r][c]    for r,c in [0,size)
  Everything off the grid is background (0) on both sides.  This is a pure
  spatial copy / row permutation -- the one-hot channel content of each cell is
  carried unchanged, so it is a Tier-S Gather (axis=2) by a flipped row index.

Encoding (Tier S, no per-cell plane):
  H = number of occupied rows = ReduceSum( (per-row occupancy) > 0 ) as a SCALAR.
  Build a 1-D row-index vector idx[30] (length-30, NOT a 30x30 broadcast plane):
      idx[r] = (H-1) - r   for r < H          (flip inside the grid)
      idx[r] = r           for r >= H          (off-grid rows stay; all bg)
  Gather(input, idx, axis=2) rearranges rows of the FREE input straight into the
  FREE output.  All index arithmetic runs in fp16 (exact for these small ints,
  2B vs 4B int32); only the final Gather index is cast to int32.  The dominant
  900B [1,1,30,30] broadcast plane of the public net is removed entirely.
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

    # ---- constants (fp16 arithmetic) ----
    init("rng", np.arange(30, dtype=np.float16), np.float16)   # [30] row indices
    init("one", np.array(1.0, np.float16), np.float16)

    # ---- H = grid size (scalar) ----
    # The grid is COMPLETELY filled (every cell of the size x size corner is a
    # non-zero colour), so the total pixel count = size^2.  Reduce ALL axes to a
    # single scalar (4B) -> H = sqrt(count).  fp16 sqrt of {16,25,36,49,64} is
    # exact, avoiding any per-row [1,30] vector.
    n("ReduceSum", ["input"], "cnt", axes=[1, 2, 3], keepdims=0)  # [1] f32 = size^2
    n("Cast", ["cnt"], "cnt16", to=TensorProto.FLOAT16)        # [1] f16
    n("Sqrt", ["cnt16"], "H")                                  # [1] f16 = grid size

    # ---- 1-D flipped row index vector (fp16) ----
    n("Less", ["rng", "H"], "ingrid")                          # [30] bool r<H
    n("Sub", ["H", "one"], "Hm1")                              # [1] = H-1
    n("Sub", ["Hm1", "rng"], "flip")                           # [30] = H-1-r
    n("Where", ["ingrid", "flip", "rng"], "idxf")              # [30] f16
    n("Cast", ["idxf"], "idx", to=TensorProto.INT32)           # [30] int32

    # ---- Gather rows of the free input into the free output ----
    n("Gather", ["input", "idx"], "output", axis=2)

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
