"""Task 150 (67a3c6ac): horizontal mirror of a square grid (reverse each row).

Rule (from ARC-GEN generator):
  A square grid of side `size` (3..9), every cell coloured from {6,2,1,7}
  (never colour 0), sits at the top-left corner of the 30x30 canvas.  Off-grid
  cells are ALL-ZERO (no channel set).  The output reverses every row:
      out[:,:,:,c] = in[:,:,:, size-1-c]   for c < size
      out[:,:,:,c] = 0 (all-zero)          for c >= size
  This is a PURE spatial permutation along axis 3 (columns) -> ONE Gather whose
  output IS the free graph output (Tier S, the result plane is free).

Encoding (no full-grid plane materialises; only one length-30 index vector):
  total  = ReduceSum(input)        scalar = size*size (grid fully filled, every
           cell a nonzero colour) -> the ONLY reduction of the fp32 input, a
           single scalar (4B) instead of a [30] column profile.
  size   = Sqrt(total)             scalar grid side (exact for size 3..9).
  idx[c] = size-1-c   (NO clamp needed: for c>=size this is negative and ONNX
           Gather wraps idx+30, which lands on columns [size..29] -- all off-grid
           and therefore all-zero, so the output zero-fills automatically).
  output = Gather(input, idx, axis=3).
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

    # ---- constants (integer index arithmetic) ----
    init("arange", np.arange(30, dtype=np.int32), np.int32)   # [30] int32
    init("one", np.array(1, np.int32), np.int32)

    # ---- grid side `size` from total pixel count (= size*size) ----
    n("ReduceSum", ["input"], "total", keepdims=0)   # scalar fp32 = size*size
    n("Sqrt", ["total"], "sizef")                    # scalar fp32 = size
    n("Cast", ["sizef"], "size", to=TensorProto.INT32)   # scalar int32

    # ---- reversed column-index vector (size-1-c; negatives wrap to off-grid) ----
    n("Sub", ["size", "one"], "sizem1")          # scalar  (size-1)
    n("Sub", ["sizem1", "arange"], "idx")         # [30] int32  (size-1-c)

    # gather along columns (axis 3) -> free output
    n("Gather", ["input", "idx"], "output", axis=3)

    graph = helper.make_graph(nodes, "task150", [
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    ], [
        helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    ], inits)

    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = IR_VERSION
    return model
