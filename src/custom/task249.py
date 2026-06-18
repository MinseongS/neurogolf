"""Task 249 (a416b8f3): horizontal self-duplication of the grid.

Rule (from ARC-GEN generator):
  Input is a width x height grid (one-hot, top-left of 30x30 canvas).
  Output is a (2*width) x height grid: each input column c is copied to
  output columns c AND width+c.  i.e. out[:, i] = in[:, i mod_dup width] where
  out column i maps to input column m[i] = i (if i<W) else i-W.

  This holds for ALL output columns i in 0..29:
    - i < W       -> input col i            (in-grid, first copy)
    - W <= i < 2W -> input col i-W          (in-grid, second copy)
    - i >= 2W     -> input col i-W >= W     (OFF-grid input col = all-zero
                                             one-hot = correct empty output)
  So NO clip is needed (i-W in [0,29] always since W>=3, i<=29).
  Rows pass through unchanged (Gather is on the width axis only); off-grid
  rows are all-zero in the input -> all-zero in the output.

Encoding (Tier S, pure spatial Gather, output is FREE):
  W (grid width) = number of occupied columns = sum over columns of the
  per-column occupancy ReduceMax(input, axes=[channels, rows]).  Off-grid
  columns are all-zero one-hot so occupancy=0; in-grid columns (incl bg
  color-0 cells, which set channel 0 = 1) have occupancy=1.
  Column index map m = Where(arange<W, arange, arange-W) built in fp16 to
  halve the length-30 working vectors, then Cast->int32 (not int64, halves
  the index plane) and Gather(input, m, axis=3) straight into the free output.

  Dominant intermediate: the [1,1,1,30] fp32 column-occupancy plane (120B) is
  irreducible (ReduceMax inherits the fp32 input dtype; casting the input to
  fp16 would cost 18000B).  Everything else is a length-30 fp16/int32 vector.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, outs, name, **attrs):
        nodes.append(helper.make_node(op, ins, outs, name=name, **attrs))
        return outs[0]

    # ---- constants ----
    init("arange", np.arange(30, dtype=np.float16).reshape(30), np.float16)
    init("sum_ax", np.array([1, 2, 3], np.int64), np.int64)

    # ---- W = number of occupied columns (fp32 scalar) ----
    # per-column occupancy: 1 if any channel/row has a pixel, else 0
    n("ReduceMax", ["input"], ["colocc"], "colocc",
      axes=[1, 2], keepdims=1)                       # [1,1,1,30] fp32
    n("ReduceSum", ["colocc", "sum_ax"], ["W32"], "W",
      keepdims=0)                                    # [1] fp32 -> width (axis0 kept)
    n("Cast", ["W32"], ["W16"], "castW", to=onnx.TensorProto.FLOAT16)  # [1] fp16

    # ---- column index map m[i] = i if i<W else i-W (fp16 length-30) ----
    n("Less", ["arange", "W16"], ["lt"], "lt")        # [30] bool: i<W
    n("Sub", ["arange", "W16"], ["shifted"], "sub")   # [30] fp16: i-W
    n("Where", ["lt", "arange", "shifted"], ["m16"], "where")  # [30] fp16
    n("Cast", ["m16"], ["m"], "castM", to=onnx.TensorProto.INT32)  # [30] int32

    # ---- Gather columns straight into the free output ----
    n("Gather", ["input", "m"], ["output"], "gather", axis=3)

    graph = helper.make_graph(
        nodes, "task249",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = IR_VERSION
    return model
