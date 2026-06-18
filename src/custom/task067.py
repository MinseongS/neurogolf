"""task067 (ARC-AGI 2dee498d) — output = the left size x size block.

Rule (from the ARC-GEN generator, verified fresh):
  The generator builds an input grid of shape (size) rows x (3*size) cols, with
  three side-by-side size x size blocks:
    block 0 (cols 0..size-1)        = colors[r][c]
    block 1 (cols size..2size-1)    = colors, optionally vertically flipped
    block 2 (cols 2size..3size-1)   = colors[r][c]
  The OUTPUT is the size x size grid colors[r][c] == block 0 == block 2.
  So output is exactly the LEFT size columns of the input, cropped to size rows.

  Encoding: input/output are one-hot [1,10,30,30].  Within the grid EVERY cell
  (including colour-0 cells) sets channel 0 = 1, so an in-grid column is always
  "occupied"; off-grid columns (>= 3*size) are all-zero.  Output rows >= size are
  off-grid in the output (all-zero), and since the input is also all-zero there,
  a plain column-crop of the input reproduces the output exactly.

Construction (pure copy / crop, all-scalar except one bool keep-row):
  total cells = ReduceSum(input, axes=[1,2,3]) = 3 * size^2  (one [1,1,1,1] scalar).
  size^2 = total / 3.  We keep column c iff c < size, i.e. iff c^2 < size^2 (both
  >= 0).  Compare a CONSTANT squared column ramp [0,1,4,9,...,29^2] against size^2
  (with a +0.5 epsilon to avoid float ties) -> keep_b [1,1,1,30] bool.
  output = Where(keep_b, input, 0).  No per-column occupancy plane, no cumsum conv;
  the only full-width intermediate is the 30-byte bool keep mask.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    B = TensorProto.BOOL

    # total cells = 3 * size^2  (scalar)
    n("ReduceSum", ["input"], "total", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]
    init("three", np.array([[[[3.0]]]], np.float32), np.float32)
    n("Div", ["total", "three"], "size2")                          # [1,1,1,1] = size^2

    # constant squared column ramp [0,1,4,...,29^2] as [1,1,1,30]
    cramp2 = (np.arange(S, dtype=np.float32) ** 2).reshape(1, 1, 1, S)
    init("cramp2", cramp2, np.float32)
    # keep column c iff c^2 < size^2  (strict: c=size has c^2==size^2 -> dropped)
    n("Greater", ["size2", "cramp2"], "keep_b")                    # [1,1,1,30] bool

    init("zero", np.array(0.0, np.float32), np.float32)
    n("Where", ["keep_b", "input", "zero"], "output")              # [1,10,30,30]

    x = helper.make_tensor_value_info("input", F, [1, 10, S, S])
    y = helper.make_tensor_value_info("output", F, [1, 10, S, S])
    graph = helper.make_graph(nodes, "task067", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
