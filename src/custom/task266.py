"""Task 266 (ARC-AGI a9f96cdd) — red dot -> 4 fixed-colour diagonal stamps.

Rule (from the generator): the input is ALWAYS a 3x5 grid (height=3, width=5)
that is all black (0) except a SINGLE red (2) pixel at (row, col).  The output
is the same 3x5 grid that stamps four FIXED colours at the diagonal neighbours
of the red pixel (each only if it lands inside the 3x5 grid):

    green (3) at (row-1, col-1)
    pink  (6) at (row-1, col+1)
    cyan  (8) at (row+1, col-1)
    orange(7) at (row+1, col+1)

Everything else inside the grid is black (channel 0 = 1); everything OUTSIDE
the 3x5 grid is all-zero (no channel set) because convert_to_numpy only fills
the 3x5 cells.

This is a pure SHIFT-AND-RECOLOUR of the single red channel; each stamp colour
is the red channel shifted by one diagonal.  Encoding (SMALL-ACTIVE-CANVAS +
colour-index-plane escape, all intermediates tiny):

  1. Slice input channel 2 (red) over the 3x5 active region -> red [1,1,3,5]
     fp32 (60B).  Off-grid stamps drop automatically: we only ever produce a
     3x5 result and the trailing Pad zero-fills the rest of the canvas.

  2. ONE 3x3 SAME-pad Conv with a fixed [1,1,3,3] kernel turns red into a single
     COLOUR-INDEX plane lab[1,1,3,5] (60B):
         lab[R,C] = 3*red[R+1,C+1]   (green)
                  + 6*red[R+1,C-1]   (pink)
                  + 7*red[R-1,C-1]   (orange)
                  + 8*red[R-1,C+1]   (cyan)
     ONNX Conv is cross-correlation with pad=1, so out[R,C] = sum_{dr,dc}
     W[0,0,dr,dc]*red[R+dr-1,C+dc-1].  A green stamp at (R,C) needs red at
     (R+1,C+1) -> dr=2,dc=2 -> W[2,2]=3, etc.  The four diagonal cells are
     distinct so the contributions never collide; every non-stamp cell stays 0
     (= black background).

  3. Equal(lab, arange[1,10,1,1]) expands the index plane into the 10-channel
     one-hot in the SMALL 3x5 space -> [1,10,3,5] bool (150B).  bg cells (lab=0)
     light channel 0; stamp cells light their colour channel.

  4. Pad [1,10,3,5] -> [1,10,30,30] (constant 0) -- this Pad IS the graph output,
     so no 30x30 carrier plane is ever materialised, and off-grid cells become
     all-zero, matching the empty target.

mem ~270B, params ~few dozen.  The harness scores (out>0) on the bool output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, np_dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=np_dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- 1. slice input channel 2 (red), 3x5 active region ----
    init("starts", np.array([2, 0, 0], np.int64), np.int64)   # ch2, r0, c0
    init("ends", np.array([3, 3, 5], np.int64), np.int64)     # ch3(excl), r3, c5
    init("axes_s", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "starts", "ends", "axes_s"], "red")  # [1,1,3,5] fp32

    # ---- 2. colour-index conv: lab[1,1,3,5] ----
    W = np.zeros((1, 1, 3, 3), np.float32)
    W[0, 0, 2, 2] = 3.0   # green  : red at (+1,+1)
    W[0, 0, 2, 0] = 6.0   # pink   : red at (+1,-1)
    W[0, 0, 0, 0] = 7.0   # orange : red at (-1,-1)
    W[0, 0, 0, 2] = 8.0   # cyan   : red at (-1,+1)
    init("W", W, np.float32)
    n("Conv", ["red", "W"], "lab",
      kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])  # [1,1,3,5]

    # ---- 3. expand to 10-channel one-hot in the small 3x5 space ----
    arange = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("arange", arange, np.float32)
    n("Equal", ["lab", "arange"], "oneh")                     # [1,10,3,5] bool

    # ---- 4. Pad [1,10,3,5] -> [1,10,30,30] (this IS output) ----
    # ORT Pad rejects bool, so pad in uint8: cast bool -> uint8, Pad, declare
    # the output uint8 (harness scores out>0 identically).
    n("Cast", ["oneh"], "oneh_u8", to=TensorProto.UINT8)      # [1,10,3,5] u8
    pads = np.array([0, 0, 0, 0, 0, 0, 27, 25], np.int64)     # H 3->30, W 5->30
    init("pads", pads, np.int64)
    init("zerov", np.array([0], np.uint8), np.uint8)
    n("Pad", ["oneh_u8", "pads", "zerov"], "output", mode="constant")

    in_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    out_vi = helper.make_tensor_value_info("output", TensorProto.UINT8,
                                           [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task266", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
