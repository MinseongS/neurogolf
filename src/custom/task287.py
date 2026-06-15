"""Task 287 (ARC-AGI b8825c91): restore a 4-fold-symmetric pattern.

The grid is always 16x16 (2*size, size=8).  The true pattern is symmetric under
both the horizontal mirror c -> 15-c and the vertical mirror r -> 15-r (hence the
full 4-fold dihedral symmetry).  In the input a few rectangular regions are
overwritten with yellow(4); the underlying pattern never contains yellow.  The
output restores every cell to the unique non-yellow colour shared by its four
symmetric partners.

Floor-break (label map + final Equal, 16x16 canvas):
  idx[r,c]   = sum_k k * input[k]            (one-hot colour index, 0..9)
  v[r,c]     = idx if idx != 4 (yellow) else 0
  L16[r,c]   = max( v, flipV(v), flipH(v), flipVH(v) )   -> the true colour
  (exactly one of the four partners is the real colour; yellow maps to 0, so the
   max picks the real colour.)
  Pad L16 (16x16) to 30x30 with sentinel 10 (off-grid matches nothing), then
  output = Equal(L, arange[1,10,1,1]) into the free BOOL output (opset 11).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 16  # grid side (2*size)


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
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    # --- colour index idx = sum_k k*input[k] via a 1x1 Conv (mem 0) ----------
    # weight k on each channel, yellow(4) weighted 0 so masked cells read as 0.
    kw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    kw[0, 4, 0, 0] = 0.0
    init("kw", kw, np.float32)
    n("Conv", ["input", "kw"], "vfull")                        # [1,1,30,30] f32
    # slice top-left 16x16
    init("st", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("en", np.array([1, 1, S, S], np.int64), np.int64)
    init("ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["vfull", "st", "en", "ax"], "v", )             # [1,1,16,16] f32

    # --- four symmetric partners (flip about centre of the 16x16 canvas) -----
    init("rrev", np.arange(S - 1, -1, -1, dtype=np.int64), np.int64)  # 15..0
    init("crev", np.arange(S - 1, -1, -1, dtype=np.int64), np.int64)
    n("Gather", ["v", "rrev"], "vV", axis=2)                   # flip rows
    n("Gather", ["v", "crev"], "vH", axis=3)                   # flip cols
    n("Gather", ["vV", "crev"], "vVH", axis=3)                 # flip both

    n("Max", ["v", "vV"], "m1")
    n("Max", ["vH", "vVH"], "m2")
    n("Max", ["m1", "m2"], "vmax")                             # [1,1,16,16] f32
    n("Cast", ["vmax"], "L16", to=U8)                          # uint8 labels 0..9

    # --- pad to 30x30 with sentinel 10, then final Equal ---------------------
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - S, 30 - S], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L16", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30]
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task287", [x], [y], inits)
    return helper.make_model(
        g, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
