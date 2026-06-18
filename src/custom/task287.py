"""Task 287 (ARC-AGI b8825c91): restore a 4-fold-symmetric pattern.

The grid is always 16x16 (2*size, size=8).  The true pattern is symmetric under
both the horizontal mirror c -> 15-c and the vertical mirror r -> 15-r (hence the
full 4-fold dihedral symmetry).  In the input a few small rectangular regions are
overwritten with yellow(4); the underlying pattern never contains yellow.  The
output restores every cell to the unique non-yellow colour shared by its four
symmetric partners.

Floor-break (label map + final Equal, 16x16 canvas, uint8 working planes):
  vf30 = sum_k k*input[k]  via a 1x1 Conv whose yellow channel weight is 0, so
         cutout cells read 0 (a low sentinel). The conv runs over the FULL free
         input (slicing the 10-ch input first would cost 10240B); the single-
         channel result is cast to uint8 (30x30, 900B) then sliced to the active
         16x16 corner (256B) -- cheaper than an fp32 slice (1024B).
  Orbit max is done as TWO separable mirror folds (max is associative):
  s1 = max(v, flipV(v)) (V-symmetric), L16 = max(s1, flipH(s1)) (fully symmetric),
  each max = Where(Greater(a,b),a,b) (uint8 256B/plane; ORT has no uint8 Max).
  Exactly one of a cell's four partners is the real colour, yellow -> 0, so the
  max recovers the true colour.
  Pad the 16x16 uint8 label to 30x30 with sentinel 10 (off-grid matches nothing),
  then output = Equal(L, arange[1,10,1,1]) into the free BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 16  # grid side (2*size)


def build(task):
    inits, nodes, vinfos = [], [], []

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # --- colour index v = sum_k k*input[k] via a 1x1 Conv (yellow weight 0) ----
    # Conv on the FULL free input (10-ch slice would cost 10240B), then slice the
    # cheap single-channel result down to the active 16x16 corner.
    kw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    kw[0, 4, 0, 0] = 0.0
    init("kw", kw, np.float32)
    n("Conv", ["input", "kw"], "vf30")
    vi("vf30", F, [1, 1, 30, 30])
    n("Cast", ["vf30"], "v30", to=U8)            # 30x30 uint8 (900) < fp32 slice
    vi("v30", U8, [1, 1, 30, 30])
    init("st", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("en", np.array([1, 1, S, S], np.int64), np.int64)
    init("ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["v30", "st", "en", "ax"], "v")
    vi("v", U8, [1, 1, S, S])

    # --- orbit max via two separable mirror folds (max is associative) --------
    # fold vertically (v vs flipV), then fold the result horizontally.
    init("rev", np.arange(S - 1, -1, -1, dtype=np.int64), np.int64)  # 15..0

    def umax(a, b, gname, out):
        n("Greater", [a, b], gname)
        vi(gname, B, [1, 1, S, S])
        n("Where", [gname, a, b], out)
        vi(out, U8, [1, 1, S, S])
        return out

    n("Gather", ["v", "rev"], "vV", axis=2)        # flip rows
    vi("vV", U8, [1, 1, S, S])
    umax("v", "vV", "g1", "s1")                     # V-symmetric
    n("Gather", ["s1", "rev"], "s1H", axis=3)      # flip cols of s1
    vi("s1H", U8, [1, 1, S, S])
    umax("s1", "s1H", "g2", "L16")                  # fully symmetric

    # --- pad to 30x30 with sentinel 10, then final Equal -> free BOOL output ----
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - S, 30 - S], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L16", "padpads", "padval"], "L", mode="constant")
    vi("L", U8, [1, 1, 30, 30])
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task287", [x], [y], inits, value_info=vinfos)
    return helper.make_model(
        g, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
