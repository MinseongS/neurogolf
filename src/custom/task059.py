"""Task 059 (ARC-AGI 29623171) — fill the busiest mini-cell(s) solid.

Rule (from the generator): an 11x11 grid is a 3x3 array of 3x3 mini-cells
separated by a gray (5) "hollywood squares" frame at rows/cols 3 and 7.  Each
mini-cell holds 0..max coloured pixels in a single per-instance colour `c`
(c != gray).  The OUTPUT keeps the same gray frame; every mini-cell whose
coloured-pixel COUNT equals the maximum count (over the 9 cells; ties allowed)
is filled SOLID with `c`; all other cells become background (0).

Recovery (closed-form, separable — no flood-fill / argmax op):
  * colf = colour-index plane on the 11x11 region (1x1 weighted sum of the FREE
    10-ch input).
  * Gather the 9 interior rows/cols {0,1,2,4,5,6,8,9,10} -> a 9x9 gap-free
    "compact" grid of the mini-cells (frame rows/cols dropped).
  * colour c = ReduceMax(compact)  (interior has no gray, all coloured = c).
  * per-block count = 3x3-stride sum of (compact>0) -> [1,3,3]; max = ReduceMax;
    selected block(s) = Equal(count, max)  (ties kept).
  * label L[r,c] on 11x11 = frame?5 : (selected[block(r,c)] ? c : 0); pad to
    30x30 with off-grid sentinel 10; free BOOL output = Equal(L, arange).

Memory: the lone full-canvas tensor is the 11x11-region colf conv output
(fp32 3600B before the interior Gather); everything after is <=484B.
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

    # ---- colour-index plane via a 1x1 conv on the FREE input ----
    kw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("kw", kw, np.float32)
    n("Conv", ["input", "kw"], "colf", kernel_shape=[1, 1])  # [1,1,30,30] fp32

    # ---- gather interior rows/cols -> 9x9 compact grid ----
    keep = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10], np.int64)
    init("keep", keep, np.int64)
    n("Gather", ["colf", "keep"], "cr", axis=2)          # [1,1,9,30] fp32
    n("Gather", ["cr", "keep"], "compact", axis=3)       # [1,1,9,9] fp32

    # ---- colour scalar c = max over the (frame-free) interior ----
    n("ReduceMax", ["compact"], "cmax", axes=[2, 3], keepdims=1)  # [1,1,1,1]

    # ---- per-block coloured-pixel count (3x3 blocks) ----
    init("zero", np.array(0.0, np.float32), np.float32)
    n("Greater", ["compact", "zero"], "occb")            # [1,1,9,9] bool
    n("Cast", ["occb"], "occf", to=F32)                  # [1,1,9,9] fp32
    init("rs5", np.array([1, 3, 3, 3, 3], np.int64), np.int64)
    n("Reshape", ["occf", "rs5"], "occ5")                # [1,3,3,3,3] (R,mr,C,mc)
    n("ReduceSum", ["occ5"], "cnt", axes=[2, 4], keepdims=0)  # [1,3,3] (R,C)

    # ---- selected blocks: count == max (ties kept) ----
    n("ReduceMax", ["cnt"], "mx", axes=[1, 2], keepdims=1)   # [1,1,1]
    n("Equal", ["cnt", "mx"], "selb")                    # [1,3,3] bool
    n("Cast", ["selb"], "self", to=F32)                  # [1,3,3] fp32
    init("c9", np.array([9], np.int64), np.int64)
    n("Reshape", ["self", "c9"], "selflat")              # [9] fp32 (1.0 if max)

    # ---- 11x11 colour-index label ----
    r = np.arange(11)
    c = np.arange(11)
    rr, cc = np.meshgrid(r, c, indexing="ij")            # [11,11]
    blkidx = ((rr // 4) * 3 + (cc // 4)).astype(np.int64)  # [11,11] in 0..8
    init("blkidx", blkidx, np.int64)
    n("Gather", ["selflat", "blkidx"], "selmap", axis=0)  # [11,11] fp32 {0,1}

    # cval = c * selmap  (c on selected interior cells, 0 elsewhere)
    init("scal", np.array([1], np.int64), np.int64)
    n("Reshape", ["cmax", "scal"], "cmax1")              # [1] scalar-ish
    n("Mul", ["selmap", "cmax1"], "cval")                # [11,11] fp32

    # frame (gray=5) overrides everything
    frame = ((rr == 3) | (rr == 7) | (cc == 3) | (cc == 7))
    init("frame", frame, np.bool_)
    n("Cast", ["cval"], "cvalu8", to=U8)                 # [11,11] uint8 (c or 0)
    init("u5", np.array(5, np.uint8), np.uint8)
    n("Where", ["frame", "u5", "cvalu8"], "L11")         # [11,11] uint8
    init("Lshape", np.array([1, 1, 11, 11], np.int64), np.int64)
    n("Reshape", ["L11", "Lshape"], "L114")              # [1,1,11,11]

    # pad to 30x30 with off-grid sentinel 10 (matches no channel 0..9)
    init("u10", np.array(10, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 19, 19], np.int64), np.int64)
    n("Pad", ["L114", "pads", "u10"], "L", mode="constant")  # [1,1,30,30] uint8

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")                  # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task059", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
