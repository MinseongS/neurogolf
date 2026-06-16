"""Task 011 (ARC-AGI 09629e4f) — pick the odd mini-cell, upscale it 3x.

Rule (from the generator): the 11x11 grid is a 3x3 array of 3x3 mini-cells
separated by a gray (5) "hollywood squares" frame at rows/cols 3 and 7.  Each
mini-cell holds some rainbow-coloured pixels (colours in {2,3,4,6,8}); exactly
ONE mini-cell ("chosen") has 4 coloured pixels, every other has 5.  The OUTPUT
keeps the same gray frame, and fills each output mini-cell block (mr,mc) SOLID
with the colour of the chosen cell's pixel at interior position (mr,mc) (bg 0 if
that interior cell is empty).  I.e. output = the chosen 3x3 mini-cell upscaled
3x onto the frame.

Recovery (offset-free, generalising — no flood-fill, no argmax op):
  * colf = colour-index plane on the 11x11 region (free input, 1x1 weighted sum).
  * Gather the 9 interior rows/cols {0,1,2,4,5,6,8,9,10} -> a 9x9 "compact" grid
    = the 9 mini-cells packed gap-free, compact[R*3+mr][C*3+mc]=cell(R,C)[mr,mc].
  * per-block count = 3x3-stride sum of (compact>0); chosen block = count<5
    (exactly one cell has 4, the rest 5 -> unique, no ReduceMin/ArgMax needed).
  * chosen cell content = Sum_{R,C} sel[R,C]*compact[block]  ([1,1,3,3] fp32).
  * output index map L[r,c] = (frame ? 5 : cellflat[(r//4)*3 + (c//4)]) on 11x11,
    padded to 30x30 with off-grid sentinel 10; free BOOL output = Equal(L,arange).

Memory: the lone ~30x30-ish tensor is the 11x11 colf slice (fp32 484B); every
other intermediate is <=200B (9x9 compact, 3x3 counts, 11x11 uint8 label).
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
    # colf = sum_k k*input_k.  Conv weight [1,10,1,1] = [0,1,...,9].
    # Output [1,1,30,30] fp32 (3600B) -- the lone full-canvas tensor.
    kw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("kw", kw, np.float32)
    n("Conv", ["input", "kw"], "colf", kernel_shape=[1, 1])  # [1,1,30,30] fp32

    # ---- gather interior rows/cols -> 9x9 compact grid ----
    keep = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10], np.int64)
    init("keep", keep, np.int64)
    n("Gather", ["colf", "keep"], "cr", axis=2)          # [1,1,9,30]
    n("Gather", ["cr", "keep"], "compact", axis=3)       # [1,1,9,9] fp32

    # ---- per-block coloured-pixel count (3x3 blocks) ----
    init("zero", np.array(0.0, np.float32), np.float32)
    n("Greater", ["compact", "zero"], "occb")            # [1,1,9,9] bool
    n("Cast", ["occb"], "occf", to=F32)                  # [1,1,9,9] fp32
    # sum each 3x3 block: reshape [1,1,3,3,3,3] (R,mr,C,mc) then reduce mr,mc.
    init("rs6", np.array([1, 3, 3, 3, 3], np.int64), np.int64)
    n("Reshape", ["occf", "rs6"], "occ5")                # [1,3,3,3,3] (R,mr,C,mc)
    n("ReduceSum", ["occ5"], "cnt", axes=[2, 4], keepdims=0)  # [1,3,3] (R,C)

    # ---- chosen block selector: count < 5 (unique cell with 4) ----
    init("five", np.array(5.0, np.float32), np.float32)
    n("Less", ["cnt", "five"], "selb")                   # [1,3,3] bool
    n("Cast", ["selb"], "self", to=F32)                  # [1,3,3] fp32

    # ---- chosen cell content = sum_{R,C} sel[R,C]*compact[block] ----
    # compact as (R,mr,C,mc); multiply by sel broadcast (R,1,C,1), sum over R,C.
    n("Reshape", ["compact", "rs6"], "cmp5")             # [1,3,3,3,3] (R,mr,C,mc)
    init("selrs", np.array([1, 3, 1, 3, 1], np.int64), np.int64)
    n("Reshape", ["self", "selrs"], "sel5")              # [1,3,1,3,1]
    n("Mul", ["cmp5", "sel5"], "cmpsel")                 # [1,3,3,3,3]
    n("ReduceSum", ["cmpsel"], "cell", axes=[1, 3], keepdims=0)  # [1,3,3] (mr,mc)
    init("c9", np.array([9], np.int64), np.int64)
    n("Reshape", ["cell", "c9"], "cellflat")             # [9] fp32 chosen cell

    # ---- build 11x11 colour-index label ----
    # block content B[r,c] = cellflat[(r//4)*3 + (c//4)]  (gather const index map)
    r = np.arange(11)
    c = np.arange(11)
    rr, cc = np.meshgrid(r, c, indexing="ij")            # [11,11]
    blkidx = ((rr // 4) * 3 + (cc // 4)).astype(np.int64)  # [11,11] in 0..8
    init("blkidx", blkidx, np.int64)
    n("Gather", ["cellflat", "blkidx"], "blkval", axis=0)  # [11,11] fp32
    n("Cast", ["blkval"], "blku8", to=U8)                # [11,11] uint8

    # frame mask: rows/cols 3 or 7 are gray (5)
    frame = ((rr == 3) | (rr == 7) | (cc == 3) | (cc == 7))
    init("frame", frame, np.bool_)
    init("u5", np.array(5, np.uint8), np.uint8)
    n("Where", ["frame", "u5", "blku8"], "L11")          # [11,11] uint8
    init("Lshape", np.array([1, 1, 11, 11], np.int64), np.int64)
    n("Reshape", ["L11", "Lshape"], "L114")              # [1,1,11,11]

    # pad to 30x30 with off-grid sentinel 10 (matches no channel 0..9 -> all-off)
    init("u10", np.array(10, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 19, 19], np.int64), np.int64)
    n("Pad", ["L114", "pads", "u10"], "L", mode="constant")  # [1,1,30,30] uint8

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")                  # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task011", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
