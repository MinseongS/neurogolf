"""Task 010 (ARC-AGI 08ed6ac7) — recolour gray bars by height-rank.

Rule (from the generator): 9x9 grid.  Four gray (colour 5) bars hang from the
bottom of the grid at columns {1,3,5,7} (column = order[bar]*2+1, order a
permutation of 0..3).  Each bar has a distinct height in 1..9.  The OUTPUT keeps
the identical bar shapes but recolours each bar by its height RANK: the tallest
bar -> colour 1, second tallest -> 2, ..., shortest -> 4.  Background stays 0.

Recovery (count-parametric, NO 30x30 plane):
  * Column heights = per-column gray-cell counts = Gather channel 5 of
    ReduceSum(input, axes=[2]) -> a [1,1,1,9] height vector h (cols 0..8).
    ReduceSum over rows of the FREE input is [1,10,1,30] (1200B) -- the only
    moderately-sized tensor; everything after is <=81 elem.
  * rank colour per column: cnt[i] = #cols strictly taller than i (pairwise
    Greater over the 9-vector, an 81-elem bool plane); rankcol = (cnt+1) only
    where the column has a bar (empty cols stay 0 automatically because the fill
    mask excludes them).
  * bar shape is fully determined by height (solid, bottom-anchored): a cell
    (r,c) is filled iff r + h[c] >= 9.  So the whole 9x9 label map is rebuilt
    from h with NO per-cell occupancy plane:
        L[r,c] = filled[r,c] ? rankcol[c] : 0   (0 = background colour).
  * Pad L 9x9 -> 30x30 with off-grid sentinel 10, then the FREE BOOL output =
    Equal(L, arange[1,10,1,1]).
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

    # ---- per-column gray heights (NO 30x30 plane) ----
    # row-sum of the gray channel folded into ONE no-pad Conv on the FREE input:
    # W[out=1,in=10,kh=30,kw=1] = 1.0 at channel 5 across all rows, else 0 ->
    # gh30[1,1,1,30] = number of gray cells per column (120B).
    cw = np.zeros((1, 10, 30, 1), np.float32); cw[0, 5, :, 0] = 1.0
    init("cw", cw, np.float32)
    n("Conv", ["input", "cw"], "gh30", kernel_shape=[30, 1])    # [1,1,1,30]
    cols = np.arange(9, dtype=np.int64)
    init("cols", cols, np.int64)
    n("Gather", ["gh30", "cols"], "ghcol", axis=3)              # [1,1,1,9]
    init("h9", np.array([9], np.int64), np.int64)
    n("Reshape", ["ghcol", "h9"], "hf32")                       # [9] heights fp32
    F16 = TensorProto.FLOAT16
    n("Cast", ["hf32"], "h", to=F16)                            # [9] heights fp16

    # ---- rank colour per column: cnt[i] = #cols strictly taller than i ----
    # all downstream [9,9] planes in fp16 (162B vs 324B); ints <=9 are fp16-exact.
    init("hi_s", np.array([9, 1], np.int64), np.int64)
    n("Reshape", ["h", "hi_s"], "hi")                           # [9,1] fp16
    init("hj_s", np.array([1, 9], np.int64), np.int64)
    n("Reshape", ["h", "hj_s"], "hj")                           # [1,9] fp16
    n("Greater", ["hj", "hi"], "tallerb")                       # [9,9] bool (h[j]>h[i])
    n("Cast", ["tallerb"], "tallerf", to=F16)                   # [9,9] fp16
    n("ReduceSum", ["tallerf"], "cnt", axes=[1], keepdims=0)    # [9] fp16
    init("one_f", np.array(1.0, np.float16), np.float16)
    n("Add", ["cnt", "one_f"], "rankcol")                       # [9] fp16 (1..4)

    # ---- fill mask: (r,c) filled iff r + h[c] >= 9 ----
    # rowramp r=0..8 as [9,1]; h as [1,9]; sum -> [9,9]; >= 9 (Not(Less(.,9))).
    rowramp = np.arange(9, dtype=np.float16).reshape(9, 1)
    init("rowramp", rowramp, np.float16)
    n("Reshape", ["h", "hj_s"], "hrow")                         # [1,9] fp16
    n("Add", ["rowramp", "hrow"], "rc")                         # [9,9] = r + h[c] fp16
    init("nine_f", np.array(9.0, np.float16), np.float16)
    n("Less", ["rc", "nine_f"], "emptyb")                       # [9,9] r+h<9 (NOT filled)
    n("Not", ["emptyb"], "filledb")                             # [9,9] filled

    # ---- label L[r,c] = filled ? rankcol[c] : 0 ----
    n("Reshape", ["rankcol", "hj_s"], "rankrow")                # [1,9] fp16
    init("zero_f", np.array(0.0, np.float16), np.float16)
    n("Where", ["filledb", "rankrow", "zero_f"], "Lf")          # [9,9] fp16 (broadcast)
    n("Cast", ["Lf"], "Lu8", to=U8)                             # [9,9] uint8
    init("L4s", np.array([1, 1, 9, 9], np.int64), np.int64)
    n("Reshape", ["Lu8", "L4s"], "L4")                          # [1,1,9,9]

    # ---- pad to 30x30 with off-grid sentinel 10, then Equal -> BOOL output ----
    init("u10", np.array(10, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 21, 21], np.int64), np.int64)
    n("Pad", ["L4", "pads", "u10"], "L", mode="constant")       # [1,1,30,30] uint8

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")                         # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task010", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
