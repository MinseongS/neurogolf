"""Task 047 (ARC-AGI 23581191) — two seed pixels -> full crosshairs + red crossings.

Rule (from the generator, grid is always 9x9):
  * input: a single cyan(8) pixel at (r0,c0) and a single orange(7) pixel at
    (r1,c1), with r0!=r1, c0!=c1, all in 1..size-3.
  * output: column c0 painted cyan, column c1 painted orange, row r0 painted
    cyan, row r1 painted orange (rows overwrite columns), then the two cross
    intersections (r0,c1) and (r1,c0) are painted red(2).

Closed-form per-cell colour.  Let rowval = 8 at r0 / 7 at r1 / 0 else, and
colval = 8 at c0 / 7 at c1 / 0 else.  Because rows/cols are distinct, the final
colour is a PURE FUNCTION of the single scalar s = rowval + colval:
    s : 0 -> 0(bg)   7 -> 7(orange)   8 -> 8(cyan)
       14 -> 7       15 -> 2(red)     16 -> 8
(8+8 cyan crossing -> cyan; 7+7 orange crossing -> orange; 8+7 / 7+8 -> red;
single lines keep their own colour).  So one Gather through a 17-entry table
maps s straight to the colour index -- no Where/cross/equal planes at all.

Recovery (no flood-fill, no argmax, work entirely in the 9x9 active region):
  * r0/c0 = ReduceMax of the cyan(8) channel over cols/rows  -> [1,1,9,1]/[1,1,1,9]
  * r1/c1 = ReduceMax of the orange(7) channel               -> same
  * rowval = 8*R0 + 7*R1 ; colval = 8*C0 + 7*C1 (fp16, exact integers)
  * s = rowval + colval (broadcasts to [1,1,9,9]); L = table[s] (uint8); then
    Equal(L, arange[1,10,1,1]) -> [1,10,9,9] BOOL, Pad -> [1,10,30,30] output.

Memory: the lone notable tensor is the [1,10,9,9] bool pre-pad (810B); every
other intermediate is <=162B.  No full 30x30 plane is ever materialised.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
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

    # ---- slice the cyan(8) and orange(7) channels to the 9x9 active region ----
    init("s89", np.array([8, 0, 0], np.int64), np.int64)
    init("e89", np.array([9, 9, 9], np.int64), np.int64)
    init("ax89", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "s89", "e89", "ax89"], "cyan")    # [1,1,9,9] fp32

    init("s78", np.array([7, 0, 0], np.int64), np.int64)
    init("e78", np.array([8, 9, 9], np.int64), np.int64)
    n("Slice", ["input", "s78", "e78", "ax89"], "orng")    # [1,1,9,9] fp32

    # ---- row/col seed indicators (single 1 each), fp16 ----
    n("ReduceMax", ["cyan"], "R0f", axes=[3], keepdims=1)  # [1,1,9,1] fp32
    n("ReduceMax", ["cyan"], "C0f", axes=[2], keepdims=1)  # [1,1,1,9] fp32
    n("ReduceMax", ["orng"], "R1f", axes=[3], keepdims=1)
    n("ReduceMax", ["orng"], "C1f", axes=[2], keepdims=1)
    for s in ("R0f", "C0f", "R1f", "C1f"):
        n("Cast", [s], s[:-1] + "h", to=F16)               # fp16 indicators

    # ---- rowval = 8*R0 + 7*R1 ; colval = 8*C0 + 7*C1 ----
    init("w8", np.array(8.0, np.float16), np.float16)
    init("w7", np.array(7.0, np.float16), np.float16)
    n("Mul", ["R0h", "w8"], "rv0")
    n("Mul", ["R1h", "w7"], "rv1")
    n("Add", ["rv0", "rv1"], "rowval")                     # [1,1,9,1] fp16
    n("Mul", ["C0h", "w8"], "cv0")
    n("Mul", ["C1h", "w7"], "cv1")
    n("Add", ["cv0", "cv1"], "colval")                     # [1,1,1,9] fp16

    # ---- s = rowval + colval, then table lookup s -> colour index ----
    n("Add", ["rowval", "colval"], "s16")                  # [1,1,9,9] fp16
    n("Cast", ["s16"], "sidx", to=TensorProto.INT32)       # [1,1,9,9] int32 idx (0..16)
    table = np.zeros(17, np.uint8)
    table[7] = 7; table[8] = 8; table[14] = 7; table[15] = 2; table[16] = 8
    init("tbl", table, np.uint8)
    n("Gather", ["tbl", "sidx"], "L", axis=0)              # [1,1,9,9] uint8

    # ---- expand to one-hot channels + pad to 30x30 ----
    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "oh")                        # [1,10,9,9] BOOL (810B)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 21, 21], np.int64), np.int64)
    n("Pad", ["oh", "pads"], "output", mode="constant")    # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task047", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 13)])
