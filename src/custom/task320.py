"""Task 320 (ARC ce9e57f2): four bottom-anchored vertical red(2) bars in odd
columns 1,3,5,7.  For each bar of length L, recolour its bottom floor(L/2) cells
to cyan(8); the remaining top cells stay red(2).  Background 0 stays 0.  Output
is the SAME-size grid.

Closed-form, fully separable per column -> tier-A floor-break (no flood-fill):
  * A red cell becomes cyan iff (#red strictly below it in its column) < L/2,
    where L = #red in that column.  `below` is a strictly-upper-triangular
    row-MatMul; L is a full column-sum.  half = floor(L/2) via fp16 Mod.
  * WORKING CANVAS = 11x9.  width is always 2*4+1=9; height=max(len)+1<=11 since
    lengths in [2,10].  Slice ONLY channel 2 (red) to [1,1,11,9] (small fp32).
  * 10-channel expansion lives ONLY in the FREE bool output: a uint8 label map
    L (0/2/8) is Pad-ed back to 30x30 (off-grid -> 0 -> channel 0 background)
    and emitted as Equal(L, arange) -> BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
HC, WC = 11, 9  # working canvas: height<=11, width==9


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # ---- red plane: slice channel 2, cropped to HCxWC -> small fp32 ----------
    init("rst", np.array([2, 0, 0], np.int64), np.int64)
    init("ren", np.array([3, HC, WC], np.int64), np.int64)
    init("rax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "rst", "ren", "rax"], "red_f")        # [1,1,HC,WC] fp32
    n("Cast", ["red_f"], "R", to=F16)                          # 0/1 fp16

    # ---- A red cell is cyan iff #red below it < floor(L/2), L=#red in column.
    # Integer identity: below < floor(L/2)  <=>  2*below + 1 < L.  Fold the
    # 2*below and the +1 into ONE matmul B (2 strictly-below, 1 on diagonal): at
    # a red cell B@R = 2*below + R[r,c]*1 = 2*below + 1.  L[c] = ones-row @ R. --
    B = 2.0 * np.triu(np.ones((HC, HC), np.float16), 1) + np.eye(HC, dtype=np.float16)
    init("B", B, np.float16)
    n("MatMul", ["B", "R"], "t")                               # 2*below+1 (red)

    # L[c] = total red in column = column-sum, broadcasts over rows (no matrix)
    n("ReduceSum", ["R"], "Lcol", axes=[2], keepdims=1)        # [1,1,1,WC] fp16

    n("Less", ["t", "Lcol"], "lt")                             # 2*below+1 < L
    init("Q", np.array(0.5, np.float16), np.float16)
    n("Greater", ["R", "Q"], "isred")                          # bool red mask
    n("And", ["lt", "isred"], "iscyan")                        # bool cyan mask

    # ---- in-grid row mask: height = max(len)+1 is data-dependent.  The longest
    # bar spans rows [1, height-1], so in-grid rows = {row 0} U {rows with red}.
    # Recover from R alone (no 10-channel slice): rowhasred OR row==0. --------
    n("ReduceMax", ["R", ], "rowred", axes=[3], keepdims=1)    # [1,1,HC,1] fp16
    n("Greater", ["rowred", "Q"], "hasred")                    # bool
    row0 = np.zeros((1, 1, HC, 1), bool); row0[0, 0, 0, 0] = True
    init("row0", row0, np.bool_)
    n("Or", ["hasred", "row0"], "ingrid")                      # bool in-grid row

    # ---- uint8 label map: in-grid bg 0, red 2, cyan 8, off-grid 10 (no match)
    init("v2u", np.array(2, np.uint8), np.uint8)
    init("v8u", np.array(8, np.uint8), np.uint8)
    init("v0u", np.array(0, np.uint8), np.uint8)
    init("v10u", np.array(10, np.uint8), np.uint8)             # off-grid sentinel
    n("Where", ["ingrid", "v0u", "v10u"], "Lg")               # 0 in-grid else 10
    n("Where", ["isred", "v2u", "Lg"], "Lred")                # red where red
    n("Where", ["iscyan", "v8u", "Lred"], "Lc")               # cyan overrides

    # pad label back to 30x30 with sentinel 10 (off-canvas -> no channel match)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - HC, 30 - WC], np.int64),
         np.int64)
    init("pv", np.array(10, np.uint8), np.uint8)
    n("Pad", ["Lc", "pads", "pv"], "L", mode="constant")

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                        # -> BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
