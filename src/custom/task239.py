"""Task 239 (ARC 9af7a82c): color histogram / sorted bar chart.

Each input color c appears count[c] times (counts are distinct among the
present colors, guaranteed by the generator). The output is a bar chart of
width = number of distinct colors (ow), height = max count (oh). Columns are
sorted by count descending; the column at rank[c] = #{c': count[c'] > count[c]}
is filled with color c for rows 0..count[c]-1. Background (0) fills the rest of
the output rectangle (rows < oh, cols < ow).

Memory floor-break (label map + final Equal on a small canvas):
  For every output column j there is at most one color whose rank == j, so the
  bar chart collapses to two 1-D (per-column) vectors:
    colAt[j]    = the color occupying column j      (0 where j has no bar)
    heightAt[j] = that color's count (bar height)
  built by scattering each present color c onto column rank[c] via a tiny
  [1,1,10,W] one-hot (rank[c]==j) and summing over the color axis.

  The label map L[r,j] is then a single uint8 plane:
    L = colAt[j]  if r < heightAt[j]                (a bar cell)
        0         elif r < oh and j < ow            (in-rectangle background)
        10        else                              (outside -> matches nothing)
  All work is on a WxW canvas (oh, ow <= 16 < W) then Pad to 30x30 (sentinel
  10) and the final op  output = Equal(L, arange[1,10,1,1])  writes straight
  into the free BOOL output (opset 11) -- no [1,10,30,30] intermediate ever
  materialises.  Every value is a small integer, exact in float32 / uint8.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WR = 16  # working canvas height (oh = max count <= 16)
WC = 9   # working canvas width (ow = #distinct colors <= 9)


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

    # --- per-color counts ------------------------------------------------
    n("ReduceSum", ["input"], "cnt_c", axes=[2, 3], keepdims=1)   # [1,10,1,1]
    n("Transpose", ["cnt_c"], "cnt", perm=[0, 2, 3, 1])           # [1,1,1,10]
    n("ReduceMax", ["cnt"], "oh", axes=[3], keepdims=1)           # [1,1,1,1]
    init("zero", np.array(0.0, np.float32), np.float32)
    n("Greater", ["cnt", "zero"], "presb")                        # [1,1,1,10]
    n("Cast", ["presb"], "presf", to=F)
    n("ReduceSum", ["presf"], "ow", axes=[3], keepdims=1)         # [1,1,1,1]

    # rank[c] = #{c': count[c'] > count[c]} ; absent colors -> large rank.
    # The whole [1,1,10,WC] scatter chain runs in fp16 (counts <= 16, colors
    # <= 9 are exact in fp16) so each scatter plane is 180 B not 360 B.
    H16 = TensorProto.FLOAT16
    n("Transpose", ["cnt_c"], "cntA", perm=[0, 2, 1, 3])          # [1,1,10,1]
    n("Greater", ["cntA", "cnt"], "gtb")                          # [1,1,10,10]
    n("Cast", ["gtb"], "gtf", to=H16)
    n("ReduceSum", ["gtf"], "rank", axes=[2], keepdims=1)         # [1,1,1,10] f16

    # one-hot scatter onehot[c,j] = present[c] & (rank[c] == j)  [1,1,10,WC]
    n("Transpose", ["rank"], "rankc", perm=[0, 1, 3, 2])          # [1,1,10,1] f16
    init("colidx16", np.arange(WC, dtype=np.float16).reshape(1, 1, 1, WC),
         np.float16)
    n("Sub", ["colidx16", "rankc"], "d")                          # [1,1,10,WC] f16
    n("Abs", ["d"], "ad")
    init("half16", np.array(0.5, np.float16), np.float16)
    n("Less", ["ad", "half16"], "ohb")                            # rank==j
    n("Cast", ["ohb"], "ohf", to=H16)
    n("Cast", ["presf"], "presf16", to=H16)
    n("Transpose", ["presf16"], "presc", perm=[0, 1, 3, 2])       # [1,1,10,1] f16
    n("Mul", ["ohf", "presc"], "oh2")                             # [1,1,10,WC] f16

    # colAt[j] = sum_c c * oh2 ; heightAt[j] = sum_c cnt[c] * oh2
    init("carr16", np.arange(10, dtype=np.float16).reshape(1, 1, 10, 1),
         np.float16)
    n("Mul", ["oh2", "carr16"], "colparts")
    n("ReduceSum", ["colparts"], "colAt", axes=[2], keepdims=1)   # [1,1,1,WC] f16
    n("Cast", ["cnt_c"], "cnt_c16", to=H16)
    n("Transpose", ["cnt_c16"], "cntcol", perm=[0, 2, 1, 3])      # [1,1,10,1] f16
    n("Mul", ["oh2", "cntcol"], "hparts")
    n("ReduceSum", ["hparts"], "heightAt", axes=[2], keepdims=1)  # [1,1,1,WC] f16
    n("Cast", ["colAt"], "colAt_u8", to=U8)                       # [1,1,1,WC]

    # --- label map L on the WR x WC canvas (row/col masks in fp16) --------
    init("rowidx", np.arange(WR, dtype=np.float16).reshape(1, 1, WR, 1),
         np.float16)
    n("Cast", ["oh"], "oh16", to=H16)
    n("Cast", ["ow"], "ow16", to=H16)
    n("Less", ["rowidx", "heightAt"], "barb")                     # r<height
    n("Less", ["rowidx", "oh16"], "rinb")                         # [1,1,WR,1]
    n("Less", ["colidx16", "ow16"], "cinb")                       # [1,1,1,WC]
    n("And", ["rinb", "cinb"], "insideb")                         # [1,1,WR,WC]
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v10", np.array(10, np.uint8), np.uint8)
    n("Where", ["insideb", "v0", "v10"], "Lin")                   # 0 inside else 10
    n("Where", ["barb", "colAt_u8", "Lin"], "Lwk")                # bar color overrides

    # pad to 30x30 (sentinel 10), then final Equal -> free BOOL output
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - WR, 30 - WC], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["Lwk", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30]
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task239", [x], [y], inits)
    return helper.make_model(
        g, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
