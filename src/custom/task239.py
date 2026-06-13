"""Task 239 (ARC 9af7a82c): color histogram / sorted bar chart.

Each input color c appears count[c] times (counts are distinct among the
present colors, guaranteed by the generator). The output is a bar chart of
width = number of distinct colors, height = max count. Columns are sorted by
count descending; column at rank[c] = #{c': count[c'] > count[c]} is filled
with color c for rows 0..count[c]-1. Background (0) fills the rest of the
output rectangle (rows < oh, cols < ow).

Math: output[c,r,j] for c>=1 = [r<count[c]] * [j==rank[c]] (an outer product
of a row mask and a col mask). Channel 0 (background) = inside - cover where
inside[r,j] = [r<oh]*[j<ow] and cover = sum over c>=1 of the bars.

Build: a row matrix Vr [1,1,30,10] (cols 0..8 = [r<count[1..9]], col 9 =
[r<oh]) and a col matrix Vc [1,1,10,30] (rows 0..8 = [j==rank[1..9]], row 9 =
[j<ow]). A constant coefficient tensor My [1,10,10,10] selects per output
channel: ch c>=1 = Vr[:,c-1] (x) Vc[c-1]; ch 0 = Vr[:,9](x)Vc[9] minus the
sum of all 9 bars. tmp = MatMul(My, Vc) [1,10,10,30]; output = MatMul(Vr, tmp).
All values integer, partial sums tiny -> exact in float32.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # --- per-color counts -> [1,1,1,10] (colors on axis 3) ---------------
    n("ReduceSum", ["input"], "cnt_c", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    n("Transpose", ["cnt_c"], "cnt", perm=[0, 2, 3, 1])          # [1,1,1,10]

    # oh = max count (scalar broadcastable), ow = #present colors
    n("ReduceMax", ["cnt"], "oh", axes=[3], keepdims=1)          # [1,1,1,1]
    init("zero", np.array(0.0, np.float32))
    n("Greater", ["cnt", "zero"], "presb")                       # bool [1,1,1,10]
    n("Cast", ["presb"], "presf", to=int(onnx.TensorProto.FLOAT))
    n("ReduceSum", ["presf"], "ow", axes=[3], keepdims=1)        # [1,1,1,1]

    # rank[c] = #{c': count[c'] > count[c]}  -> [1,1,1,10]
    n("Transpose", ["cnt_c"], "cntA", perm=[0, 2, 1, 3])         # [1,1,10,1] (c')
    n("Greater", ["cntA", "cnt"], "gtb")                         # [1,1,10,10]
    n("Cast", ["gtb"], "gtf", to=int(onnx.TensorProto.FLOAT))
    n("ReduceSum", ["gtf"], "rank", axes=[2], keepdims=1)        # [1,1,1,10]

    # --- row matrix Vr [1,1,30,11] ---------------------------------------
    # cols 0..9 = [r < count[c]]; col 10 = [r < oh]
    n("Concat", ["cnt", "oh"], "T", axis=3)                      # [1,1,1,11]
    init("rowidx", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1))
    n("Greater", ["T", "rowidx"], "Vrb")        # T[k] > r == r < T[k]  [1,1,30,11]
    n("Cast", ["Vrb"], "Vr", to=int(onnx.TensorProto.FLOAT))

    # --- col matrix Vc [1,1,11,30] ---------------------------------------
    # rows 0..9 = [j == rank[c]]; row 10 = [j < ow]
    n("Transpose", ["rank"], "rankc", perm=[0, 1, 3, 2])         # [1,1,10,1]
    init("colidx", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30))
    # [j == rank] via 1 - clip(|j - rank|, 0, 1)   (integers)
    n("Sub", ["colidx", "rankc"], "dcol")                        # [1,1,10,30]
    n("Abs", ["dcol"], "adcol")
    n("Clip", ["adcol"], "cl1", min=0.0, max=1.0)
    init("one", np.array(1.0, np.float32))
    n("Sub", ["one", "cl1"], "Vc10")                             # [1,1,10,30]
    n("Less", ["colidx", "ow"], "owrowb")                        # [1,1,1,30]
    n("Cast", ["owrowb"], "owrow", to=int(onnx.TensorProto.FLOAT))
    n("Concat", ["Vc10", "owrow"], "Vc", axis=2)                 # [1,1,11,30]

    # --- coefficient tensor My [1,10,11,11] ------------------------------
    My = np.zeros((1, 10, 11, 11), np.float32)
    for c in range(1, 10):
        My[0, c, c, c] = 1.0                   # bar of color c
    My[0, 0, 10, 10] = 1.0                      # inside
    for i in range(1, 10):
        My[0, 0, i, i] = -1.0                   # minus each bar = inside - cover
    init("My", My)

    n("MatMul", ["My", "Vc"], "tmp")           # [1,10,11,30]
    n("MatMul", ["Vr", "tmp"], "output")       # [1,10,30,30] free

    return _model(nodes, inits)
