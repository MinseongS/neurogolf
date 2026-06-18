"""task299 (ARC-AGI bdad9b1f) — complete the cyan column + red row, mark intersection yellow.

Rule (from generator, size=6):
  A vertical "street" cyan(8) sits at column `col`; a horizontal "street" red(2) at row `row`.
  In the INPUT only the first 2 cells of each street are drawn (cyan in rows {0,1} of col;
  red in cols {0,1} of row). In the OUTPUT both streets are drawn FULL length and their
  intersection (row,col) is yellow(4). An optional horizontal flip is applied to BOTH input
  and output identically -> the rule is orientation-agnostic.

  Identification (flip-invariant; exactly one row has red, exactly one col has cyan):
    a[r] = (row r contains red)  = ReduceMax over cols of channel-2 slice  -> [1,1,6,1]
    b[c] = (col c contains cyan) = ReduceMax over rows of channel-8 slice  -> [1,1,1,6]

  Output one-hot per cell (priority yellow > red/cyan > bg) is a SEPARABLE outer product of
  the tiny indicator vectors a,b and their complements na=1-a, nb=1-b:
    ch4 (yellow, a&b ) = a  * b      ch2 (red,  a&~b) = a  * nb
    ch8 (cyan, ~a&b )  = na * b      ch0 (bg,  ~a&~b) = na * nb
  Pack the row factors into rowsel[1,10,6,1] and the col factors into colsel[1,10,1,6]
  (per output channel: 0->na/nb, 2->a/nb, 4->a/b, 8->na/b, all others 0) so that ONE Mul
  produces the entire [1,10,6,6] one-hot, then Pad it (off-grid -> all-zero, correct) into
  the FREE [1,10,30,30] output. No [1,10,30,30] or [1,1,30,30] intermediate is materialized;
  the dominant intermediate is the [1,10,6,6] fp16 one-hot (720B), which is the irreducible
  10-channel-expansion floor for this task.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I64 = TensorProto.INT64

W = 6  # active grid (size=6)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("ax", np.array([1, 2, 3], np.int64), np.int64)

    # a[r] = row r contains red (channel 2)
    init("r_s", np.array([2, 0, 0], np.int64), np.int64)
    init("r_e", np.array([3, W, W], np.int64), np.int64)
    n("Slice", ["input", "r_s", "r_e", "ax"], "red")               # [1,1,6,6] f32
    n("ReduceMax", ["red"], "af", axes=[3], keepdims=1)            # [1,1,6,1] f32 {0,1}
    n("Cast", ["af"], "a", to=F16)                                 # [1,1,6,1] f16

    # b[c] = col c contains cyan (channel 8)
    init("c_s", np.array([8, 0, 0], np.int64), np.int64)
    init("c_e", np.array([9, W, W], np.int64), np.int64)
    n("Slice", ["input", "c_s", "c_e", "ax"], "cyan")              # [1,1,6,6] f32
    n("ReduceMax", ["cyan"], "bf", axes=[2], keepdims=1)           # [1,1,1,6] f32 {0,1}
    n("Cast", ["bf"], "b", to=F16)                                 # [1,1,1,6] f16

    # complements
    init("one", np.array(1.0, np.float16), np.float16)
    n("Sub", ["one", "a"], "na")                                   # [1,1,6,1] ~a
    n("Sub", ["one", "b"], "nb")                                   # [1,1,1,6] ~b

    # row/col selector stacks (channel order: 0,2,4,8 live; rest zero)
    init("zc", np.zeros((1, 1, W, 1), np.float16), np.float16)
    init("zr", np.zeros((1, 1, 1, W), np.float16), np.float16)
    n("Concat", ["na", "zc", "a", "zc", "a", "zc", "zc", "zc", "na", "zc"],
      "rowsel", axis=1)                                            # [1,10,6,1]
    n("Concat", ["nb", "zr", "nb", "zr", "b", "zr", "zr", "zr", "b", "zr"],
      "colsel", axis=1)                                            # [1,10,1,6]
    n("Mul", ["rowsel", "colsel"], "oh")                           # [1,10,6,6] one-hot

    # Pad to [1,10,30,30] = FREE output; off-grid cells -> all channels 0 (correct)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("zh", np.array(0.0, np.float16), np.float16)
    n("Pad", ["oh", "pads", "zh"], "output", mode="constant")      # [1,10,30,30]

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F16, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task299", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
