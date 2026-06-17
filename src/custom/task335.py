"""Task 335 (ARC d4a91cb9): common.hpwl L-route.

Input: black canvas with a red(2) dot at (r0,c0) [start] and a cyan(8) dot at
(r1,c1) [end].  Output: the two dots unchanged PLUS a yellow(4) L-path —
  * horizontal seg: row r0, columns STRICTLY between c0 and c1, and
  * vertical seg:  col c1, rows from r0 (inclusive) to r1 (exclusive).

Construction (Where-route, beats the shared _hpwl double-MatMul builder):
  The path cells are all background in the input, so the entire output is just
      output = Where(on_path, yellow_onehot, input)
  which PRESERVES both endpoints and the black background (ch0) for free — so,
  unlike the per-channel double-MatMul builder, we never reconstruct the grid
  extent (no rowin/colin basis vectors).

  on_path is a rank-2 outer product (the L = two disjoint separable segments):
      on_path = rowred ⊗ cH  +  rV ⊗ colcyan
  built as A[1,1,30,2] @ Bm[1,1,2,30] in fp16 (the single 30x30 plane), then
  Greater→bool feeds the final Where.

  ONE value-conv per axis (col0->1, col1->2) yields BOTH dot rows/cols from a
  single collapse-conv (red = clip(v) - (v - clip(v)); cyan = v - clip(v)),
  halving the conv params vs separate red/cyan convs.  All 1-D work is fp16.

  mem 4080, params 373 (mem+params 4453) -> 16.599 pts vs prior 16.294 (+0.305);
  FRESH 200/200.  The 30x30 fp16 product (1800B) + its bool (900B) dominate.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F = TensorProto.FLOAT
H16 = TensorProto.FLOAT16


def build(task, col0=2, col1=8, colp=4):
    inits, nodes = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    W = 18  # active grid is <=20 wide/tall, dots at index 1..17 -> window 18 covers them
    init("cmin", np.array(0.0, np.float16), dtype=np.float16)
    init("cmax", np.array(1.0, np.float16), dtype=np.float16)

    # --- row stream: collapse cols -> [1,1,30,1]; value 1 at red-row, 2 at cyan-row
    Wr = np.zeros((1, 10, 1, W), np.float32)
    Wr[0, col0, 0, :] = 1.0
    Wr[0, col1, 0, :] = 2.0
    init("Wr", Wr)
    n("Conv", ["input", "Wr"], "vr_f", strides=[1, W])
    n("Cast", ["vr_f"], "vr", to=H16)
    n("Clip", ["vr", "cmin", "cmax"], "wr")        # occupancy of either dot-row
    n("Sub", ["vr", "wr"], "gr")                   # cyan-row indicator
    n("Sub", ["wr", "gr"], "rowred")               # red-row indicator
    n("MaxPool", ["wr"], "pr", kernel_shape=[30, 1], pads=[29, 0, 0, 0])
    n("MaxPool", ["wr"], "qr", kernel_shape=[30, 1], pads=[0, 0, 29, 0])
    n("Mul", ["pr", "qr"], "rclosed")              # closed row interval [min,max]
    n("Sub", ["rclosed", "gr"], "rV")              # rows [r0..r1): keep red-row, drop cyan-row

    # --- col stream: collapse rows -> [1,1,1,30]; value 1 at red-col, 2 at cyan-col
    Wc = np.zeros((1, 10, W, 1), np.float32)
    Wc[0, col0, :, 0] = 1.0
    Wc[0, col1, :, 0] = 2.0
    init("Wc", Wc)
    n("Conv", ["input", "Wc"], "vc_f", strides=[W, 1])
    n("Cast", ["vc_f"], "vc", to=H16)
    n("Clip", ["vc", "cmin", "cmax"], "wc")        # occupancy of either dot-col
    n("Sub", ["vc", "wc"], "colcyan")              # cyan-col indicator
    n("MaxPool", ["wc"], "pc", kernel_shape=[1, 30], pads=[0, 29, 0, 0])
    n("MaxPool", ["wc"], "qc", kernel_shape=[1, 30], pads=[0, 0, 0, 29])
    n("Mul", ["pc", "qc"], "cclosed")              # closed col interval [min,max]
    n("Sub", ["cclosed", "wc"], "cH")              # cols strictly between (drop both endpoints)

    # --- rank-2 outer product: on_path = rowred*cH + rV*colcyan ---
    n("Concat", ["rowred", "rV"], "A", axis=3)     # [1,1,30,2] fp16
    n("Concat", ["cH", "colcyan"], "Bm", axis=2)   # [1,1,2,30] fp16
    n("MatMul", ["A", "Bm"], "onpath_f")           # [1,1,30,30] fp16, exact {0,1}
    init("half", np.array([0.5], np.float16), dtype=np.float16)
    n("Greater", ["onpath_f", "half"], "onpath")   # bool

    yoh = np.zeros((1, 10, 1, 1), np.float32)
    yoh[0, colp, 0, 0] = 1.0
    init("yoh", yoh)
    n("Where", ["onpath", "yoh", "input"], "output")

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
