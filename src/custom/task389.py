"""task389 (ARC-AGI f76d97a5) — recolour gray dots to the background colour on a
black canvas.

Rule (from the ARC-GEN generator, verified fresh):
  Input is a size x size grid (size 3..5) entirely filled with one non-gray colour
  `color`, with some cells overwritten by gray (=5).  Output is the SAME grid but
  black (=0) everywhere, with `color` placed at exactly the former gray positions.

  So per cell:
    input gray (5)      -> output color
    input color (bg)    -> output black (0)
  and off-grid cells stay all-zero in both.

Encoding (single 1x1 Conv, output is the FREE tensor):
  We want a [10,10,1,1] Conv weight W (W[out_ch,in_ch]) with exactly two ones:
    W[color, 5] = 1   (gray cell -> output channel `color`)
    W[0, color] = 1   (background cell -> output channel 0 = black)
  Then Conv(input, W) is the entire output, materialised straight into the FREE
  output tensor.

  `color` = the only non-gray, non-background channel present in the input.
  presence = ReduceMax(input) over the spatial axes, then mask out channels 0 and 5;
  ArgMax over channels gives the scalar colour index.  A runtime one-hot of `color`
  on the output axis, multiplied by a constant gray one-hot (in axis) gives term A;
  a constant black(0) one-hot on the output axis times a runtime colour one-hot on
  the in axis gives term B.  W = termA + termB, built directly in [10,10,1,1] shape
  (no 2D->4D reshape, no separate W_2d copy).

  Dominant intermediates: three [10,10,1,1] fp32 tensors (termA, termB, W) at 400B
  each = 1200B; the colour-index chain is a handful of <=40B tensors.  The output
  one-hot expansion costs nothing (it is the FREE output of the Conv).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []
    seen = set()

    def init(name, arr, dt):
        if name in seen:
            return name
        seen.add(name)
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16

    # ---- recover scalar colour index ---------------------------------------
    # presence[k] = does channel k occur anywhere?  (max over the spatial axes)
    n("ReduceMax", ["input"], "presence_raw", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    # zero out channel 0 (black/background-in-output) and channel 5 (gray)
    mask = np.ones((1, 10, 1, 1), np.float32)
    mask[0, 0] = 0.0
    mask[0, 5] = 0.0
    init("mask_nonfive", mask, np.float32)
    n("Mul", ["presence_raw", "mask_nonfive"], "presence")              # [1,10,1,1]
    n("ArgMax", ["presence"], "color_idx", axis=1, keepdims=1)          # [1,1,1,1] int64
    init("rs1", np.array([1], np.int64), np.int64)
    n("Reshape", ["color_idx", "rs1"], "color_v")                       # [1] int64

    # one-hot of `color` (fp16 values -> half-size working tensors)
    init("depth10", np.array(10, np.int64), np.int64)
    init("vals01", np.array([0.0, 1.0], np.float16), np.float16)
    n("OneHot", ["color_v", "depth10", "vals01"], "color_oh_2d")        # [1,10] fp16
    init("rs_out", np.array([10, 1, 1, 1], np.int64), np.int64)
    n("Reshape", ["color_oh_2d", "rs_out"], "color_oh_out")             # [10,1,1,1]
    init("rs_in", np.array([1, 10, 1, 1], np.int64), np.int64)
    n("Reshape", ["color_oh_2d", "rs_in"], "color_oh_in")              # [1,10,1,1]

    # constant one-hots (free initialisers), fp16
    gray_oh_in = np.zeros((1, 10, 1, 1), np.float16)
    gray_oh_in[0, 5] = 1.0
    init("gray_oh_in", gray_oh_in, np.float16)            # in-axis one-hot at 5
    black_oh_out = np.zeros((10, 1, 1, 1), np.float16)
    black_oh_out[0] = 1.0
    init("black_oh_out", black_oh_out, np.float16)        # out-axis one-hot at 0

    # term A: gray (in=5) -> output channel `color`  (fp16, 200B)
    n("Mul", ["color_oh_out", "gray_oh_in"], "termA")     # [10,10,1,1] fp16
    # term B: background (in=color) -> output channel 0 (black)
    n("Mul", ["black_oh_out", "color_oh_in"], "termB")    # [10,10,1,1] fp16
    n("Add", ["termA", "termB"], "W16")                   # [10,10,1,1] fp16
    n("Cast", ["W16"], "W", to=F)                         # [10,10,1,1] fp32 (Conv weight)

    n("Conv", ["input", "W"], "output")                   # [1,10,30,30] FREE

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task389", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
