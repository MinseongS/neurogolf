"""task356 (ARC-AGI ded97339) — connect collinear cyan pixels with cyan spans.

Rule (from generator):
  Fixed 10x10 grid (size=10), placed top-left of the 30x30 canvas (rest is
  background color 0).  Random cyan(8) pixels are scattered.  In the OUTPUT, for
  every pair of cyan pixels sharing a ROW the cells between them are filled cyan;
  likewise for every pair sharing a COLUMN.  Per row the closed span
  [min cyan col, max cyan col] becomes cyan; same per column.  Original cyan
  pixels stay cyan (and single-pixel rows/cols fill only that one cell).

  cell becomes cyan iff it lies in some row-span OR some col-span:
    h_span[r,c] = (cyan in row r at col<=c) AND (cyan in row r at col>=c)
    v_span[r,c] = (cyan in col c at row<=r) AND (cyan in col c at row>=r)
  Since both the endpoints AND the fill are cyan, the span mask IS the output
  cyan mask (no separate "not endpoint" gate needed).  Off-grid is background
  with no cyan, so spans never leak off the 10x10 grid.

Encoding (route the 10-ch expansion into the FREE Where output):
  Work on the active 10x10 canvas.  C = cyan plane (channel 8), cast f16.
  prefix/suffix-OR per row & per col INDEPENDENTLY via fp16 MaxPool with a
  full-length 1-D window + one-sided pad (task350 lever, ZERO params); the rule
  is NOT row(x)col separable so full 2-D planes are required:
    leftOR  = maxpool prefix along cols ; rightOR = maxpool suffix along cols
    upOR    = maxpool prefix along rows ; downOR  = maxpool suffix along rows
  hspan = (leftOR*rightOR)>0 ; vspan = (upOR*downOR)>0 ; span = hspan|vspan.
  output = Where(span_padded_to_30, cyan_onehot, input).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8

N = 30
W = 10  # fixed active grid (size=10)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- cyan plane (channel 8) on the WxW active canvas, cast to f16 -------
    init("cy_s", np.array([8, 0, 0], np.int64), np.int64)
    init("cy_e", np.array([9, W, W], np.int64), np.int64)
    init("cy_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "cy_s", "cy_e", "cy_ax"], "cyan_f32")  # [1,1,W,W] f32
    n("Cast", ["cyan_f32"], "C", to=F16)  # [1,1,W,W] f16

    # prefix/suffix-OR along each axis via fp16 MaxPool with asymmetric padding
    # (full-length 1-D window; one-sided pad = directional running max). No params.
    n("MaxPool", ["C"], "leftOR", kernel_shape=[1, W], pads=[0, W - 1, 0, 0])
    n("MaxPool", ["C"], "rightOR", kernel_shape=[1, W], pads=[0, 0, 0, W - 1])
    n("Mul", ["leftOR", "rightOR"], "hprod")  # >0 iff in h-span

    n("MaxPool", ["C"], "upOR", kernel_shape=[W, 1], pads=[W - 1, 0, 0, 0])
    n("MaxPool", ["C"], "downOR", kernel_shape=[W, 1], pads=[0, 0, W - 1, 0])
    n("Mul", ["upOR", "downOR"], "vprod")     # >0 iff in v-span

    # span = hprod>0 OR vprod>0
    init("ZH", np.array(0.0, np.float16), np.float16)
    n("Greater", ["hprod", "ZH"], "hb")       # [1,1,W,W] bool
    n("Greater", ["vprod", "ZH"], "vb")
    n("Or", ["hb", "vb"], "span")             # [1,1,W,W] bool

    # ---- pad back to 30x30 for the Where cond ------------------------------
    n("Cast", ["span"], "span_u8", to=U8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, N - W, N - W], np.int64), np.int64)
    init("ZU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["span_u8", "pads", "ZU8"], "span30", mode="constant")  # [1,1,30,30] u8
    n("Cast", ["span30"], "fill", to=BOOL)

    # ---- cyan one-hot (color 8) --------------------------------------------
    oh = np.zeros((1, 10, 1, 1), np.float32)
    oh[0, 8, 0, 0] = 1.0
    init("cyan_oh", oh, np.float32)

    n("Where", ["fill", "cyan_oh", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", F32, [1, 10, N, N])
    graph = helper.make_graph(nodes, "task356", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
