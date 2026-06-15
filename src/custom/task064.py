"""Task 064 (ARC 2c608aff): dots aligned with a box shoot rays toward it.

Rule (verified on all 267 examples):
- one solid boxcolor rectangle, sparse dotcolor pixels on bgcolor background
- a dot whose row lies in the box's row span fills horizontally from itself
  up to the box edge with dotcolor; same vertically for column-span dots;
  diagonal (corner-region) dots stay single pixels

Implementation (colors vary per example, so channel roles are computed
in-graph from tiny aggregates):
- bg = max-count channel; box = non-bg channel with count == nrows*ncols
  and count >= 9 (solid rectangle signature); dot = remaining present channel
- dynamic-weight 1x1 Conv builds E0 = +1 at dots, -30 at box cells
- prefix/suffix sums of E0 along rows/cols via 1x30 / 30x1 all-ones Convs
  with asymmetric pads: cum > 0 iff a dot precedes with no box in between
  (-30 sentinel kills any prefix containing box cells)
- row/col span masks from ReduceMin(E0) < -1, folded into the threshold of
  the Greater via Where(span, 0.5, 1e9)
- final Where(fillmask, dot_onehot, input) writes `output` directly
"""

import numpy as np
import onnx
from onnx import TensorProto

from ..builders import _model


def build(task):
    nodes, inits = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(onnx.numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(onnx.helper.make_node(op, inputs, [out], **attrs))
        return out

    F16 = TensorProto.FLOAT16
    init("kH", np.ones((1, 1, 1, 30), np.float16), np.float16)
    init("kV", np.ones((1, 1, 30, 1), np.float16), np.float16)
    init("c30", np.array(30.0, np.float32))
    init("neg1", np.array(-1.0, np.float16), np.float16)
    init("half", np.array(0.5, np.float16), np.float16)
    init("big", np.array(1e4, np.float16), np.float16)
    init("zero_i", np.array(0, np.int32), np.int32)
    init("eight_i", np.array(8, np.int32), np.int32)

    # --- channel roles from tiny aggregates ---
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    n("ReduceMax", ["input"], "R", axes=[3], keepdims=1)       # [1,10,30,1]
    n("ReduceMax", ["input"], "C", axes=[2], keepdims=1)       # [1,10,1,30]
    n("ReduceSum", ["R"], "nr", axes=[2], keepdims=1)          # [1,10,1,1]
    n("ReduceSum", ["C"], "nc", axes=[3], keepdims=1)
    n("Mul", ["nr", "nc"], "prod")
    n("Cast", ["cnt"], "cnti", to=TensorProto.INT32)
    n("Cast", ["prod"], "prodi", to=TensorProto.INT32)
    n("ReduceMax", ["cnti"], "mx", keepdims=1)                 # [1,1,1,1] i
    n("Equal", ["cnti", "mx"], "bg_b")                         # [1,10,1,1] b
    n("Not", ["bg_b"], "nbg_b")
    n("Equal", ["cnti", "prodi"], "rect_b")
    n("Greater", ["cnti", "eight_i"], "ge9_b")
    n("And", ["rect_b", "ge9_b"], "a1")
    n("And", ["a1", "nbg_b"], "box_b")
    n("Greater", ["cnti", "zero_i"], "pres_b")
    n("Not", ["box_b"], "nbox_b")
    n("And", ["pres_b", "nbg_b"], "a2")
    n("And", ["a2", "nbox_b"], "dot_b")
    n("Cast", ["dot_b"], "dot_f", to=TensorProto.FLOAT)        # [1,10,1,1] f
    n("Cast", ["box_b"], "box_f", to=TensorProto.FLOAT)
    n("Mul", ["box_f", "c30"], "b30")
    n("Sub", ["dot_f", "b30"], "Wrun")                         # [1,10,1,1] f

    # --- E0: +1 at dots, -30 at box cells (dynamic-weight 1x1 conv) ---
    n("Conv", ["input", "Wrun"], "E0f")                        # [1,1,30,30] f32
    n("Cast", ["E0f"], "E0", to=F16)                           # [1,1,30,30] f16

    # --- span masks -> per-row/col thresholds (fp16) ---
    n("ReduceMin", ["E0"], "rmin", axes=[3], keepdims=1)       # [1,1,30,1] f16
    n("ReduceMin", ["E0"], "cmin", axes=[2], keepdims=1)       # [1,1,1,30] f16
    n("Less", ["rmin", "neg1"], "rs_b")
    n("Less", ["cmin", "neg1"], "cs_b")
    n("Where", ["rs_b", "half", "big"], "thrH")                # [1,1,30,1] f16
    n("Where", ["cs_b", "half", "big"], "thrV")

    # --- directional cumulative sums (box-blocked by -30 sentinel), fp16 ---
    n("Conv", ["E0", "kH"], "cumL", pads=[0, 29, 0, 0])        # prefix in row
    n("Conv", ["E0", "kH"], "cumR", pads=[0, 0, 0, 29])        # suffix in row
    n("Conv", ["E0", "kV"], "cumT", pads=[29, 0, 0, 0])        # prefix in col
    n("Conv", ["E0", "kV"], "cumB", pads=[0, 0, 29, 0])        # suffix in col

    # --- fill mask ---
    n("Greater", ["cumL", "thrH"], "gL")                       # bool canvases
    n("Greater", ["cumR", "thrH"], "gR")
    n("Greater", ["cumT", "thrV"], "gT")
    n("Greater", ["cumB", "thrV"], "gB")
    n("Or", ["gL", "gR"], "oH")
    n("Or", ["gT", "gB"], "oV")
    n("Or", ["oH", "oV"], "F0")

    # --- output: fill cells become dotcolor one-hot, rest pass through ---
    n("Where", ["F0", "dot_f", "input"], "output")

    return _model(nodes, inits)
