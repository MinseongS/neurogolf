"""Task 139 (ARC-GEN 60b61512): fill each sprite's 3x3 window background -> orange.

Rule (from ARC-GEN generator, verified fresh):
  A 9x9 grid carries two 3x3 sprites of yellow(4) pixels.  Each sprite is a
  Conway blob that ALWAYS spans its full 3x3 bounding box.  The two sprites sit
  at fixed grid-aligned 3x3 windows; an `xpose` flag (inverted transpose) picks
  one of exactly two layouts:
      xpose=0: window A = rows1-3 cols0-2,  window B = rows4-6 cols5-7
      xpose=1: window A = rows1-3 cols2-4,  window B = rows6-8 cols5-7
  Output: inside each window every background(0) cell becomes orange(7); yellow
  cells stay yellow(4).  Everything outside both windows stays background.

  So per cell the output label is:
      4  if input==yellow
      7  if (input==0) and cell in a window
      0  otherwise
  (the two cases never collide because a window cell is either yellow or 0).

Encoding (label-map + final Equal, Tier B, all-tiny intermediates):
  The benchmark feeds one-hot [1,10,30,30]; the 9x9 grid sits top-left.
  - Yellow presence Y = input[:,4:5,0:9,0:9]  (a Slice of the FREE input, f32).
  - xpose detection is a SCALAR: yellow can only appear in rows 7-8 when
    xpose=1 (window B drops to rows6-8), so xp = ReduceMax(Y over rows7-8) > 0.
  - Two constant 9x9 window masks M0, M1; select W = Where(xp, M1, M0).
  - Label L = Where(Y>0, c4, Where(W>0, c7, c0))  on [1,1,9,9].
  - Pad L to 30x30 with sentinel 10 (matches no colour channel -> off-grid
    cells are all-zero, matching the benchmark off-grid encoding).
  - Final op Equal(Lp, arange[0..9]) writes straight into the FREE BOOL output,
    so the 10-channel expansion costs no memory.

  Dominant intermediate: the padded label map Lp (uint8 [1,1,30,30] = 900 B),
  irreducible because it must be 30x30 to broadcast against the 10 colour
  channels in the final Equal.  Everything else is the 9x9 working tensors
  (<=324 B each) and tiny scalars.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

G = 9  # active grid size


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- yellow presence plane [1,1,9,9] (Slice of FREE input, ch 4) ----
    init("y_s", np.array([0, 4, 0, 0], np.int64), np.int64)
    init("y_e", np.array([1, 5, G, G], np.int64), np.int64)
    init("ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "y_s", "y_e", "ax"], "Y")  # [1,1,9,9] f32

    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["Y", "half"], "Yp")  # bool [1,1,9,9]

    # ---- xpose scalar: any yellow in rows 7-8 ? ----
    init("r78_s", np.array([0, 0, 7, 0], np.int64), np.int64)
    init("r78_e", np.array([1, 1, 9, G], np.int64), np.int64)
    init("ax2", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["Y", "r78_s", "r78_e", "ax2"], "Y78")  # [1,1,2,9]
    n("ReduceMax", ["Y78"], "Y78m", axes=[0, 1, 2, 3], keepdims=1)  # [1,1,1,1]
    n("Greater", ["Y78m", "half"], "xp")  # bool scalar [1,1,1,1]

    # ---- two fixed window LABEL maps [1,1,9,9] uint8 (7 inside window else 0) ----
    m0 = np.zeros((1, 1, G, G), np.uint8)
    m0[0, 0, 1:4, 0:3] = 7   # window A
    m0[0, 0, 4:7, 5:8] = 7   # window B
    m1 = np.zeros((1, 1, G, G), np.uint8)
    m1[0, 0, 1:4, 2:5] = 7   # window A
    m1[0, 0, 6:9, 5:8] = 7   # window B
    init("M0", m0, np.uint8)
    init("M1", m1, np.uint8)
    n("Where", ["xp", "M1", "M0"], "Lw")  # [1,1,9,9] uint8: 7 in windows else 0

    # ---- overlay yellow=4 on top ----
    init("c4", np.array(4, np.uint8), np.uint8)  # scalar broadcast
    n("Where", ["Yp", "c4", "Lw"], "L")  # 4 where yellow else Lw -> [1,1,9,9]

    # ---- pad to 30x30 with sentinel 10, final Equal into FREE BOOL output ----
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - G, 30 - G], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L", "pads", "padval"], "Lp", mode="constant")  # [1,1,30,30] uint8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["Lp", "chan"], "output")  # FREE BOOL [1,10,30,30]

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
