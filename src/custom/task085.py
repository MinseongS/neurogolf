"""task085 (ARC 3bdb4ada): punchcard middle-row alternation.

Each colored 'punchcard' is a 3-row-tall horizontal bar (one bar per row band).
In the *middle* row of each bar, cells at an ODD offset from the bar's left edge
are erased to background. Runs have odd width and there is at most one run per
row, so the offset parity equals the parity of the running count of colored
cells along that row (horizontal prefix-count). The middle row of a bar sits at
a per-column vertical prefix-count == 2 (mod 3) (rows go Top,Middle,Bottom and
each bar contributes a length-3 multiple to any column it covers).

Pipeline (integer-valued, exact in float32):
  occ     = sum of color channels 1-9                              [1,1,30,30]
  hpre    = occ @ L      (L upper-tri ones)  horizontal prefix-count
  even    = (hpre mod 2 == 0)                  -> erase candidate (odd offset)
  vpre    = U @ occ      (U lower-tri ones)    vertical prefix-count
  mid     = (vpre mod 3 == 2)                  -> middle row
  occ_b   = occ > 0                            -> colored cell
  removed = occ_b & even & mid                                     (bool)
  output  = Where(removed, e0, input)          erased cells -> background
"""

import numpy as np
import onnx
from onnx import helper, TensorProto

from src.harness import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS


def build(task):
    nodes, inits = [], []

    def init(name, arr, dtype=TensorProto.FLOAT):
        arr = np.asarray(arr)
        inits.append(helper.make_tensor(name, dtype, arr.shape,
                                         arr.flatten().tolist()))

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))

    # occ = sum over color channels 1..9
    w_occ = np.zeros((1, 10, 1, 1), np.float32)
    w_occ[0, 1:, 0, 0] = 1.0
    init("w_occ", w_occ)
    n("Conv", ["input", "w_occ"], "occ", kernel_shape=[1, 1])
    n("Greater", ["occ", "half"], "occ_b")
    init("half", np.array([0.5], np.float32))

    # horizontal prefix-count -> even (odd offset from run start)
    L = np.triu(np.ones((30, 30), np.float32))  # L[i,j]=1 if i<=j
    init("Lmat", L)
    n("MatMul", ["occ", "Lmat"], "hpre")
    init("two_f", np.array([2.0], np.float32))
    n("Mod", ["hpre", "two_f"], "hmod", fmod=1)
    n("Less", ["hmod", "half"], "even_b")            # hmod==0 -> even

    # vertical prefix-count -> middle row (mod 3 == 2)
    U = np.tril(np.ones((30, 30), np.float32))  # U[r,i]=1 if i<=r
    init("Umat", U)
    n("MatMul", ["Umat", "occ"], "vpre")
    init("three_f", np.array([3.0], np.float32))
    n("Mod", ["vpre", "three_f"], "vmod", fmod=1)
    init("onehalf", np.array([1.5], np.float32))
    n("Greater", ["vmod", "onehalf"], "mid_b")       # vmod==2 -> middle

    # removed = occ_b & even_b & mid_b
    n("And", ["occ_b", "even_b"], "oe")
    n("And", ["oe", "mid_b"], "removed")

    # erased cells become one-hot background channel 0
    e0 = np.zeros((1, 10, 1, 1), np.float32)
    e0[0, 0, 0, 0] = 1.0
    init("e0", e0)
    n("Where", ["removed", "e0", "input"], "output")

    graph = helper.make_graph(nodes, "task085", [
        helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    ], [
        helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    ], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=OPSET_IMPORTS)
