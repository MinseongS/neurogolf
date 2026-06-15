"""task085 (ARC 3bdb4ada): punchcard middle-row alternation.

Each colored 'punchcard' is a 3-row-tall horizontal bar (one bar per row band).
In the *middle* row of each bar, cells at an ODD offset from the bar's left edge
are erased to background. Runs have odd width and there is at most one run per
row, so the offset parity equals the parity of the running count of colored
cells along that row (horizontal prefix-count). The middle row of a bar sits at
a per-column vertical prefix-count == 2 (mod 3) (rows go Top,Middle,Bottom and
each bar contributes a length-3 multiple to any column it covers).

Pipeline (integer-valued, exact in float32/float16):
  occ32   = sum of color channels 1-9 (Conv reads the free `input`)  [1,1,30,30]
  occ16f  = Cast(occ32, fp16)
  occ     = Slice(occ16f, rows 0..15)   -> active 16x30 canvas
  occ_b   = occ > 0                            -> colored cell        (bool)
  hpre    = occ @ L      (L upper-tri ones)  horizontal prefix-count
  even_b  = (hpre mod 2 == 0)                  -> erase candidate (odd offset)
  vpre    = U @ occ      (U lower-tri ones)    vertical prefix-count
  mid_b   = (vpre mod 3 == 2)                  -> middle row
  rem16   = occ_b & even_b & mid_b                            (bool [1,1,16,30])
  removed = Cast(Pad(rem16) back to 30 rows, bool)
  output  = Where(removed, e0, input)          erased cells -> background (free)

Memory floor-break: the generator caps grid height at 16 rows, so after the one
unavoidable fp32 channel-reduction every per-cell prefix-sum / mod plane runs in
fp16 on a 16x30 canvas (480 cells, ~half of 30x30).  `removed` is padded back to
30 rows (0 fill) for the free `Where` into the fp32 `input`.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F = TensorProto.FLOAT
H16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
HROWS = 16  # active-row bound (grid height <= 16)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # occ32 = sum over color channels 1..9 (Conv reads free `input`)
    w_occ = np.zeros((1, 10, 1, 1), np.float32)
    w_occ[0, 1:, 0, 0] = 1.0
    init("w_occ", w_occ, np.float32)
    n("Conv", ["input", "w_occ"], "occ32", kernel_shape=[1, 1])   # [1,1,30,30] f32
    n("Cast", ["occ32"], "occ16f", to=H16)                        # [1,1,30,30] f16

    # slice to the active 16 rows
    init("sl_st", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("sl_en", np.array([1, 1, HROWS, 30], np.int64), np.int64)
    init("sl_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["occ16f", "sl_st", "sl_en", "sl_ax"], "occ")      # [1,1,16,30] f16
    init("half16", np.array(0.5, np.float16), np.float16)
    n("Greater", ["occ", "half16"], "occ_b")            # colored cell (bool)

    # horizontal prefix-count -> even (odd offset from run start)  [30x30 L]
    L = np.triu(np.ones((30, 30), np.float16))          # L[i,j]=1 if i<=j
    init("Lmat", L, np.float16)
    n("MatMul", ["occ", "Lmat"], "hpre")                # [1,1,16,30] fp16
    init("two16", np.array(2.0, np.float16), np.float16)
    n("Mod", ["hpre", "two16"], "hmod", fmod=1)
    n("Less", ["hmod", "half16"], "even_b")             # hmod==0 -> even

    # vertical prefix-count -> middle row (mod 3 == 2)  [16x16 U]
    U = np.tril(np.ones((HROWS, HROWS), np.float16))    # U[r,i]=1 if i<=r
    init("Umat", U, np.float16)
    n("MatMul", ["Umat", "occ"], "vpre")                # [1,1,16,30] fp16
    init("three16", np.array(3.0, np.float16), np.float16)
    n("Mod", ["vpre", "three16"], "vmod", fmod=1)
    init("onehalf16", np.array(1.5, np.float16), np.float16)
    n("Greater", ["vmod", "onehalf16"], "mid_b")        # vmod==2 -> middle

    # removed = occ_b & even_b & mid_b  (on the 16-row canvas)
    n("And", ["occ_b", "even_b"], "oe")
    n("And", ["oe", "mid_b"], "rem16b")
    n("Cast", ["rem16b"], "rem16", to=U8)               # Pad needs non-bool

    # pad removed back to 30 rows (0 fill) for the free Where into input
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - HROWS, 0], np.int64), np.int64)
    init("padval", np.array(0, np.uint8), np.uint8)
    n("Pad", ["rem16", "padpads", "padval"], "remU", mode="constant")
    n("Cast", ["remU"], "removed", to=BOOL)             # [1,1,30,30] bool

    # erased cells become one-hot background channel 0 (free Where into input)
    e0 = np.zeros((1, 10, 1, 1), np.float32)
    e0[0, 0, 0, 0] = 1.0
    init("e0", e0, np.float32)
    n("Where", ["removed", "e0", "input"], "output")

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task085", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
