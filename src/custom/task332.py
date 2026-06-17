"""task332 (ARC-AGI d406998b) — "recolour alternate-parity gray pixels to green".

Rule (from the generator):
  Grid is height=3 rows by width=W columns (W = 10..20), embedded top-left in the
  30x30 canvas, background = 0.  Each column c in 0..W-1 holds exactly ONE gray(5)
  pixel at row rows[c].  Output keeps every pixel in place but recolours it:
      output[r][c] = green(3)  if  c % 2 != W % 2
                     gray(5)   otherwise
  i.e. the pixel becomes green iff (c + W) is ODD (equivalently the column has the
  opposite parity to the width); otherwise it stays gray.  All off-grid / background
  cells stay background.

Encoding (closed-form, separable; route the 10-ch expansion into the FREE output):
  The ONLY change vs input is: gray pixels in green-parity columns flip 5 -> 3.
    output = Where(cond, green_onehot, input)
  where
    cond[R,C] = gray_present(R,C)  AND  greencol(C)
    gray_present = input gray channel (ch5) sliced to the active 3 rows.
    greencol(C)  = ((C + W) mod 2 == 1)   over a [1,1,1,30] column ramp,
                   W = ReduceSum of the per-column presence vector (one gray per
                   in-grid column, so the count of occupied columns IS W).
  cond is computed on a tiny [1,1,3,30] plane, padded to [1,1,30,30] uint8 (900B),
  cast to bool; green_onehot is a constant [1,10,1,1].  Where broadcasts the cond
  and the [1,10,1,1] value against the FREE [1,10,30,30] input, so the only
  10-channel tensor is the free output.  Dominant intermediate = the 30x30 uint8
  cond plane (900B); fp16 Mod is integer-exact for these small magnitudes.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

H = 3  # grid height is always 3 for this task


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- gray-presence on the active 3 rows: input gray channel (ch5) -------
    init("gm_s", np.array([5, 0, 0], np.int64), np.int64)
    init("gm_e", np.array([6, H, 30], np.int64), np.int64)
    init("gm_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "gm_s", "gm_e", "gm_ax"], "gm_f32")   # [1,1,3,30] f32 {0,5}
    # normalise to a {0,1} presence mask (fp16): gray==5 -> any nonzero -> >0
    n("Cast", ["gm_f32"], "gm16", to=F16)                       # [1,1,3,30] fp16
    init("ZERO16", np.array(0.0, np.float16), np.float16)
    n("Greater", ["gm16", "ZERO16"], "gray_b")                  # [1,1,3,30] bool

    # ---- W = number of occupied columns ------------------------------------
    # per-column presence = max over the 3 rows  ->  [1,1,1,30]
    n("ReduceMax", ["gm16"], "colpres", axes=[2], keepdims=1)   # [1,1,1,30] fp16 {0,5}
    n("Greater", ["colpres", "ZERO16"], "colpres_b")            # bool
    n("Cast", ["colpres_b"], "colpres1", to=F16)                # [1,1,1,30] fp16 {0,1}
    n("ReduceSum", ["colpres1"], "W", axes=[3], keepdims=1)     # [1,1,1,1] fp16 = width

    # ---- greencol(C) = ((C + W) mod 2 == 1) over a column ramp -------------
    colramp = np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30)
    init("colramp", colramp, np.float16)                        # [1,1,1,30]
    n("Add", ["colramp", "W"], "cW")                            # [1,1,1,30] fp16 (C+W)
    init("TWO16", np.array(2.0, np.float16), np.float16)
    n("Mod", ["cW", "TWO16"], "par", fmod=1)                    # [1,1,1,30] fp16 {0,1}
    init("ONE16", np.array(1.0, np.float16), np.float16)
    n("Equal", ["par", "ONE16"], "greencol")                    # [1,1,1,30] bool

    # ---- cond = gray_present AND greencol  on the 3-row plane --------------
    n("And", ["gray_b", "greencol"], "cond3")                   # [1,1,3,30] bool

    # ---- pad to 30x30 (uint8) then -> bool --------------------------------
    n("Cast", ["cond3"], "cond3_u8", to=U8)                     # [1,1,3,30] uint8
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - H, 0], np.int64), np.int64)
    init("ZEROU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["cond3_u8", "pads", "ZEROU8"], "cond30", mode="constant")  # [1,1,30,30] u8
    n("Cast", ["cond30"], "cond", to=BOOL)                      # [1,1,30,30] bool

    # ---- green one-hot constant [1,10,1,1] (green = colour 3) --------------
    green = np.zeros((1, 10, 1, 1), dtype=np.float32)
    green[0, 3, 0, 0] = 1.0
    init("green_oh", green, np.float32)                         # [1,10,1,1] f32

    # ---- output = Where(cond, green_onehot, input) : FREE [1,10,30,30] ------
    n("Where", ["cond", "green_oh", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task332", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
