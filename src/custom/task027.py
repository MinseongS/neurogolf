"""task027 (ARC-AGI 1b60fb0c) — add the missing 4th rotational copy (in red) of a
3-fold-symmetric blue figure.

Rule (from the generator):
  A small "creature" shape S is placed FOUR times under the cyclic rotation group C4
  about a centre.  Three copies are drawn in blue(1) (the INPUT); the fourth (missing)
  copy is drawn in red(2) and ADDED to make the OUTPUT.  The whole figure (blue+red)
  has exact C4 rotational symmetry about a centre at index (9+off)/2 on each axis,
  where off in {0,1} is a per-instance random single-pixel offset.

  Because blue is exactly three of the four C4 copies, rotating blue by 90 degrees
  about the correct centre brings the missing copy into view:
      red = rot90_cen(blue)  AND NOT blue ,   cen = 9 + off  ("2*centre").
  rot90_cen(X)[r,c] = X[cen-c, r].  For cen=9:  rot90_9(X) = transpose(reverse_rows(X))
  (no matrix needed).  For cen=10:  rot90_10(X)[r,c] = X[10-c, r] = rot90_9(X)[r, c-1],
  i.e. rot90_9 shifted right one column.

  off is recovered offset-free: the CORRECT centre is the one for which red (the cells
  exposed by the rotation) is SMALLEST — exactly one clean copy; the wrong centre
  scatters blue into a larger set.  Pick the centre minimising
  |rot90_cen(blue) AND NOT blue|.  (Verified exact on 40000 fresh instances.)

  The grid is 10x10 at the top-left of the 30x30 canvas, so all work is on the 10x10
  region (off-grid cells are background and never become red).

Encoding (Tier A — rotation by reverse+transpose, no MatMul; route the 10-ch
expansion into the FREE Where output):
  blue = input[:, 1:2, 0:10, 0:10] (fp16, [1,1,10,10]).
  g9  = transpose(reverse_rows(blue))          (rot90 about cen=9).
  g10 = g9 shifted right one column            (rot90 about cen=10).
  cand = (g > 0) AND (blue == 0) ; sz = sum(cand) ; pick the smaller-sized cand.
  Pad cand to 30x30 -> cond ; output = Where(cond, red_onehot[ch2], input).
  Dominant intermediate is the [1,1,30,30] cond / its uint8 pad pre-image (~900B each).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

G = 10   # active grid size (always 10x10 for this task)
N = 30   # full canvas


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- blue slice (channel 1) on the 10x10 region ------------------------
    init("blue_s", np.array([1, 0, 0], np.int64), np.int64)
    init("blue_e", np.array([2, G, G], np.int64), np.int64)
    init("blue_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "blue_s", "blue_e", "blue_ax"], "blue_f32")  # [1,1,10,10] f32
    n("Cast", ["blue_f32"], "blue", to=F16)                            # [1,1,10,10] fp16 {0,1}

    init("ZEROH", np.array(0.0, np.float16), np.float16)

    # ---- rot90 about cen=9: g9 = transpose(reverse_rows(blue)) -------------
    init("rev_s", np.array([G - 1], np.int64), np.int64)
    init("rev_e", np.array([-(G + 1)], np.int64), np.int64)   # down to index 0 inclusive
    init("rev_ax", np.array([2], np.int64), np.int64)
    init("rev_st", np.array([-1], np.int64), np.int64)
    n("Slice", ["blue", "rev_s", "rev_e", "rev_ax", "rev_st"], "rev")  # reverse rows
    n("Transpose", ["rev"], "g9", perm=[0, 1, 3, 2])                   # [1,1,10,10] rot90_9

    # ---- rot90 about cen=10: g10 = g9 shifted right one column -------------
    # pad one zero column on the LEFT of axis 3, then drop the last column.
    init("padL", np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), np.int64)
    n("Pad", ["g9", "padL", "ZEROH"], "g9p", mode="constant")  # [1,1,10,11]
    init("g10_s", np.array([0], np.int64), np.int64)
    init("g10_e", np.array([G], np.int64), np.int64)
    init("g10_ax", np.array([3], np.int64), np.int64)
    n("Slice", ["g9p", "g10_s", "g10_e", "g10_ax"], "g10")     # [1,1,10,10]

    # ---- per-centre candidate red + size -----------------------------------
    def cand(gname, tag):
        # cand = g AND NOT blue.  Both are {0,1}, so g > blue holds exactly there.
        n("Greater", [gname, "blue"], f"cb_{tag}")            # bool [1,1,10,10]
        n("Cast", [f"cb_{tag}"], f"cf_{tag}", to=F16)         # fp16 {0,1} (for ReduceSum)
        n("ReduceSum", [f"cf_{tag}"], f"sz_{tag}", axes=[2, 3], keepdims=1)  # scalar
        n("Cast", [f"cb_{tag}"], f"cu_{tag}", to=U8)          # uint8 {0,1} (for select)
        return f"cu_{tag}", f"sz_{tag}"

    cu9, sz9 = cand("g9", "a")
    cu10, sz10 = cand("g10", "b")

    # ---- pick centre with smallest exposed-copy size -----------------------
    n("Not", [n("Greater", [sz9, sz10], "sz9_gt")], "pick")   # sz9 <= sz10
    n("Where", ["pick", cu9, cu10], "red10_u8")               # uint8 [1,1,10,10]

    # ---- pad 10x10 -> 30x30 cond -------------------------------------------
    init("pads", np.array([0, 0, 0, 0, 0, 0, N - G, N - G], np.int64), np.int64)
    init("ZEROU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["red10_u8", "pads", "ZEROU8"], "red30", mode="constant")  # [1,1,30,30] uint8
    n("Cast", ["red30"], "cond", to=BOOL)                               # [1,1,30,30] bool

    # ---- red one-hot (colour 2) constant [1,10,1,1] ------------------------
    red_oh = np.zeros((1, 10, 1, 1), dtype=np.float32)
    red_oh[0, 2, 0, 0] = 1.0
    init("red_onehot", red_oh, np.float32)

    # ---- output = Where(cond, red_onehot, input) : FREE [1,10,30,30] -------
    n("Where", ["cond", "red_onehot", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", F32, [1, 10, N, N])
    g = helper.make_graph(nodes, "task027", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
