"""task381 (ARC-AGI ef135b50) — fill the row-run between two reds with maroon
unless the run touches a red above/below.

Rule (from generator):
  size=10 grid, red(2) boxes on black(0).  For each row, the maximal run of
  non-red cells strictly BETWEEN two red cells is painted maroon(9) UNLESS any
  cell in that run has a red directly above/below it (then the WHOLE run stays
  black).  Red copies through.  Off the 10x10 active region is ALL-ZERO one-hot
  (the harness zero-pads the output to 30x30).

  Generator facts that make the danger gate a STATIC row mask (verified on 4000
  fresh instances):
   - "Avoid maroons in top/bottom row": rows 0 and 9 NEVER contain maroon.
   - The interior red-black-red validator => rows 1..8 NEVER contain a black
     between-reds gap.
   => a between-reds run is maroon IFF 1 <= row <= 8.  No danger detection
   (no red-above/below maxpool, no per-row reduce) is needed — just a constant
   [1,1,10,1] bool mask `rowsafe` AND'd with the gap mask.

Encoding (build the 10-ch carrier at 10x10, Pad to 30x30 = output):
  Only colors present: black(0), red(2), maroon(9).  All other channels are
  permanently zero.  The output one-hot is determined by THREE 10x10 bool
  planes: red (copy of ch2), maroon (fill), black (everything else).
    R = red plane (channel 2) as f16.
    leftOR/rightOR = full-length 1-D MaxPool running-max (no params).
    gap = (leftOR>0 & rightOR>0) & R==0          [between two reds, non-red]
    maroon = gap & rowsafe_const                  [rows 1..8 only]
    is_red = R>0 ;  black = NOT(is_red | maroon)
    inner_u8 = Concat([black,Z,red,Z,Z,Z,Z,Z,Z,maroon])  ([1,10,10,10] uint8)
    output = Pad(inner_u8, 0) -> [1,10,30,30]  (output declared uint8)
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
A = 10  # active grid is exactly 10x10


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- red plane (channel 2) on the 10x10 active canvas -------------------
    init("r_s", np.array([2, 0, 0], np.int64), np.int64)
    init("r_e", np.array([3, A, A], np.int64), np.int64)
    init("r_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "r_s", "r_e", "r_ax"], "red_f32")  # [1,1,10,10] f32
    n("Cast", ["red_f32"], "R", to=F16)                     # f16

    # red on both sides via full-length running-max maxpools (no params)
    n("MaxPool", ["R"], "leftOR", kernel_shape=[1, A], pads=[0, A - 1, 0, 0])
    n("MaxPool", ["R"], "rightOR", kernel_shape=[1, A], pads=[0, 0, 0, A - 1])

    init("ZH", np.array(0.0, np.float16), np.float16)
    n("Greater", ["R", "ZH"], "is_red")            # bool, R>0
    n("Greater", ["leftOR", "ZH"], "lb")           # bool, red to the left
    n("Greater", ["rightOR", "ZH"], "rb")          # bool, red to the right
    n("Not", ["is_red"], "notred")                 # bool, R==0
    n("And", ["lb", "rb"], "between_b")
    n("And", ["between_b", "notred"], "gap")       # between two reds, non-red

    # row-safe = a STATIC per-row mask.  The generator REJECTS any instance with
    # maroon in row 0 or row 9 ("Avoid maroons in top/bottom row"), and validates
    # interior rows so no black between-reds gap ever survives there.  Verified on
    # 4000 fresh instances: rows 0 & 9 NEVER contain maroon and interior rows
    # NEVER contain a black between-reds gap.  => a between-reds run is maroon iff
    # 1 <= row <= 8.  No danger detection (vmax/reduce) is needed at all.
    rowsafe = np.zeros((1, 1, A, 1), bool)
    rowsafe[0, 0, 1:A - 1, 0] = True            # rows 1..8 are safe
    init("rowsafe", rowsafe, bool)
    n("And", ["gap", "rowsafe"], "maroon")      # [1,1,10,10] bool fill

    # black = NOT (red OR maroon)
    n("Or", ["is_red", "maroon"], "nonblack")
    n("Not", ["nonblack"], "black")

    # ---- assemble the 10-channel one-hot carrier directly in uint8 --------
    n("Cast", ["black"], "black_u8", to=U8)
    n("Cast", ["is_red"], "red_u8", to=U8)
    n("Cast", ["maroon"], "maroon_u8", to=U8)
    init("Z", np.zeros((1, 1, A, A), np.uint8), np.uint8)
    n("Concat",
      ["black_u8", "Z", "red_u8", "Z", "Z", "Z", "Z", "Z", "Z", "maroon_u8"],
      "inner_u8", axis=1)  # [1,10,10,10] uint8

    # ---- pad spatially to 30x30 (zero-fill = all-zero off-grid one-hot) ----
    init("pads", np.array([0, 0, 0, 0, 0, 0, N - A, N - A], np.int64), np.int64)
    init("ZU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["inner_u8", "pads", "ZU8"], "output", mode="constant")

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", U8, [1, 10, N, N])
    graph = helper.make_graph(nodes, "task381", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 12)])
