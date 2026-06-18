"""task072 (ARC-AGI 3428a4f5) — XOR of two stacked 6x5 red panels -> green.

Rule (from generator task_3428a4f5.py):
  Input is a 13x5 grid: top panel = rows 0..5, a yellow(4) separator at row 6,
  bottom panel = rows 7..12.  Each panel holds red(2) pixels on a black(0) canvas.
  Output is a 6x5 grid: cell (r,c) = green(3) iff EXACTLY ONE of {top has red at
  (r,c), bottom has red at (r,c)} is set, else black(0).  i.e. green = XOR of the
  two panels' red masks.

Encoding (route 10-ch expansion as the FINAL Pad's input on the tiny 6x5 canvas;
never touch a 30x30 working plane):
  top  = input[:, 2:3, 0:6, 0:5]      red plane, top panel       [1,1,6,5] f32
  bot  = input[:, 2:3, 7:13, 0:5]     red plane, bottom panel    [1,1,6,5] f32
  same = Equal(top, bot)              bool: 1 where both/neither red (=> NOT green)
  green= Not(same)                    bool: 1 where exactly one red (=> green)
  Cast green, same -> uint8 {0,1}.  bg(ch0)=same (black where no green).
  stack = Concat[ bg, Z,Z, green, Z,Z,Z,Z,Z,Z ]  -> [1,10,6,5] uint8 (300B)
          (Z = a single shared [1,1,6,5] uint8 zero init, reused 7x)
  output= Pad(stack, [0,0,0,0, 0,0,24,25])  -> [1,10,30,30] uint8 (FREE output).
  Off-grid cells are all-zero (target output grid is only 6x5), matching the
  generator.  Scorer thresholds output>0 so uint8 {0,1} passes exactly while
  halving every working plane vs fp16.

  Dominant intermediate: the [1,10,6,5] uint8 stack = 300B (the irreducible 10-ch
  expansion on the active 6x5 region); two fp32 red slices 120B each are the only
  other notable tensors.  Beats the public Concat-of-fp16 net (mem 1260 -> ~660).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

RED = 2          # red channel
H = 6            # output / panel height
Wc = 5           # width
TOP0, TOP1 = 0, 6
BOT0, BOT1 = 7, 13


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- slice the red plane (channel 2) for both panels -------------------
    init("ax", np.array([1, 2, 3], np.int64), np.int64)
    init("t_s", np.array([RED, TOP0, 0], np.int64), np.int64)
    init("t_e", np.array([RED + 1, TOP1, Wc], np.int64), np.int64)
    init("b_s", np.array([RED, BOT0, 0], np.int64), np.int64)
    init("b_e", np.array([RED + 1, BOT1, Wc], np.int64), np.int64)
    n("Slice", ["input", "t_s", "t_e", "ax"], "top")   # [1,1,6,5] f32
    n("Slice", ["input", "b_s", "b_e", "ax"], "bot")   # [1,1,6,5] f32

    # ---- XOR via Equal/Not -------------------------------------------------
    n("Equal", ["top", "bot"], "same")   # bool: 1 where NOT green (bg)
    n("Not", ["same"], "green_b")        # bool: 1 where green

    n("Cast", ["same"], "bg_u8", to=U8)        # [1,1,6,5] uint8  (channel 0)
    n("Cast", ["green_b"], "green_u8", to=U8)  # [1,1,6,5] uint8  (channel 3)

    # ---- single shared zero plane reused in the Concat ---------------------
    init("Z", np.zeros((1, 1, H, Wc), np.uint8), np.uint8)

    n("Concat",
      ["bg_u8", "Z", "Z", "green_u8", "Z", "Z", "Z", "Z", "Z", "Z"],
      "stack", axis=1)                          # [1,10,6,5] uint8

    # ---- pad to 30x30 (off-grid = all-zero, matches generator) -------------
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - H, 30 - Wc], np.int64), np.int64)
    init("ZU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["stack", "pads", "ZU8"], "output", mode="constant")  # [1,10,30,30] u8

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task072", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
