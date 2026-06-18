"""task160 (ARC-AGI 6c434453) — replace every hollow-box sprite with a red plus.

Rule (from generator task_6c434453.py, size=10 fixed):
  num=5 blue sprites are drawn on a 10x10 grid at non-overlapping, non-edge-adjacent
  positions.  Sprite types (idx): 0=box (hollow 3x3 ring, 8 blue cells, empty centre),
  1=plus, 2=minus, 3=square2x2, 4=corner-L, 5=corner-2x2.  ALL drawn blue in the INPUT.
  In the OUTPUT, every box (idx 0) is REPLACED by a red PLUS centred in the same 3x3
  region (the 5-cell plus: centre + 4 orthogonal neighbours), coloured red(2).  Every
  NON-box sprite is copied unchanged (same shape, blue).  Sprites are guaranteed
  isolated (pairwise gap >2 OR not edge-adjacent), so the only 3x3 neighbourhood with
  all 8 ring cells blue is a box.

  Exact encoding (verified fresh 200/200), colour-index plane L over the 10x10 region:
    B          = blue channel slice, [1,1,10,10] f16 {0,1}
    boxcentre  = (ringConv(B) == 8)            ring kernel = 3x3 ones, centre 0, SAME pad
                 (peak 8 ONLY at a box centre; isolated sprites can't fake it)
    boxring    = dilate(boxcentre, ring kernel) > 0   the 8 blue cells to erase
    plusmask   = dilate(boxcentre, plus kernel) > 0   the 5 red plus cells
    blue_out   = B AND NOT boxring
    red_out    = plusmask
    L          = 1*blue_out + 2*red_out   (disjoint: red plus cells were erased from blue)
  Pad L to 30x30 with sentinel 255 (off-grid matches no colour -> all-zero target),
  output = Equal(L_u8, arange[0..9]) -> BOOL [1,10,30,30] routed into the FREE output.

  All full-canvas work is on the 10x10 active region in fp16.

  pts 16.94, mem 3100, params 51, fresh 200/200.  Dominant intermediate = the 30x30
  uint8 sentinel-padded L carrier (900B) + the one-time fp32 blue slice (400B); the
  ring/plus detection+dilation convs are tiny 10x10 fp16/bool planes.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

N = 10  # active grid is always 10x10


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- slice blue channel (1) on the 10x10 active region -----------------
    init("sl_s", np.array([1, 0, 0], np.int64), np.int64)
    init("sl_e", np.array([2, N, N], np.int64), np.int64)
    init("sl_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "sl_s", "sl_e", "sl_ax"], "B_f32")   # [1,1,10,10] f32
    n("Cast", ["B_f32"], "B", to=F16)                          # [1,1,10,10] f16 {0,1}

    # ---- box centre = 3x3 ring sum == 8 (SAME pad) -------------------------
    ring = np.ones((1, 1, 3, 3), np.float16)
    ring[0, 0, 1, 1] = 0.0
    init("Kring", ring, np.float16)
    n("Conv", ["B", "Kring"], "ringcv", pads=[1, 1, 1, 1])     # [1,1,10,10] f16
    init("EIGHT", np.array(8.0, np.float16), np.float16)
    n("Equal", ["ringcv", "EIGHT"], "boxc_b")                  # bool [1,1,10,10]
    n("Cast", ["boxc_b"], "boxc", to=F16)                      # f16 {0,1}

    # ---- dilate box centre by the ring (cells to erase) --------------------
    n("Conv", ["boxc", "Kring"], "ringdil", pads=[1, 1, 1, 1])  # [1,1,10,10]
    init("ZH", np.array(0.0, np.float16), np.float16)
    n("Greater", ["ringdil", "ZH"], "boxring_b")               # bool: 8 ring cells

    # ---- dilate box centre by the plus shape (red plus cells) --------------
    plus = np.zeros((1, 1, 3, 3), np.float16)
    plus[0, 0, 1, 1] = 1.0
    plus[0, 0, 0, 1] = 1.0
    plus[0, 0, 2, 1] = 1.0
    plus[0, 0, 1, 0] = 1.0
    plus[0, 0, 1, 2] = 1.0
    init("Kplus", plus, np.float16)
    n("Conv", ["boxc", "Kplus"], "plusdil", pads=[1, 1, 1, 1])  # [1,1,10,10]
    n("Greater", ["plusdil", "ZH"], "red_b")                   # bool: 5 plus cells

    # ---- blue_out = B AND NOT boxring ; red_out = plusmask -----------------
    n("Cast", ["B"], "B_b", to=BOOL)                           # blue bool
    n("Not", ["boxring_b"], "notring")
    n("And", ["B_b", "notring"], "blue_b")                     # bool blue_out

    # ---- L = Where(red, 2, Where(blue, 1, 0)) over uint8 -------------------
    init("U0", np.array(0, np.uint8), np.uint8)
    init("U1", np.array(1, np.uint8), np.uint8)
    init("U2", np.array(2, np.uint8), np.uint8)
    n("Where", ["blue_b", "U1", "U0"], "L_blue")               # uint8 {0,1}
    n("Where", ["red_b", "U2", "L_blue"], "L_u8")              # uint8 {0,1,2}

    # ---- pad L to 30x30 with sentinel 255 ----------------------------------
    init("Lpads", np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64), np.int64)
    init("SENT", np.array(255, np.uint8), np.uint8)
    n("Pad", ["L_u8", "Lpads", "SENT"], "L30", mode="constant")  # [1,1,30,30] u8

    # ---- output = Equal(L, arange[0..9]) -> BOOL [1,10,30,30] (FREE) -------
    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "arange"], "output")                    # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task160", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
