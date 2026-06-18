"""task338 (ARC-AGI d5d6de2d) — "fill the hollow interior of each red box with green".

Rule (from the generator):
  A size x size grid (size = 5*mult, mult in [2,5] -> size in {10,15,20,25}), background
  black(0).  Several non-overlapping (separated by >=1) solid red(2) rectangles are drawn,
  each then has its INTERIOR (the (tall-2) x (wide-2) inner block) reset to black(0) — i.e.
  every box is a 1-cell-thick red ring around a black hole.  In the OUTPUT the interior holes
  become green(3); everything else (red ring + outside background) -> black(0).

  Closed-form, NOT a flood-fill (the task204 column ray-cast):
    red = input==2
    Hm  = a HORIZONTAL-WALL cell = red cell with red neighbours BOTH left and right
          (3 consecutive reds) -> fires only on top/bottom box edges, NOT on the 1-wide
          vertical side walls, NOT on isolated/gap reds.
    enc = PARITY of Hm cells strictly ABOVE each cell (column ray-cast: tril MatMul + Mod-2).
          An interior cell sits below exactly ONE horizontal wall (its box's top edge) -> odd.
          Outside / gap cells see 0 or 2 horizontal walls -> even.  Separated boxes never merge
          because Hm marks only genuine horizontal edges, so the crossing count stays local.
    green = enc AND NOT red                  (the interior hole)
  Output: green->3, on-grid bg->0 (black), off-grid->nothing (sentinel).

Encoding (route the one-hot into the FREE output):
  Active canvas W=25 (size <= 25 always); off-canvas padded with sentinel 99.
    red_f32 = input[:, 2:3, 0:W, 0:W]  [1,1,25,25] f32
  Hm via a 1x3 sum-Conv (bias -2, Relu) ; enc via lower-triangular MatMul + Mod-2 ;
  in-grid mask via 1-D occupancy profiles ; L (u8) Pad->30x30 ; Equal(L, arange) -> BOOL.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8

W = 25  # active canvas (size <= 25 always)
G = 30  # full ONNX canvas


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- red plane (channel 2) on the 25x25 canvas -------------------------
    init("r_s", np.array([2, 0, 0], np.int64), np.int64)
    init("r_e", np.array([3, W, W], np.int64), np.int64)
    init("r_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "r_s", "r_e", "r_ax"], "red_f")   # [1,1,25,25] f32
    n("Cast", ["red_f"], "red", to=F16)                    # f16 {0,1} (kills fp32 downstream)
    init("half", np.array(0.5, np.float16), np.float16)
    n("Greater", ["red", "half"], "redb")                  # bool red mask

    # ---- Hm = horizontal-wall cell (red with red L & R neighbour) ----------
    # 1x3 all-ones Conv (SAME-pad horizontally) + bias -2, Relu -> 1 iff all three red.
    Wc = np.ones((1, 1, 1, 3), np.float16)
    init("Wc", Wc, np.float16)
    init("bm2", np.array([-2.0], np.float16), np.float16)
    n("Conv", ["red", "Wc", "bm2"], "c1h", pads=[0, 1, 0, 1])  # fp16 [1,1,25,25]
    n("Relu", ["c1h"], "Hm")                               # fp16 {0,1}

    init("q", np.array(0.5, np.float16), np.float16)
    init("two", np.array(2.0, np.float16), np.float16)

    # ---- enc = parity of Hm strictly above (column ray-cast) ---------------
    Tl = np.tril(np.ones((W, W), np.float16), -1)          # Tl[r,r']=1 iff r'<r
    init("Tl", Tl.reshape(1, 1, W, W), np.float16)
    n("MatMul", ["Tl", "Hm"], "cnt")                       # count of Hm above [1,1,25,25]
    n("Mod", ["cnt", "two"], "par", fmod=1)                # f16 {0,1} (integer-exact)
    n("Greater", ["par", "q"], "encb")                     # bool: odd crossings above

    # ---- green = enc AND NOT red -------------------------------------------
    n("Not", ["redb"], "notred")
    n("And", ["encb", "notred"], "greenb")                 # bool interior hole

    # ---- in-grid mask via 1-D occupancy profiles ---------------------------
    # The grid is a solid size x size square (bg sets ch0=1 everywhere in-grid), so
    # in-grid = rowany (x) colany — two tiny profiles, no 3600B channel reduce.
    n("ReduceMax", ["input"], "rowp30", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["input"], "colp30", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    init("z1", np.array([0], np.int64), np.int64)
    init("wn", np.array([W], np.int64), np.int64)
    init("a2", np.array([2], np.int64), np.int64)
    init("a3", np.array([3], np.int64), np.int64)
    n("Slice", ["rowp30", "z1", "wn", "a2"], "rowp")       # [1,1,W,1]
    n("Slice", ["colp30", "z1", "wn", "a3"], "colp")       # [1,1,1,W]
    init("halff", np.array(0.5, np.float32), np.float32)
    n("Greater", ["rowp", "halff"], "rowb")
    n("Greater", ["colp", "halff"], "colb")
    n("And", ["rowb", "colb"], "gridb")                    # [1,1,W,W] bool in-grid

    # ---- uint8 label map: green->3, in-grid bg->0, off-grid->99 ------------
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u3", np.array(3, np.uint8), np.uint8)
    init("u99", np.array(99, np.uint8), np.uint8)
    n("Where", ["gridb", "u0", "u99"], "Lg")               # 0 in-grid else 99
    n("Where", ["greenb", "u3", "Lg"], "L25")              # interior overrides -> 3

    # ---- pad to 30x30 with sentinel 99 (off-canvas -> no channel) ----------
    init("pads", np.array([0, 0, 0, 0, 0, 0, G - W, G - W], np.int64), np.int64)
    n("Pad", ["L25", "pads", "u99"], "L30", mode="constant")  # [1,1,30,30] u8

    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "arange"], "output")                # BOOL [1,10,30,30]

    graph = helper.make_graph(
        nodes, "task338",
        [helper.make_tensor_value_info("input", F32, [1, 10, G, G])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, G, G])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
