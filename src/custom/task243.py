"""task243 (ARC-AGI 9edfc990) — 4-connected blue flood-fill into background.

Rule (from the generator):
  A size x size grid (size in 12..18) is filled with random colours (~50% black
  background, colour 0; the rest in 1..9).  NB the generator's `common.blue()` is
  colour index **1** (NOT 2).  Every blue (colour 1) cell is a BFS seed.  The
  output equals the input, except that blue floods 4-connectedly into every
  BACKGROUND (black, colour 0) cell reachable from a seed (any non-bg cell blocks
  the flood).  Flooded background cells become blue; all other cells are unchanged.

  This is a genuine 4-connectivity flood over a variable-size grid — no closed-form
  / separable escape.  The active canvas is gen-bounded to <=18x18 (top-left
  anchored), so the flood runs on a tiny [1,1,18,18] plane.

Encoding (bounded BFS, fp16 cross-Conv flood — task048 idiom):
  Slice bg(ch0) and blue(ch1) to 18x18 (fp32 Slice), Cast to fp16.  Off-grid cells
    have an all-zero one-hot (ch0=0 too), so they are NOT bg/passable and the flood
    cannot leak off-grid -- no in-grid mask needed.  passable = bg + blue.
  reach0 = blue seeds.
  per round:  count = Conv_cross(reach,[[0,1,0],[1,1,1],[0,1,0]])   (4-neighbour+self)
              reach = Min(passable, count)   (passable AND count>=1, stays {0,1})
    -- ONE Conv does the whole 4-neighbour dilate, so only 2 planes/round (count,
       reach).  An equivalent uint8 MaxPool flood needs 4 planes/round (pv,ph,Max,Min)
       at half the bytes => identical 1296B/round, but more overhead; fp16 Conv wins.
       (uint8 Conv is INVALID_GRAPH even at opset 18, so MaxPool is the uint8 path.)
  The FINAL round gates to bg instead of passable, so its output IS the flooded-bg
    mask (reach AND bg) -- fuses the reach and flooded planes into one Min.
  N_ROUNDS = 38 covers the worst-case BFS geodesic (max 38 over 200000 fresh; frac
    D>33 = 6e-5).  Output = Where(flooded_mask, blue_onehot, input)  (FREE output).

Dominant intermediate: the N_ROUNDS x {Conv + Min} fp16 [1,1,18,18] planes (648B each,
~49KB).  Irreducible: 4-connectivity forces a cross dilate every step; 1-cell walls
force radius-1 (a larger kernel leaks across a 1-thick wall); D=38 is data-required
(worst-case geodesic); fp16 is the dtype floor (ORT has no uint8/bool Conv even at
opset 18).  The two fp32 channel slices (1296B each) are the only non-flood planes.

FLOOD-AT-FLOOR: bare flood planes at robust D=39 alone = 50544B already exceed the
exp(25-14.16)=51021B budget for +0.3, so full-robustness +0.3 over 13.86 is mathematically
impossible.  This robust net scores 14.06 (+0.20, MARGINAL) and passes fresh 500/500.
Reaching +0.3 requires undershooting to D=28-33, which silently leaks the high-D tail
(D=28: 7/20000 fresh; D=33: 2/20000) -- a non-generalizing trade rejected here.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

WORK = 18        # max active grid side (generator size <= 18, top-left anchored)
N_ROUNDS = 38    # worst-case BFS geodesic = 38 over 200000 fresh instances


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- crop bg(ch0) and blue(ch1) to the WORKxWORK active region -----------
    # Direct 2-channel slicing beats the colf-Conv pack here: this flood needs only
    # 2 channels, so a colour-index plane (3600B colf + 1296B slice) is pure overhead.
    init("axes",   np.array([1, 2, 3], np.int64), np.int64)
    init("steps",  np.array([1, 1, 1], np.int64), np.int64)
    init("starts0", np.array([0, 0, 0], np.int64), np.int64)
    init("ends0",   np.array([1, WORK, WORK], np.int64), np.int64)
    init("starts1", np.array([1, 0, 0], np.int64), np.int64)   # blue = channel 1
    init("ends1",   np.array([2, WORK, WORK], np.int64), np.int64)
    n("Slice", ["input", "starts0", "ends0", "axes", "steps"], "bg32")    # [1,1,W,W] f32
    n("Slice", ["input", "starts1", "ends1", "axes", "steps"], "blue32")  # [1,1,W,W] f32
    n("Cast", ["bg32"], "bg16", to=F16)                                   # [1,1,W,W] f16
    n("Cast", ["blue32"], "blue16", to=F16)                               # [1,1,W,W] f16
    n("Sum", ["bg16", "blue16"], "passable")                              # [1,1,W,W] f16 (bg OR blue)

    # ---- bounded BFS: count = Conv_cross(reach); reach = Min(passable, count) --
    init("flood_kernel",
         np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.float16).reshape(1, 1, 3, 3),
         np.float16)
    cur = "blue16"  # reach0 = blue seeds
    for i in range(N_ROUNDS):
        cnt = n("Conv", [cur, "flood_kernel"], f"count{i}", pads=[1, 1, 1, 1])
        # final round gates to bg16: Min(bg, cnt) == reach AND bg == flooded-bg mask
        gate = "bg16" if i == N_ROUNDS - 1 else "passable"
        cur = n("Min", [gate, cnt], f"reach{i + 1}")                      # [1,1,W,W] f16 {0,1}

    flooded16 = cur                                                       # [1,1,W,W] flooded bg
    # pad back to full 30x30, then threshold -> bool for Where
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("zero16", np.array(0.0, np.float16), np.float16)
    n("Pad", [flooded16, "pads", "zero16"], "flooded_full")               # [1,1,30,30] f16
    init("half", np.array(0.5, np.float16), np.float16)
    n("Greater", ["flooded_full", "half"], "flooded_mask")                # [1,1,30,30] bool

    # ---- output = Where(flooded_mask, blue_onehot, input) ---------------------
    blue_onehot = np.zeros((1, 10, 1, 1), np.float32)
    blue_onehot[0, 1, 0, 0] = 1.0   # blue = channel 1
    init("blue_onehot", blue_onehot, np.float32)
    n("Where", ["flooded_mask", "blue_onehot", "input"], "output")        # [1,10,30,30] f32

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task243", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
