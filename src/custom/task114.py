"""task114 (ARC-AGI 49d1d64f) — frame a small grid with edge-extended border.

Rule (from the ARC-GEN generator, verified fresh):
  A width x height grid (each 2..3) of colours from {1,2,3,4,8} sits at the
  ORIGIN (rows 0..h-1, cols 0..w-1) of the canvas; rest is background 0.  When
  width==height==3 the centre pixel is forced to background.  The output is a
  (h+2) x (w+2) grid:
    - the grid is copied into the centre:           out[1+r][1+c] = grid[r][c]
    - top row extends up:                           out[0][1+c]   = grid[0][c]
    - bottom row extends down:                      out[h+1][1+c] = grid[h-1][c]
    - left col extends left:                        out[1+r][0]   = grid[r][0]
    - right col extends right:                      out[1+r][w+1] = grid[r][w-1]
    - the four corners stay background.

IMPORTANT target detail (convert_to_numpy in src.harness): the one-hot target is
built only over the (h+2)x(w+2) OUTPUT grid, so cells OUTSIDE it have NO channel
set (all-zero), while in-grid background cells (the four corners, and a 3x3 black
centre) get channel 0 = 1.  So corners != off-region.

Construction (row/col-SEPARABLE index remap of a tiny 3x3 source, routed into the
FREE one-hot output — no full [1,10,30,30] plane ever materialised):
  The output cell (R,C) maps to source grid cell
      sr = clamp(R-1, 0, h-1)   (depends only on R, h)
      sc = clamp(C-1, 0, w-1)   (depends only on C, w)
  so out[R][C] = grid[sr][sc] EXCEPT corners -> 0, off-region -> all-zero.
  Corner = (R in {0,h+1}) AND (C in {0,w+1});  off-region = R>h+1 OR C>w+1.

  - Slice the input to the only-possible-active FOREGROUND 3x3 block
    [1,8,3,3] (channels 1..8 cover the colour set {1,2,3,4,8}; ch0 has weight 0
    so dropping it is free) — 288B fp32, the single dominant intermediate.
  - Channel-collapse to a colour-index plane val3 = sum_k k*input_k via a 1x1
    Conv, cast fp16.  h/w = #rows/#cols with a nonzero max colour (grid is
    contiguous from the origin and every edge cell is coloured -> exact count).
  - srcR/srcC = clamp(arange-1, 0, dim-1)  (fp16 Clip; fp16 Sub) -> int32 [5];
    Gather rows then cols of val3 -> remapped [1,1,5,5] plane P.
  - M5 = Where(outreg, -1, Where(corner, 0, P)): off-region -> -1 sentinel (the
    Equal below gives all-False there = no channel set), corner -> 0 (channel 0).
  - Equal(M5, arange[0..9]) -> [1,10,5,5] bool one-hot, then Pad(False) to
    [1,10,30,30] (opset-16 Pad accepts bool) = the FREE bool output.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30
SRC = 3   # max grid extent (width,height <= 3)
OUT = 5   # max output extent (h+2 <= 5)


def build(task):
    inits, nodes = [], []
    seen = set()

    def init(name, arr, dt):
        if name in seen:
            return name
        seen.add(name)
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16
    NI = TensorProto.INT32
    B = TensorProto.BOOL

    # ---- slice input to the only-possible active 3x3 corner of the FOREGROUND
    # channels 1..8 (drop background ch0: weight-0 anyway, and colours are 1..8) --
    init("s0", np.array([1, 0, 0], np.int64), np.int64)       # ch>=1, r>=0, c>=0
    init("sSRC", np.array([9, SRC, SRC], np.int64), np.int64)  # ch<9, r<3, c<3
    init("ax123", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "s0", "sSRC", "ax123"], "in3")       # [1,8,3,3] fp32 (dominant)
    chW = np.arange(1, 9, dtype=np.float32).reshape(1, 8, 1, 1)  # colours 1..8
    init("chW", chW, np.float32)
    n("Conv", ["in3", "chW"], "val3f")                        # [1,1,3,3] fp32 colour-index
    n("Cast", ["val3f"], "val3", to=H)                        # [1,1,3,3] fp16

    # ---- recover h, w as fp16 scalars ---------------------------------------
    # h = #rows with a nonzero colour (grid is contiguous from the origin and every
    # grid edge cell is coloured, so this counts exactly the grid height even with
    # the 3x3 black centre).  Use per-row/col MAX colour -> >0 -> count.
    init("zh", np.array(0.0, np.float16), np.float16)
    init("axall", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("ReduceMax", ["val3"], "rowmax", axes=[3], keepdims=1)     # [1,1,3,1] max colour/row
    n("Greater", ["rowmax", "zh"], "rowhasb")                    # bool
    n("Cast", ["rowhasb"], "rowhas", to=H)
    n("ReduceSum", ["rowhas", "axall"], "hF", keepdims=0)        # scalar fp16 = h
    n("ReduceMax", ["val3"], "colmax", axes=[2], keepdims=1)     # [1,1,1,3]
    n("Greater", ["colmax", "zh"], "colhasb")
    n("Cast", ["colhasb"], "colhas", to=H)
    n("ReduceSum", ["colhas", "axall"], "wF", keepdims=0)        # scalar fp16 = w

    # ---- source-index vectors srcR/srcC = clamp(arange-1, 0, dim-1) ----------
    # rowR is the OUTPUT row index 0..4 as a [1,1,5,1] fp16 column; colC [1,1,1,5].
    rowR = np.arange(OUT, dtype=np.float16).reshape(1, 1, OUT, 1)
    colC = np.arange(OUT, dtype=np.float16).reshape(1, 1, 1, OUT)
    init("rowR", rowR, np.float16)
    init("colC", colC, np.float16)
    init("oneH", np.array(1.0, np.float16), np.float16)
    # flat fp16 ramps [5] for index math (reshaped views of rowR/colC data)
    init("rampH", np.arange(OUT, dtype=np.float16), np.float16)   # [5] 0..4 fp16

    def srcidx(dimS, tag):
        # clamp(arange-1, 0, dim-1) in fp16 (Clip works under ORT_DISABLE_ALL)
        n("Sub", ["rampH", "oneH"], f"rm1_{tag}")             # [5] fp16 arange-1
        n("Sub", [dimS, "oneH"], f"hi_{tag}")                 # scalar fp16 dim-1
        n("Clip", [f"rm1_{tag}", "zh", f"hi_{tag}"], f"src_{tag}")  # [5] fp16
        n("Cast", [f"src_{tag}"], f"srci_{tag}", to=NI)       # [5] int32
        return f"srci_{tag}"

    srcR = srcidx("hF", "R")
    srcC = srcidx("wF", "C")

    # ---- gather rows then cols of val3 -> remapped [1,1,5,5] plane P ----------
    n("Gather", ["val3", srcR], "Pr", axis=2)                 # [1,1,5,3] fp16
    n("Gather", ["Pr", srcC], "P", axis=3)                    # [1,1,5,5] fp16

    # ---- keep mask (fp16 broadcast, only ONE 5x5 bool materialised) ----------
    # edgeR = (R==0) | (R==h+1); inR = R<=h+1.  notkeep = ~inR | ~inC | (edgeR&edgeC)
    n("Add", ["hF", "oneH"], "hp1")                           # h+1  (fp16 scalar)
    n("Add", ["wF", "oneH"], "wp1")
    # edge markers as fp16 0/1 columns via Equal->Cast
    n("Equal", ["rowR", "zh"], "r0")                          # R==0
    n("Equal", ["rowR", "hp1"], "rb")                         # R==h+1
    n("Or", ["r0", "rb"], "edgeR")                            # [1,1,5,1] bool
    n("Equal", ["colC", "zh"], "c0")
    n("Equal", ["colC", "wp1"], "cb")
    n("Or", ["c0", "cb"], "edgeC")                            # [1,1,1,5] bool
    n("Greater", ["rowR", "hp1"], "outR")                     # R>h+1  [1,1,5,1]
    n("Greater", ["colC", "wp1"], "outC")                     # C>w+1  [1,1,1,5]
    n("And", ["edgeR", "edgeC"], "corner")                    # [1,1,5,5] bool
    n("Or", ["outR", "outC"], "outreg")                       # [1,1,5,5] bool

    # Three cases: out-of-region -> -1 sentinel (no channel set, matches target's
    # all-zero cells outside the (h+2)x(w+2) grid); corner -> 0 (background colour,
    # channel 0 set); else -> the remapped colour P.
    init("negH", np.array(-1.0, np.float16), np.float16)
    n("Where", ["corner", "zh", "P"], "Mc")                   # corner -> 0
    n("Where", ["outreg", "negH", "Mc"], "M5")                # out-region -> -1

    # ---- one-hot at 5x5 then Pad(False) to 30x30 = FREE bool output ----------
    kvals = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("kvals", kvals, np.float16)
    n("Equal", ["M5", "kvals"], "oh5")                        # [1,10,5,5] bool
    init("pad30", np.array([0, 0, 0, 0, 0, 0, S - OUT, S - OUT], np.int64), np.int64)
    init("padF", np.array(False, np.bool_), np.bool_)
    n("Pad", ["oh5", "pad30", "padF"], "output")              # [1,10,30,30] bool (FREE)

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task114", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 16)])
