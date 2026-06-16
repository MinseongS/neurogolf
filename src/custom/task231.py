"""task231 (ARC-AGI 963e52fc) — horizontally tile a periodic pattern to 2x width.

Rule (from the ARC-GEN generator, verified fresh):
  Input is a 5 x W grid (W in 6..10).  A periodic stripe pattern fills `tall`
  (1..2) consecutive rows starting at `offset` (1..2); the pattern repeats
  horizontally with period `wide` (2..3) and spans the whole width:
      cell(r, c) = colors[(r % tall) * wide + c % wide]   (within the pattern rows)
  The OUTPUT is a 5 x (2*W) grid: the SAME periodic pattern extended to double
  width:  out(r, c) = colors[(r % tall)*wide + c % wide]  for c in [0, 2W).

Closed-form reconstruction (no big planes):
  Because the input already holds a full period in columns [0, wide) <= [0, W),
  out(r, c) = input(r, c mod wide).  So the whole output is a column GATHER of
  the input:  output = Gather(input, src, axis=3), with
      src[c] = (c mod period)      if c < 2W   (period in {2,3})
      src[c] = 29 (a guaranteed-empty pad column)   if c >= 2W
  Gather copies whole columns, so rows >= 5 (all-zero in the padded input) stay
  zero for free, and pad columns pull the all-zero column 29 -> correct padding.

  period detection (period 2 vs 3): a per-column injective signature
      colsig[c] = sum_r rowW[r] * colorindex(r, c),   rowW = (1,10,100,1000,1e4)
  (distinct row weights make the column's colour-tuple recoverable, so two
  columns collide iff they are identical).  The pattern is period-2 iff
  colsig[c] == colsig[c+2] for every occupied column pair (c, c+2 both < W).
  A genuinely-period-3 pattern can only match this when every row is constant,
  in which case tiling by 2 reproduces the identical grid anyway -> safe.

  2W extent: colocc[c] = 1 iff c < W (per-column occupancy).  keep[c] = 1 iff
  c < 2W  ==  colocc[floor(c/2)]  (a Gather by the const [0,0,1,1,...] index).

Memory: the only sizeable intermediate is the row-contraction MatMul
[1,10,1,30] (1200B); everything else is [30] / [1,1,1,30] vectors.  The final
Gather IS the output (free).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    NI = TensorProto.INT32
    B = TensorProto.BOOL

    # ---- per-column occupancy: colocc[c]=1 iff c < W ------------------------
    n("ReduceMax", ["input"], "colocc4", axes=[1, 2], keepdims=1)  # [1,1,1,30] fp32
    init("flat", np.array([S], np.int64), np.int64)
    n("Reshape", ["colocc4", "flat"], "coloccV")                   # [30] fp32

    # ---- injective per-column colour signature -----------------------------
    # rowW[r]*colorindex(r,c) summed over rows & channels.
    rowW = np.zeros((1, 1, 1, S), np.float32)
    rowW[0, 0, 0, :5] = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    init("rowW", rowW, np.float32)
    n("MatMul", ["rowW", "input"], "rowwt")          # [1,10,1,30] fp32  (contract rows)
    chW = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("chW", chW, np.float32)
    n("Conv", ["rowwt", "chW"], "colsig4")           # [1,1,1,30] fp32  (weight channels)
    n("Reshape", ["colsig4", "flat"], "colsigV")     # [30] fp32

    # ---- period-2 test: colsig[c] == colsig[c+2] for every occupied pair ----
    init("shift2", np.array([min(c + 2, S - 1) for c in range(S)], np.int32), np.int32)
    n("Gather", ["colsigV", "shift2"], "colsigSh", axis=0)   # [30] colsig[c+2]
    n("Gather", ["coloccV", "shift2"], "coloccSh", axis=0)   # [30] occ[c+2]
    # mismatch on a pair that is in-range (c+2 < W, i.e. occ[c+2]==1)
    n("Equal", ["colsigV", "colsigSh"], "sameB")            # [30] bool
    n("Not", ["sameB"], "diffB")                            # [30] bool
    init("halfF", np.array(0.5, np.float32), np.float32)
    n("Greater", ["coloccSh", "halfF"], "occShB")           # [30] bool (c+2 < W)
    n("And", ["diffB", "occShB"], "badB")                   # [30] bool
    n("Cast", ["badB"], "badF", to=F)                       # [30] fp32
    n("ReduceSum", ["badF"], "nbad", axes=[0], keepdims=1)  # [1] fp32
    init("zeroF", np.array(0.0, np.float32), np.float32)
    n("Equal", ["nbad", "zeroF"], "p2B")                    # [1] bool  (period 2?)

    # ---- per-output-column source index ------------------------------------
    mod2 = np.array([c % 2 for c in range(S)], np.int32)
    mod3 = np.array([c % 3 for c in range(S)], np.int32)
    init("mod2", mod2, np.int32)
    init("mod3", mod3, np.int32)
    n("Where", ["p2B", "mod2", "mod3"], "periodIdx")        # [30] int32 (broadcast scalar)

    # keep[c] = 1 iff c < 2W == colocc[floor(c/2)]
    half = np.array([c // 2 for c in range(S)], np.int32)
    init("halfIdx", half, np.int32)
    n("Gather", ["coloccV", "halfIdx"], "keepF", axis=0)    # [30] fp32 (1 iff c<2W)
    n("Greater", ["keepF", "halfF"], "keepB")               # [30] bool
    init("pad29", np.full((S,), S - 1, np.int32), np.int32)  # all 29 (empty pad col)
    n("Where", ["keepB", "periodIdx", "pad29"], "srcIdx")   # [30] int32

    # ---- the whole output is a single column gather (FREE) ------------------
    n("Gather", ["input", "srcIdx"], "output", axis=3)      # [1,10,30,30] fp32

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task231", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
