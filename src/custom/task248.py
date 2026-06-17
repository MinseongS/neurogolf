"""task248 (ARC-AGI a3df8b1e) — bounce path of a ball in a 10 x W grid.

Rule (from the ARC-GEN generator `common.bounce`, verified fresh):
  Input is a 10 x W grid (W in 2..10), all black (channel 0) except a single
  blue pixel (channel 1) at the bottom-left corner (row 9, col 0).  Only W
  varies between instances; the input is otherwise fixed (height is always 10).

  OUTPUT is a 10 x W grid: a blue ball starts at (row 9, col 0) and bounces.
  Going UP one row per step, the column follows a triangle wave reflecting off
  col 0 and col W-1.  With step s = 9 - r (s = 0 at the bottom row),
      period p = 2*(W-1),  m = s mod p,  c_path(r) = min(m, p - m).
  Output cell (r, c) is blue iff c == c_path(r) (within the grid), else black.

Closed-form reconstruction (no colour-index Conv, no flood/scan):
  Width W is the per-column occupancy span: colocc[c] = ReduceMax(input over
  channels & rows) = 1 iff c < W;  W = sum(colocc).  Rows are always 0..9 so
  the row mask is a CONSTANT, baked into init tensors (no row reductions).

  Per-row path column pcol[r] = min(m, p-m) with s = max(9-r, 0) baked as a
  const vector; p = 2*(W-1) from the width scalar.  A const row-offset bumps
  rows r>=10 to a >=99 sentinel so no column matches them (off-grid rows stay
  blank).  Then  pathB[r,c] = (c == pcolG[r])  is the single blue plane.

  One full-plane Sum band-routes the 10-channel one-hot into the FREE output:
      L = 2*path + colin + rowConst        (path, colin in {0,1})
        blue  (r<10, c=pcol)  : 2 + 1 + 1 = 4   -> channel 1
        black (r<10, c<W)     : 0 + 1 + 1 = 2   -> channel 0
        r<10, c>=W            : 0 + 0 + 1 = 1   -> no channel
        r>=10, c<W            : 0 + 1 + 0 = 1   -> no channel
      output = Equal(L, band[1,10,1,1]),  band[0]=2, band[1]=4, others = 100.

  Full planes (fp16/bool): pathB (bool), pathF (fp16), L (fp16) — everything
  else is <=120B vectors / scalars / free constant inits.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30
H = 10  # grid height is always 10


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H16 = TensorProto.FLOAT16
    B = TensorProto.BOOL

    # ---- width scalar from per-column occupancy -----------------------------
    n("ReduceMax", ["input"], "colocc", axes=[1, 2], keepdims=1)   # [1,1,1,30] fp32
    init("halfF", np.array(0.5, np.float32), np.float32)
    n("Greater", ["colocc", "halfF"], "colinB")   # [1,1,1,30] bool  c < W
    n("Cast", ["colinB"], "colinF", to=H16)        # [1,1,1,30] fp16 (tiny)
    n("ReduceSum", ["colocc"], "Wsc", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1] fp32

    # ---- per-row path column pcol[r] = min(m, p-m),  s = max(9-r, 0) --------
    # Computed in fp16 (tiny [1,1,30,1] vectors).  Min avoided via the identity
    #   min(m, p-m) = p/2 - |m - p/2|   (Abs is fp16-safe; Min is not).
    sconst = np.maximum(9 - np.arange(S), 0).astype(np.float16).reshape(1, 1, S, 1)
    init("sconst", sconst, np.float16)             # baked: depends only on r
    init("oneF32", np.array(1.0, np.float32), np.float32)

    n("Sub", ["Wsc", "oneF32"], "Wm1")             # [1,1,1,1] fp32  W-1
    n("Cast", ["Wm1"], "Wm1H", to=H16)             # [1,1,1,1] fp16
    # p = 2(W-1), half_p = (W-1)
    n("Add", ["Wm1H", "Wm1H"], "p")                # [1,1,1,1] fp16  p = 2(W-1)
    n("Mod", ["sconst", "p"], "m", fmod=1)         # [1,1,30,1] fp16
    n("Sub", ["m", "Wm1H"], "t")                   # [1,1,30,1]  m - p/2
    n("Abs", ["t"], "at")                          # [1,1,30,1]  |m - p/2|
    n("Sub", ["Wm1H", "at"], "pcol")               # [1,1,30,1]  p/2 - |..| = min

    # bump rows r>=10 to a >=99 sentinel so the OneHot index is out of range
    # (-> that row gets no blue cell; off-grid rows stay blank).
    rowoff = np.where(np.arange(S) < H, 0.0, 99.0).astype(np.float16).reshape(1, 1, S, 1)
    init("rowoff", rowoff, np.float16)
    n("Add", ["pcol", "rowoff"], "pcolG")          # [1,1,30,1] fp16

    # ---- blue plane via OneHot (NO comparison bool, NO numeric-path cast) ---
    #   blueval[r,c] = 2 iff c == pcolG[r] (else 0); the on/off values are baked.
    n("Cast", ["pcolG"], "pcolGi", to=TensorProto.INT64)  # [1,1,30,1] int64
    n("Squeeze", ["pcolGi"], "pcolIdx", axes=[3])         # [1,1,30] int64
    init("depth30", np.array(S, np.int64), np.int64)
    init("onoff", np.array([0.0, 2.0], np.float16), np.float16)  # [off, on] fp16
    n("OneHot", ["pcolIdx", "depth30", "onoff"], "blueval", axis=-1)  # [1,1,30,30] fp16

    # ---- band-routed colour-index plane L (ONE full-plane Sum) -------------
    #   L = 2*path + colin + rowConst   (colin, rowConst in {0,1}; rowConst const)
    #     blue  (r<10, c=pcol) : 2 + 1 + 1 = 4   -> channel 1
    #     black (r<10, c<W)    : 0 + 1 + 1 = 2   -> channel 0
    #     r<10, c>=W           : 0 + 0 + 1 = 1   -> no channel
    #     r>=10, c<W           : 0 + 1 + 0 = 1   -> no channel
    rowc = np.where(np.arange(S) < H, 1.0, 0.0).astype(np.float16).reshape(1, 1, S, 1)
    init("rowConst", rowc, np.float16)
    n("Sum", ["blueval", "colinF", "rowConst"], "L")  # [1,1,30,30] fp16

    # ---- route 10-channel one-hot into FREE bool output --------------------
    band = np.full((10,), 100.0, np.float16)
    band[0] = 2.0   # black
    band[1] = 4.0   # blue
    init("band10", band.reshape(1, 10, 1, 1), np.float16)
    n("Equal", ["L", "band10"], "output")          # [1,10,30,30] bool

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task248", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
