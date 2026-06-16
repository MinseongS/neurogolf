"""task343 (ARC-AGI d8c310e9) — horizontally extend a periodic stripe pattern.

Rule (from the ARC-GEN generator, verified fresh 200/200):
  Input/output are 5 x 15 grids (embedded in the 30x30 one-hot canvas).  A
  vertical stripe block of width `len(lengths)` (3 or 4) is laid down column by
  column; with `flip=1` every other block is column-reversed, so the pattern is
  periodic with EFFECTIVE PERIOD  EP = len(lengths)        (flip = 0)
                                  EP = 2 * len(lengths)    (flip = 1).
  Only `visible` columns are drawn in the input; the OUTPUT tiles the same
  periodic pattern across all 15 columns.  Because the input already holds a
  full effective period in columns [0, EP), the closed form is

      out(r, c) = input(r, c mod EP)          (verified 3000/3000)

  so the ENTIRE output is one column GATHER of the input (FREE).

Period detection (smallest consistent period over the visible extent):
  - per-column injective signature  colsig[c] = sum_r rowW[r] * colorindex(r,c)
    (5 distinct row weights => a column's colour-tuple is recoverable, so two
    columns collide iff identical).
  - per-column occupancy occ[c] = 1 iff column c has any non-background pixel
    (ReduceMax over channels 1..9), which marks the visible extent.
  - For each candidate period p in 1..15 the period is CONSISTENT iff there is
    no occupied column c with colsig[c] != colsig[c mod p].  EP = the smallest
    consistent p.  (The first EP columns are always occupied, verified.)
  - srcIdx[c] = c mod EP, fed straight into Gather(input, srcIdx, axis=3).

Memory: the only sizeable intermediate is the row-contraction MatMul
[1,10,1,30] (1200B); everything else is tiny [15]/[15,15] vectors. The final
Gather IS the output (free).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30          # full canvas
W = 15          # active grid width
MAXP = 8        # max candidate period to test (largest real EP = 8, verified)


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

    # ---- injective per-column colour signature -----------------------------
    # colsig[c] = sum_{ch,r} chW[ch] * rowW[r] * input[ch,r,c]; colorindex is 0
    # for background, so colsig[c] > 0 iff column c has a non-background pixel.
    # A SINGLE Conv (kernel [1,10,30,1], no pad) contracts BOTH the channel and
    # row axes at once -> [1,1,1,30] directly, with no [1,10,1,30] intermediate.
    rowW = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0] + [0.0] * (S - 5), np.float32)
    chW = np.arange(10, dtype=np.float32)
    kern = np.zeros((1, 10, S, 1), np.float32)
    for ch in range(10):
        for r in range(S):
            kern[0, ch, r, 0] = chW[ch] * rowW[r]
    init("sigK", kern, np.float32)
    n("Conv", ["input", "sigK"], "colsig4")          # [1,1,1,30] fp32  (contract ch & rows)
    init("flatS", np.array([S], np.int64), np.int64)
    n("Reshape", ["colsig4", "flatS"], "colsigFull") # [30] fp32
    init("colstart", np.array([0], np.int64), np.int64)
    init("colW", np.array([W], np.int64), np.int64)
    init("ax0", np.array([0], np.int64), np.int64)
    n("Slice", ["colsigFull", "colstart", "colW", "ax0"], "colsigV")  # [15] fp32

    # per-column occupancy over the active grid: occ[c]=1 iff colsig[c] > 0
    init("zeroF", np.array(0.0, np.float32), np.float32)
    n("Greater", ["colsigV", "zeroF"], "occB")                     # [15] bool

    # ---- period consistency over all p in 1..MAXP --------------------------
    # modidx[p-1, c] = c mod p     -> gather colsig at those indices => [MAXP, W]
    modmat = np.array([[c % (p + 1) for c in range(W)] for p in range(MAXP)], np.int32)
    init("modmat", modmat, np.int32)
    n("Gather", ["colsigV", "modmat"], "sigAtMod", axis=0)         # [MAXP, W] fp32
    # broadcast colsigV [1,W] vs sigAtMod [MAXP,W]
    init("rowshape", np.array([1, W], np.int64), np.int64)
    n("Reshape", ["colsigV", "rowshape"], "colsigRow")             # [1, W]
    n("Equal", ["sigAtMod", "colsigRow"], "sameB")                 # [MAXP, W] bool
    n("Not", ["sameB"], "diffB")                                   # [MAXP, W] bool
    n("Reshape", ["occB", "rowshape"], "occRow")                   # [1, W] bool
    n("And", ["diffB", "occRow"], "badB")                          # [MAXP, W] bool
    n("Cast", ["badB"], "badF", to=TensorProto.FLOAT16)            # [MAXP, W] fp16
    n("ReduceSum", ["badF"], "nbad", axes=[1], keepdims=0)         # [MAXP] fp16
    # consistent[p-1] = (nbad == 0)
    init("zeroH", np.array(0.0, np.float16), np.float16)
    n("Equal", ["nbad", "zeroH"], "consB")                         # [MAXP] bool

    # ---- pick smallest consistent period EP --------------------------------
    # score[p-1] = consistent ? (MAXP - (p-1)) : 0  ; ArgMax of score gives the
    # smallest consistent p (largest weight at smallest index).
    n("Cast", ["consB"], "consF", to=F)                            # [MAXP] fp32
    pweight = np.array([MAXP - i for i in range(MAXP)], np.float32) # decreasing
    init("pweight", pweight, np.float32)
    n("Mul", ["consF", "pweight"], "score")                        # [MAXP] fp32
    n("ArgMax", ["score"], "epIdx", axis=0, keepdims=0)            # scalar int64 = EP-1

    # ---- srcIdx[c] = c mod EP  (gather a row of the period table) -----------
    # periodtab[p-1, c] = c mod p for c < W, else 29 (guaranteed-empty pad column
    # so off-grid output columns c >= W stay all-zero).  The off-grid redirect is
    # baked into the table, so the gathered row IS the final source index -- no
    # extra Where / mask plane.
    periodtab = np.array([[(c % (p + 1) if c < W else S - 1) for c in range(S)]
                          for p in range(MAXP)], np.int32)
    init("periodtab", periodtab, np.int32)
    n("Gather", ["periodtab", "epIdx"], "srcIdx", axis=0)          # [S] int32

    # ---- the whole output is a single column gather (FREE) ------------------
    n("Gather", ["input", "srcIdx"], "output", axis=3)            # [1,10,30,30] fp32

    x = helper.make_tensor_value_info("input", F, [1, 10, S, S])
    y = helper.make_tensor_value_info("output", F, [1, 10, S, S])
    graph = helper.make_graph(nodes, "task343", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
