"""task013 (ARC-AGI 0a938d79) — alternating periodic stripes from two seeds.

Rule (from the ARC-GEN generator, verified fresh):
  Grid is W x H (W in 20..30, H in 6..12).  Two seed pixels are placed:
      grid[bottoms[0]*(H-1)][start]            = colors[0]
      grid[bottoms[1]*(H-1)][start+sep+1]      = colors[1]
  with start in [1, W//2], sep in [1,5] (period p = sep+1 in [2,6]),
  bottoms[i] in {0,1} (top or bottom row).  The OUTPUT paints FULL vertical
  stripes (whole columns, all H rows) at columns start, start+p, start+2p, ...
  (< W), alternating colour colors[0], colors[1], colors[0], ...  If `xpose`,
  the whole grid is transposed (stripes become full rows).

Closed-form, separable, no big colour-index plane:
  * Orientation (verified 0/3000 mismatches):  xpose=1 IFF both seed COLUMNS
    are in {0, W-1}.  In xpose=0 the seed columns are start, start+p with
    start>=1 (never 0); in xpose=1 the seed columns equal bottoms*(W-1).
  * Per-orientation the period axis carries the two seeds at DISTINCT positions
    (period p>=2).  Use the matching-axis colour-weighted profile:
        colval[c] = sum_{r,k} k*input[k,r,c]   (xpose=0 stripes along cols)
        rowval[r] = sum_{c,k} k*input[k,r,c]   (xpose=1 stripes along rows)
    Each colored seed sits in its own column/row, so the profile value IS the
    seed colour index.
  * firstpos = min position with profile>0, lastpos = max, p = lastpos-firstpos.
    c0color = profile[firstpos], c1color = profile[lastpos].
  * stripe colour at position t (along period axis):
        m = (t - firstpos) mod (2p)   (t>=firstpos);  m==0 -> c0color,
        m==p -> c1color, else 0 (background).
  * Build a tiny [30] period-axis colour vector `pvec`, place it as colvec
    [1,1,1,30] (xpose=0) or rowvec [1,1,30,1] (xpose=1), select by xpose ->
    Lcolor[1,1,30,30] (broadcast of two perpendicular vectors in ONE Where).
    Gate to the in-grid rectangle (rowin & colin); off-grid -> -1 so
    Equal(.,0..9)=all-zero.  output = Equal(L, arange[1,10,1,1]) routes the
    10-ch expansion into the FREE bool output.

Memory: largest intermediates are a couple of fp16 [1,1,30,30] planes (1800B)
+ the in-grid bool mask; recovery uses only [1,10,1,30]/[30] tensors.
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
    F16 = TensorProto.FLOAT16
    NI = TensorProto.INT32
    B = TensorProto.BOOL

    init("flat", np.array([S], np.int64), np.int64)
    init("zeroF", np.array(0.0, np.float32), np.float32)
    init("halfF", np.array(0.5, np.float32), np.float32)
    init("arangeF", np.arange(S, dtype=np.float32), np.float32)  # [30]
    init("BIG", np.array(999.0, np.float32), np.float32)
    init("oneF", np.array(1.0, np.float32), np.float32)
    init("twoF", np.array(2.0, np.float32), np.float32)
    init("chW", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)

    # ---- per-axis colour-weighted profiles (small) -------------------------
    n("ReduceSum", ["input"], "perch_col", axes=[2], keepdims=1)   # [1,10,1,30]
    n("ReduceSum", ["input"], "perch_row", axes=[3], keepdims=1)   # [1,10,30,1]
    n("Conv", ["perch_col", "chW"], "colval4")   # [1,1,1,30]
    n("Conv", ["perch_row", "chW"], "rowval4")   # [1,1,30,1]
    n("Reshape", ["colval4", "flat"], "colval")  # [30]
    n("Reshape", ["rowval4", "flat"], "rowval")  # [30]

    # ---- occupancy / extents ----------------------------------------------
    n("ReduceMax", ["input"], "colocc4", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    n("ReduceMax", ["input"], "rowocc4", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("Reshape", ["colocc4", "flat"], "coloccV")  # [30]
    n("ReduceSum", ["coloccV"], "Wf", axes=[0], keepdims=1)        # [1] = W
    n("Sub", ["Wf", "oneF"], "Wm1")                               # [1] = W-1

    # ---- orientation: xpose=1 iff both seed columns are in {0, W-1} --------
    n("Greater", ["colval", "zeroF"], "colSeedB")                 # [30] bool
    n("Cast", ["colSeedB"], "colSeedF", to=F)                     # [30]
    n("Equal", ["arangeF", "zeroF"], "isC0")                      # [30] bool c==0
    n("Equal", ["arangeF", "Wm1"], "isCWm1")                      # [30] bool c==W-1
    n("Or", ["isC0", "isCWm1"], "isEdgeC")                        # [30] bool
    n("Cast", ["isEdgeC"], "isEdgeCf", to=F)
    n("ReduceSum", ["colSeedF"], "nSeedCol", axes=[0], keepdims=1)         # [1]
    n("Mul", ["colSeedF", "isEdgeCf"], "seedEdge")                # [30]
    n("ReduceSum", ["seedEdge"], "nSeedEdge", axes=[0], keepdims=1)        # [1]
    n("Equal", ["nSeedEdge", "nSeedCol"], "xposeB")              # [1] bool

    # ---- select the period-axis profile ------------------------------------
    n("Where", ["xposeB", "rowval", "colval"], "pval")          # [30]

    # firstpos / lastpos / p
    n("Greater", ["pval", "zeroF"], "pSeedB")                    # [30] bool
    n("Cast", ["pSeedB"], "pSeedF", to=F)
    n("Mul", ["arangeF", "pSeedF"], "posMasked")                # [30]
    n("ReduceMax", ["posMasked"], "lastF", axes=[0], keepdims=1)  # [1]
    n("Where", ["pSeedB", "arangeF", "BIG"], "posForMin")       # [30]
    n("ReduceMin", ["posForMin"], "firstF", axes=[0], keepdims=1)  # [1]
    n("Sub", ["lastF", "firstF"], "pF")                         # [1] = p

    # seed colours
    n("Cast", ["firstF"], "firstI", to=NI)                      # [1]
    n("Cast", ["lastF"], "lastI", to=NI)                        # [1]
    n("Gather", ["pval", "firstI"], "c0col1")                   # [1]
    n("Gather", ["pval", "lastI"], "c1col1")                    # [1]

    # ---- build pvec[30] ----------------------------------------------------
    n("Sub", ["arangeF", "firstF"], "relF")                     # [30]
    n("Mul", ["pF", "twoF"], "twopF")                           # [1] = 2p
    n("Max", ["relF", "zeroF"], "relPos")                       # [30] >=0
    n("Cast", ["relPos"], "relPos16", to=F16)
    n("Cast", ["twopF"], "twop16", to=F16)
    n("Mod", ["relPos16", "twop16"], "m16", fmod=1)             # [30] fp16
    n("Cast", ["m16"], "mF", to=F)                              # [30]
    n("Equal", ["mF", "zeroF"], "m_eq0")                        # [30] bool
    n("Equal", ["mF", "pF"], "m_eqp")                           # [30] bool
    n("Less", ["relF", "zeroF"], "relNeg")                      # [30] bool
    n("Not", ["relNeg"], "relNN")                               # [30] bool rel>=0
    n("And", ["m_eq0", "relNN"], "useC0")                       # [30] bool
    n("And", ["m_eqp", "relNN"], "useC1")                       # [30] bool
    n("Where", ["useC1", "c1col1", "zeroF"], "pvec_a")          # [30]
    n("Where", ["useC0", "c0col1", "pvec_a"], "pvec")           # [30]

    # ---- place along period axis, select orientation, broadcast ------------
    init("shCol", np.array([1, 1, 1, S], np.int64), np.int64)
    init("shRow", np.array([1, 1, S, 1], np.int64), np.int64)
    n("Reshape", ["pvec", "shCol"], "colvecP")                  # [1,1,1,30]
    n("Reshape", ["pvec", "shRow"], "rowvecP")                  # [1,1,30,1]
    # cast the tiny perpendicular vectors to fp16 BEFORE the broadcast so the
    # [1,1,30,30] Lcolor plane is produced directly in fp16 (1800B not 3600B).
    n("Cast", ["colvecP"], "colvecP16", to=F16)                 # [1,1,1,30]
    n("Cast", ["rowvecP"], "rowvecP16", to=F16)                 # [1,1,30,1]
    n("Where", ["xposeB", "rowvecP16", "colvecP16"], "Lcolor16")  # [1,1,30,30] fp16

    # ---- in-grid mask + final Equal ----------------------------------------
    n("Greater", ["colocc4", "halfF"], "colinB")                # [1,1,1,30]
    n("Greater", ["rowocc4", "halfF"], "rowinB")                # [1,1,30,1]
    n("And", ["colinB", "rowinB"], "ingridB")                   # [1,1,30,30]
    init("neg1_16", np.array(-1.0, np.float16), np.float16)
    n("Where", ["ingridB", "Lcolor16", "neg1_16"], "L16")       # [1,1,30,30] fp16
    init("arangeCh16", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)
    n("Equal", ["L16", "arangeCh16"], "output")                 # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task013", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
