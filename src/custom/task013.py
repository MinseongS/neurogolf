"""task013 (ARC-AGI 0a938d79) — alternating periodic stripes from two seeds.

Rule (from the ARC-GEN generator, verified fresh):
  Grid is W x H (W in 20..30, H in 6..12).  Two seed pixels are placed:
      grid[bottoms[0]*(H-1)][start]            = colors[0]
      grid[bottoms[1]*(H-1)][start+sep+1]      = colors[1]
  with start in [1, W//2], sep in [1,5] (period p = sep+1 in [2,6]),
  bottoms[i] in {0,1} (top or bottom row).  The OUTPUT paints FULL vertical
  stripes (whole columns, all H rows) at columns start, start+p, start+2p, ...
  (< W), alternating colour colors[0], colors[1], colors[0], ...  If `xpose`,
  the whole grid is transposed (stripes become full rows).  Colours are in 1..9.

Closed-form, separable, no big colour-index plane:
  * Orientation (verified 0 mismatches):  xpose=1 IFF both seed COLUMNS are in
    {0, W-1}.  In xpose=0 the seed columns are start, start+p with start>=1
    (never 0); in xpose=1 the seed columns equal bottoms*(W-1).
  * Per-axis colour-weighted profile in ONE no-pad Conv each (collapses the
    spatial axis AND the channel axis simultaneously, dodging the 1200B
    [1,10,30,1] ReduceSum plane):
        colval[c] = Conv(input, W[1,10,30,1] with W[k]=k) -> [1,1,1,30]
        rowval[r] = Conv(input, W[1,10,1,30] with W[k]=k) -> [1,1,30,1]
    Each colored seed sits in its own column/row, so the profile value IS the
    seed colour index.
  * firstpos = min position with profile>0, lastpos = max, p = lastpos-firstpos.
    c0color = profile[firstpos], c1color = profile[lastpos].
  * stripe colour at period-axis position t:  m = (t - firstpos) mod (2p)
    (for t>=firstpos);  m==0 -> c0color, m==p -> c1color, else 0 (background).
  * pvec[30] = the period-axis colour vector, GATED to the in-grid extent
    (off-grid period positions -> -1).  Two nested broadcast Wheres place pvec on
    the correct axis and gate the cross axis, producing the single full-grid
    index plane L directly:
        xpose=0:  L = Where(rowin[1,1,30,1], pvecCol[1,1,1,30], -1)
        xpose=1:  L = Where(colin[1,1,1,30], pvecRow[1,1,30,1], -1)
    output = Equal(L, arange_ch[1,10,1,1]) routes the 10-ch expansion into the
    FREE bool output.

Memory: dominated by the full-grid index planes (fp16 [1,1,30,30]=1800B each);
all recovery uses [1,1,1,30]/[1,1,30,1]/[30] tensors (<=120B).
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
    # fp16 constants for the (exact, integer) position/colour recovery vectors.
    init("zero16", np.array(0.0, np.float16), np.float16)
    init("half16", np.array(0.5, np.float16), np.float16)
    init("arange16", np.arange(S, dtype=np.float16), np.float16)  # [30] fp16
    init("BIG16", np.array(999.0, np.float16), np.float16)
    init("one16", np.array(1.0, np.float16), np.float16)
    init("two16", np.array(2.0, np.float16), np.float16)

    # colour-weighted profile kernels (k weight per channel), collapse one axis.
    Wcol = np.zeros((1, 10, S, 1), np.float32)
    for k in range(10):
        Wcol[0, k, :, 0] = k
    init("Wcol", Wcol, np.float32)               # collapse rows -> per-col [1,1,1,30]
    Wrow = np.zeros((1, 10, 1, S), np.float32)
    for k in range(10):
        Wrow[0, k, 0, :] = k
    init("Wrow", Wrow, np.float32)               # collapse cols -> per-row [1,1,30,1]

    # ---- per-axis colour-weighted profiles in one Conv each ----------------
    n("Conv", ["input", "Wcol"], "colval4")      # [1,1,1,30] fp32
    n("Conv", ["input", "Wrow"], "rowval4")      # [1,1,30,1] fp32
    n("Cast", ["colval4", ], "colval4h", to=F16)  # [1,1,1,30] fp16
    n("Cast", ["rowval4", ], "rowval4h", to=F16)  # [1,1,30,1] fp16
    n("Reshape", ["colval4h", "flat"], "colval")  # [30] fp16
    n("Reshape", ["rowval4h", "flat"], "rowval")  # [30] fp16

    # ---- occupancy / extents (per-col, per-row) ----------------------------
    n("ReduceMax", ["input"], "colocc4", axes=[1, 2], keepdims=1)  # [1,1,1,30] fp32
    n("ReduceMax", ["input"], "rowocc4", axes=[1, 3], keepdims=1)  # [1,1,30,1] fp32
    n("Cast", ["colocc4"], "colocc4h", to=F16)    # [1,1,1,30] fp16
    n("Cast", ["rowocc4"], "rowocc4h", to=F16)    # [1,1,30,1] fp16
    n("Reshape", ["colocc4h", "flat"], "coloccV")  # [30] fp16
    n("Reshape", ["rowocc4h", "flat"], "rowoccV")  # [30] fp16
    n("ReduceSum", ["coloccV"], "Wf16", axes=[0], keepdims=1)     # [1] = W (fp16)
    n("Sub", ["Wf16", "one16"], "Wm1")                           # [1] = W-1

    # ---- orientation: xpose=1 iff both seed columns are in {0, W-1} --------
    n("Greater", ["colval", "zero16"], "colSeedB")               # [30] bool
    n("Cast", ["colSeedB"], "colSeedF", to=F16)                  # [30]
    n("Equal", ["arange16", "zero16"], "isC0")                   # [30] bool c==0
    n("Equal", ["arange16", "Wm1"], "isCWm1")                    # [30] bool c==W-1
    n("Or", ["isC0", "isCWm1"], "isEdgeC")                       # [30] bool
    n("Cast", ["isEdgeC"], "isEdgeCf", to=F16)
    n("ReduceSum", ["colSeedF"], "nSeedCol", axes=[0], keepdims=1)        # [1] fp16
    n("Mul", ["colSeedF", "isEdgeCf"], "seedEdge")              # [30] fp16
    n("ReduceSum", ["seedEdge"], "nSeedEdge", axes=[0], keepdims=1)       # [1] fp16
    n("Equal", ["nSeedEdge", "nSeedCol"], "xposeB")            # [1] bool

    # ---- select the period-axis profile ------------------------------------
    n("Where", ["xposeB", "rowval", "colval"], "pval")         # [30] fp16

    # firstpos / lastpos / p
    n("Greater", ["pval", "zero16"], "pSeedB")                 # [30] bool
    n("Cast", ["pSeedB"], "pSeedF", to=F16)
    n("Mul", ["arange16", "pSeedF"], "posMasked")              # [30] fp16
    n("ReduceMax", ["posMasked"], "lastF", axes=[0], keepdims=1)  # [1] fp16
    n("Where", ["pSeedB", "arange16", "BIG16"], "posForMin")   # [30] fp16
    n("ReduceMin", ["posForMin"], "firstF", axes=[0], keepdims=1)  # [1] fp16
    n("Sub", ["lastF", "firstF"], "pF")                        # [1] = p (fp16)

    # seed colours (Gather indices need int32)
    n("Cast", ["firstF"], "firstI", to=NI)                     # [1]
    n("Cast", ["lastF"], "lastI", to=NI)                       # [1]
    n("Gather", ["pval", "firstI"], "c0col1")                  # [1] fp16
    n("Gather", ["pval", "lastI"], "c1col1")                   # [1] fp16

    # ---- build pvec[30] (colour vector along period axis) ------------------
    n("Sub", ["arange16", "firstF"], "relF")                   # [30] fp16
    n("Mul", ["pF", "two16"], "twopF")                         # [1] = 2p fp16
    n("Max", ["relF", "zero16"], "relPos")                     # [30] >=0 fp16
    n("Mod", ["relPos", "twopF"], "m16", fmod=1)               # [30] fp16
    n("Equal", ["m16", "zero16"], "m_eq0")                     # [30] bool
    n("Equal", ["m16", "pF"], "m_eqp")                         # [30] bool
    n("Less", ["relF", "zero16"], "relNeg")                    # [30] bool
    n("Not", ["relNeg"], "relNN")                              # [30] bool rel>=0
    n("And", ["m_eq0", "relNN"], "useC0")                      # [30] bool
    n("And", ["m_eqp", "relNN"], "useC1")                      # [30] bool
    n("Where", ["useC1", "c1col1", "zero16"], "pvec_a")        # [30] fp16
    n("Where", ["useC0", "c0col1", "pvec_a"], "pvec")          # [30] fp16

    # ---- gate pvec to the IN-GRID period extent (off-grid period -> 200 sent)
    # period-axis occupancy: for xpose=0 period axis = cols -> colocc; for
    # xpose=1 period axis = rows -> rowocc.  Pick by xpose with the [30] vecs.
    n("Where", ["xposeB", "rowoccV", "coloccV"], "periodOccV")  # [30] fp16
    n("Greater", ["periodOccV", "half16"], "periodInB")        # [30] bool in-grid
    # Colour vector along the period axis, off-grid (period) -> 200 sentinel.
    init("big16", np.array(200.0, np.float16), np.float16)
    n("Where", ["periodInB", "pvec", "big16"], "periodVal16")  # [30] fp16 colour or 200

    # cross-axis in-grid gate: in-grid -> 0, off-grid -> 200 (sentinel).
    # cross occupancy per orientation: cross = rows(xpose0)/cols(xpose1)
    n("Where", ["xposeB", "coloccV", "rowoccV"], "crossOccV")  # [30] fp16
    n("Greater", ["crossOccV", "half16"], "crossInB")          # [30] bool
    n("Where", ["crossInB", "zero16", "big16"], "crossGate16")  # [30] fp16 0 or 200

    # ---- place colour on PERIOD axis, gate on CROSS axis; one Max plane -----
    # FIXED axes: col on axis-3, row on axis-2.  Select each axis' CONTENT by
    # xpose so the colour rides the period axis and the gate the cross axis.
    #   xpose=0: period=cols -> col carries colour, row carries gate
    #   xpose=1: period=rows -> row carries colour, col carries gate
    # Build colVec/rowVec directly in fp16 (Where accepts fp16) so the two
    # placed vectors are 60B and the Max plane is born fp16 with no Cast nodes.
    n("Where", ["xposeB", "crossGate16", "periodVal16"], "colVec")  # [30] fp16 (col axis)
    n("Where", ["xposeB", "periodVal16", "crossGate16"], "rowVec")  # [30] fp16 (row axis)
    init("shCol", np.array([1, 1, 1, S], np.int64), np.int64)
    init("shRow", np.array([1, 1, S, 1], np.int64), np.int64)
    n("Reshape", ["colVec", "shCol"], "colVec16")              # [1,1,1,30] fp16
    n("Reshape", ["rowVec", "shRow"], "rowVec16")              # [1,1,30,1] fp16
    # off-grid (either axis) -> 200; in-grid -> max(colour,0)=colour.
    n("Max", ["colVec16", "rowVec16"], "L16")                  # [1,1,30,30] fp16

    init("arangeCh16", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)
    n("Equal", ["L16", "arangeCh16"], "output")               # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task013", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
