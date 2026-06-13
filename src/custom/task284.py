"""Task 284 (ARC b7249182): two seed dots, each color grows a bilateral
"wrench"/cross glyph toward the centerline; whole grid optionally transposed.

Rule (canonical, seeds share a row R at columns Cl<Cr, half=(Cr-Cl+1)/2):
  left glyph (color of left seed): horizontal stem row R cols Cl..e0, a 5-tall
  vertical bar at col e0 (rows R-2..R+2), hook cells at (R-2,e0+1),(R+2,e0+1)
  where e0=(Cl+Cr-3)/2; right glyph mirrors with e1=(Cl+Cr+3)/2.
The two crosses are always 3 cols apart, straddling the seed-pair center.

Net: build all 1-D along/cross masks (tiny vectors), form left/right abstract
[1,1,30,30] planes via outer products, transpose+select for orientation, then
one runtime-weighted 1x1 Conv routes the two planes into the two seed-color
channels straight into `output`. The transpose is detected in-graph (seeds
sharing a column => transposed) so a single graph handles both orientations.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        if isinstance(inputs, str):
            inputs = [inputs]
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    init("c_one", np.array(1.0, np.float32))
    init("c_two", np.array(2.0, np.float32))
    init("c_three", np.array(3.0, np.float32))
    init("BIG", np.array(100.0, np.float32))
    idx = np.arange(30, dtype=np.float32)
    init("idxW", idx.reshape(1, 1, 1, 30))
    init("idxH", idx.reshape(1, 1, 30, 1))

    # --- per-channel occupancy (reused for both masks and color routing) ---
    n("ReduceMax", "input", "rowOccF", axes=[3], keepdims=1)     # [1,10,30,1]
    n("ReduceMax", "input", "colOccF", axes=[2], keepdims=1)     # [1,10,1,30]
    no0 = np.ones((1, 10, 1, 1), np.float32); no0[0, 0, 0, 0] = 0.0
    init("no0", no0)
    # seed occupancy excluding channel 0, reduced over channels -> 1-D masks
    n("Mul", ["rowOccF", "no0"], "rowOccS")                      # [1,10,30,1]
    n("Mul", ["colOccF", "no0"], "colOccS")                      # [1,10,1,30]
    n("ReduceMax", "rowOccS", "rowMaskH", axes=[1], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", "colOccS", "colMaskW", axes=[1], keepdims=1)  # [1,1,1,30]

    # isT = seeds share a column  <=>  >1 occupied row
    n("ReduceSum", "rowMaskH", "rowCnt", axes=[2], keepdims=1)   # [1,1,1,1]
    n("Greater", ["rowCnt", "c_one"], "isT_b")
    n("Cast", "isT_b", "isT", to=onnx.TensorProto.FLOAT)         # [1,1,1,1]
    n("Sub", ["c_one", "isT"], "notT")

    # reshape masks across axes for orientation-agnostic along/cross vectors
    init("shpW", np.array([1, 1, 1, 30], np.int64), np.int64)
    init("shpH", np.array([1, 1, 30, 1], np.int64), np.int64)
    n("Reshape", ["rowMaskH", "shpW"], "rowMaskW")               # [1,1,1,30]
    n("Reshape", ["colMaskW", "shpH"], "colMaskH")               # [1,1,30,1]

    # alongVecW [1,1,1,30] : ones at the two seed positions along long axis
    n("Mul", ["isT", "rowMaskW"], "aw_t")
    n("Mul", ["notT", "colMaskW"], "aw_n")
    n("Add", ["aw_t", "aw_n"], "alongVecW")
    # crossVecH [1,1,30,1] : one at the shared coordinate K
    n("Mul", ["isT", "colMaskH"], "ch_t")
    n("Mul", ["notT", "rowMaskH"], "ch_n")
    n("Add", ["ch_t", "ch_n"], "crossVecH")

    # --- along scalars ---
    n("Mul", ["idxW", "alongVecW"], "ia")                        # [1,1,1,30]
    n("ReduceSum", "ia", "S", axes=[3], keepdims=1)              # Cl+Cr
    n("ReduceMax", "ia", "Cr", axes=[3], keepdims=1)
    n("Sub", ["c_one", "alongVecW"], "inv_a")
    n("Mul", ["inv_a", "BIG"], "inv_big")
    n("Add", ["ia", "inv_big"], "ia_min")
    n("ReduceMin", "ia_min", "Cl", axes=[3], keepdims=1)
    # e0 = (S-3)/2 , e1 = (S+3)/2
    n("Sub", ["S", "c_three"], "Sm3")
    n("Div", ["Sm3", "c_two"], "e0")
    n("Add", ["S", "c_three"], "Sp3")
    n("Div", ["Sp3", "c_two"], "e1")

    # cross coordinate K (shared seed coordinate along the short axis)
    n("Mul", ["idxH", "crossVecH"], "ic")
    n("ReduceSum", "ic", "K", axes=[2], keepdims=1)             # [1,1,1,1]

    def col(name, terms):
        # build a [1,1,N,1] threshold column from (scalar, offset) terms
        parts = []
        for i, (base, off) in enumerate(terms):
            init(f"o_{name}{i}", np.array(off, np.float32))
            parts.append(n("Add", [base, f"o_{name}{i}"], f"t_{name}{i}"))
        return n("Concat", parts, f"col_{name}", axis=2)        # [1,1,N,1]

    def bands(name, lo, hi):
        # batched 0/1 masks: (idxW > lo) & (idxW < hi) -> [1,1,N,30]
        g = n("Greater", ["idxW", lo], f"g_{name}")
        l = n("Less", ["idxW", hi], f"l_{name}")
        gf = n("Cast", g, f"gf_{name}", to=onnx.TensorProto.FLOAT)
        lf = n("Cast", l, f"lf_{name}", to=onnx.TensorProto.FLOAT)
        return n("Mul", [gf, lf], f"b_{name}")                  # [1,1,N,30]

    # all 6 along masks at once, rows [stemL,crossLc,hookLc, stemR,crossRc,hookRc]
    alo = col("alo", [("Cl", -0.5), ("e0", -0.5), ("e0", 0.5),
                      ("e1", -0.5), ("e1", -0.5), ("e1", -1.5)])
    ahi = col("ahi", [("e0", 0.5), ("e0", 0.5), ("e0", 1.5),
                      ("Cr", 0.5), ("e1", 0.5), ("e1", -0.5)])
    alongAll = bands("along", alo, ahi)                         # [1,1,6,30]

    # 3 cross masks: rows [crossK, band5, mid3] -> ends2 = band5 - mid3
    clo = col("clo", [("K", -0.5), ("K", -2.5), ("K", -1.5)])
    chi = col("chi", [("K", 0.5), ("K", 2.5), ("K", 1.5)])
    crossB = bands("cross", clo, chi)                           # [1,1,3,30]
    init("s_b5", np.array([1], np.int64), np.int64)
    init("s_b5e", np.array([2], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    n("Slice", [crossB, "s_b5", "s_b5e", "ax2"], "band5row")    # [1,1,1,30]
    init("s_m3", np.array([2], np.int64), np.int64)
    init("s_m3e", np.array([3], np.int64), np.int64)
    n("Slice", [crossB, "s_m3", "s_m3e", "ax2"], "mid3row")
    n("Sub", ["band5row", "mid3row"], "ends2row")               # [1,1,1,30]
    init("s_cK", np.array([0], np.int64), np.int64)
    init("s_cKe", np.array([1], np.int64), np.int64)
    n("Slice", [crossB, "s_cK", "s_cKe", "ax2"], "crossKrow")
    n("Concat", ["crossKrow", "band5row", "ends2row"], "crossT", axis=2)  # [1,1,3,30]

    # split along masks into the two color triples
    init("s_L", np.array([0], np.int64), np.int64)
    init("s_Le", np.array([3], np.int64), np.int64)
    n("Slice", [alongAll, "s_L", "s_Le", "ax2"], "alongL")      # [1,1,3,30]
    init("s_R", np.array([3], np.int64), np.int64)
    init("s_Re", np.array([6], np.int64), np.int64)
    n("Slice", [alongAll, "s_R", "s_Re", "ax2"], "alongR")      # [1,1,3,30]

    # --- orientation: choose which triple lands on H vs W (factor-level) ---
    # MatMul factors: CH [1,2,30,3] (H positions x 3 comps), AW [1,2,3,30].
    # N-orientation: cross on H, along on W.  T: along on H, cross on W.
    # Both color channels processed in one batched pass.
    n("Concat", ["alongL", "alongR"], "alongLR", axis=1)        # [1,2,3,30]
    n("Transpose", ["alongLR"], "alongH", perm=[0, 1, 3, 2])    # [1,2,30,3]
    n("Transpose", ["crossT"], "crossH", perm=[0, 1, 3, 2])     # [1,1,30,3] (bcast)
    n("Mul", ["isT", "alongH"], "chT")
    n("Mul", ["notT", "crossH"], "chN")
    n("Add", ["chT", "chN"], "CH2")                            # [1,2,30,3]
    n("Mul", ["isT", "crossT"], "awT")
    n("Mul", ["notT", "alongLR"], "awN")
    n("Add", ["awT", "awN"], "AW2")                            # [1,2,3,30]

    # in-grid rectangle as a rank-1 factor (rowExtent (x) colExtent), padded to
    # 3 components with zeros so it stacks with the rank-3 glyph factors.
    n("ReduceMax", "rowOccF", "rowExt", axes=[1], keepdims=1)    # [1,1,30,1]
    n("ReduceMax", "colOccF", "colExt", axes=[1], keepdims=1)    # [1,1,1,30]
    init("zH", np.zeros((1, 1, 30, 2), np.float32))
    init("zW", np.zeros((1, 1, 2, 30), np.float32))
    n("Concat", ["rowExt", "zH"], "CH_G", axis=3)               # [1,1,30,3]
    n("Concat", ["colExt", "zW"], "AW_G", axis=2)               # [1,1,3,30]

    # Stack the 3 channels' factors and form all planes in ONE MatMul ->
    # shapes [1,3,30,30] directly (no separate canvas-sized plane tensors).
    n("Concat", ["CH2", "CH_G"], "CH3", axis=1)                # [1,3,30,3]
    n("Concat", ["AW2", "AW_G"], "AW3", axis=1)                # [1,3,3,30]
    n("MatMul", ["CH3", "AW3"], "shapes")                      # [1,3,30,30]

    # --- color routing weight [10,3,1,1] from seed colors ---
    # per-channel along-position via tiny [1,10,1,1] reductions (no canvas temp)
    n("Mul", ["idxH", "rowOccS"], "ioH")                        # [1,10,30,1]
    n("ReduceSum", "ioH", "rowPos", axes=[2], keepdims=1)       # [1,10,1,1]
    n("Mul", ["idxW", "colOccS"], "ioW")                        # [1,10,1,30]
    n("ReduceSum", "ioW", "colPos", axes=[3], keepdims=1)       # [1,10,1,1]
    n("Mul", ["isT", "rowPos"], "cp_t")
    n("Mul", ["notT", "colPos"], "cp_n")
    n("Add", ["cp_t", "cp_n"], "chPos")                        # [1,10,1,1]
    n("ReduceMax", "rowOccS", "present", axes=[2], keepdims=1)  # [1,10,1,1]
    n("Mul", ["chPos", "c_two"], "chPos2")
    n("Less", ["chPos2", "S"], "isL_b")
    n("Greater", ["chPos2", "S"], "isR_b")
    n("Cast", "isL_b", "isL", to=onnx.TensorProto.FLOAT)
    n("Cast", "isR_b", "isR", to=onnx.TensorProto.FLOAT)
    n("Mul", ["present", "isL"], "leftSel")                    # [1,10,1,1]
    n("Mul", ["present", "isR"], "rightSel")
    init("shp10111", np.array([10, 1, 1, 1], np.int64), np.int64)
    n("Reshape", ["leftSel", "shp10111"], "wL")                # [10,1,1,1]
    n("Reshape", ["rightSel", "shp10111"], "wR")
    init("wzero", np.zeros((10, 1, 1, 1), np.float32))
    n("Concat", ["wL", "wR", "wzero"], "Wdyn", axis=1)         # [10,3,1,1]
    # baseW: channel 0 = ingrid - left - right (background); colors unaffected
    baseW = np.zeros((10, 3, 1, 1), np.float32)
    baseW[0, 0, 0, 0] = -1.0; baseW[0, 1, 0, 0] = -1.0; baseW[0, 2, 0, 0] = 1.0
    init("baseW", baseW)
    n("Add", ["Wdyn", "baseW"], "Wrt")                         # [10,3,1,1]

    # final 1x1 Conv: routes left/right planes + computes ch0 background
    n("Conv", ["shapes", "Wrt"], "output", kernel_shape=[1, 1])
    return _model(nodes, inits)
