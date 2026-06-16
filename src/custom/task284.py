"""Task 284 (ARC b7249182): two seed dots, each color grows a bilateral
"wrench"/cross glyph toward the centerline; whole grid optionally transposed.

Optimized re-encode: ONE combined 6-component MatMul builds the entire glyph
(both colors) as a single [1,1,30,30] plane; cells are coloured by an along-axis
side threshold (left color if along-coord < S/2 else right color), so no second
glyph plane is needed. Off-grid -> label 10 -> all-zero one-hot.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


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

    F16 = onnx.TensorProto.FLOAT16
    init("c_one", np.array(1.0, np.float16), np.float16)
    init("c_two", np.array(2.0, np.float16), np.float16)
    init("c_three", np.array(3.0, np.float16), np.float16)
    init("half16", np.array(0.5, np.float16), np.float16)
    init("BIG", np.array(100.0, np.float16), np.float16)
    idx = np.arange(30, dtype=np.float16)
    init("idxW", idx.reshape(1, 1, 1, 30), np.float16)
    init("idxH", idx.reshape(1, 1, 30, 1), np.float16)

    # --- single colour-index plane of the input (only 2 seeds are nonzero) ---
    # colf = sum_k k*input_k via a 1x1 Conv.  fp32 [1,1,30,30] (the one heavy
    # plane); everything downstream reduces it to tiny 1-D profiles.
    cw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("convW", cw, np.float32)
    n("Conv", ["input", "convW"], "colf")                        # [1,1,30,30] f32
    # colour profiles along each axis (sum of seed colours)
    n("ReduceSum", "colf", "rowProfH", axes=[3], keepdims=1)     # [1,1,30,1]
    n("ReduceSum", "colf", "colProfW", axes=[2], keepdims=1)     # [1,1,1,30]
    # per-axis seed-presence (binary)
    init("zeroF", np.array(0.0, np.float32), np.float32)
    n("Greater", ["rowProfH", "zeroF"], "rowHas_b")              # [1,1,30,1] bool
    n("Greater", ["colProfW", "zeroF"], "colHas_b")              # [1,1,1,30] bool
    n("Cast", "rowHas_b", "rowHasF", to=F16)
    n("Cast", "colHas_b", "colHasF", to=F16)

    # isT = seeds share a column  <=>  >1 occupied seed row
    n("ReduceSum", "rowHasF", "rowCnt", axes=[2], keepdims=1)    # [1,1,1,1]
    n("Greater", ["rowCnt", "c_one"], "isT_b")
    n("Cast", "isT_b", "isT", to=F16)
    n("Sub", ["c_one", "isT"], "notT")

    # reshape H-profiles to a W vector so along/cross are orientation-agnostic
    init("shpW", np.array([1, 1, 1, 30], np.int64), np.int64)
    init("shpH", np.array([1, 1, 30, 1], np.int64), np.int64)
    n("Reshape", ["rowHasF", "shpW"], "rowHasW")                 # [1,1,1,30]
    n("Reshape", ["colHasF", "shpH"], "colHasH")                 # [1,1,30,1]
    # colour profiles, fp16, reshaped W
    n("Cast", "rowProfH", "rowProfHf", to=F16)
    n("Cast", "colProfW", "colProfWf", to=F16)
    n("Reshape", ["rowProfHf", "shpW"], "rowProfW")              # [1,1,1,30]

    # alongVecW [1,1,1,30] : seed presence along the long axis (as a W vector)
    n("Mul", ["isT", "rowHasW"], "aw_t")
    n("Mul", ["notT", "colHasF"], "aw_n")
    n("Add", ["aw_t", "aw_n"], "alongVecW")
    # crossVecH [1,1,30,1] : shared coordinate K presence (as an H vector)
    n("Mul", ["isT", "colHasH"], "ch_t")
    n("Mul", ["notT", "rowHasF"], "ch_n")
    n("Add", ["ch_t", "ch_n"], "crossVecH")
    # along colour profile (as W vector): colour at each long-axis seed position
    n("Mul", ["isT", "rowProfW"], "ac_t")
    n("Mul", ["notT", "colProfWf"], "ac_n")
    n("Add", ["ac_t", "ac_n"], "alongColW")                     # [1,1,1,30] f16

    # --- along scalars: S=Cl+Cr, Cl, Cr ---
    n("Mul", ["idxW", "alongVecW"], "ia")                        # [1,1,1,30]
    n("ReduceSum", "ia", "S", axes=[3], keepdims=1)              # Cl+Cr
    n("ReduceMax", "ia", "Cr", axes=[3], keepdims=1)
    n("Sub", ["c_one", "alongVecW"], "inv_a")
    n("Mul", ["inv_a", "BIG"], "inv_big")
    n("Add", ["ia", "inv_big"], "ia_min")
    n("ReduceMin", "ia_min", "Cl", axes=[3], keepdims=1)
    n("Sub", ["S", "c_three"], "Sm3")
    n("Div", ["Sm3", "c_two"], "e0")
    n("Add", ["S", "c_three"], "Sp3")
    n("Div", ["Sp3", "c_two"], "e1")
    n("Div", ["S", "c_two"], "mid")                              # side threshold

    # cross coordinate K
    n("Mul", ["idxH", "crossVecH"], "ic")
    n("ReduceSum", "ic", "K", axes=[2], keepdims=1)              # [1,1,1,1]

    # ---- build 6 glyph components ----
    # along factor AW [1,1,6,30] : membership along the long axis
    #   rows: 0 stemL, 1 barL, 2 hookL, 3 stemR, 4 barR, 5 hookR
    def col(name, terms):
        parts = []
        for i, (base, off) in enumerate(terms):
            init(f"o_{name}{i}", np.array(off, np.float16), np.float16)
            parts.append(n("Add", [base, f"o_{name}{i}"], f"t_{name}{i}"))
        return n("Concat", parts, f"col_{name}", axis=2)        # [1,1,N,1]

    def bands(name, lo, hi, axisvec, outname):
        g = n("Greater", [axisvec, lo], f"g_{name}")
        l = n("Less", [axisvec, hi], f"l_{name}")
        gf = n("Cast", g, f"gf_{name}", to=F16)
        lf = n("Cast", l, f"lf_{name}", to=F16)
        return n("Mul", [gf, lf], outname)

    # along bands (vs idxW)
    alo = col("alo", [("Cl", -0.5), ("e0", -0.5), ("e0", 0.5),
                      ("e1", -0.5), ("e1", -0.5), ("e1", -1.5)])
    ahi = col("ahi", [("e0", 0.5), ("e0", 0.5), ("e0", 1.5),
                      ("Cr", 0.5), ("e1", 0.5), ("e1", -0.5)])
    bands("along", alo, ahi, "idxW", "AWa")                      # [1,1,6,30]

    # cross factor (vs idxW positions): only 3 distinct rows are needed --
    #   row0 stemX: x==K ; row1 band5: K-2..K+2 ; row2 mid3: K-1..K+1.
    # hook membership ends2 = band5 - mid3.  Glyph rows reuse these.
    clo = col("clo", [("K", -0.5), ("K", -2.5), ("K", -1.5)])
    chi = col("chi", [("K", 0.5), ("K", 2.5), ("K", 1.5)])
    bands("cross", clo, chi, "idxW", "CHc")                      # [1,1,3,30]
    init("ax2", np.array([2], np.int64), np.int64)
    init("s0", np.array([0], np.int64), np.int64)
    init("s1", np.array([1], np.int64), np.int64)
    init("s2", np.array([2], np.int64), np.int64)
    init("s3", np.array([3], np.int64), np.int64)
    n("Slice", ["CHc", "s0", "s1", "ax2"], "stemX")            # x==K
    n("Slice", ["CHc", "s1", "s2", "ax2"], "band5")            # K-2..K+2
    n("Slice", ["CHc", "s2", "s3", "ax2"], "mid3a")            # K-1..K+1
    n("Sub", ["band5", "mid3a"], "ends2")                       # x in {K-2,K+2}
    # assemble CH (cross factor) rows: stem,bar,hook,stem,bar,hook
    n("Concat", ["stemX", "band5", "ends2", "stemX", "band5", "ends2"],
      "CHrows", axis=2)                                          # [1,1,6,30]

    # ---- CANONICAL frame: H=cross, W=along.  Build the whole label here, then
    #      transpose the finished label iff the instance is transposed. ----
    # CH [1,1,30,6] cross-membership over H positions; AW [1,1,6,30] along over W.
    n("Transpose", ["CHrows"], "CH", perm=[0, 1, 3, 2])         # [1,1,30,6]
    n("MatMul", ["CH", "AWa"], "glyphP")                        # [1,1,30,30] f32
    n("Greater", ["glyphP", "half16"], "glyphM")               # bool (sum>0.5 = OR)

    # in-grid extent vectors off the free input (120B each, no plane)
    n("ReduceMax", "input", "rowExtF", axes=[1, 3], keepdims=1)  # [1,1,30,1] f32
    n("ReduceMax", "input", "colExtF", axes=[1, 2], keepdims=1)  # [1,1,1,30] f32
    n("Greater", ["rowExtF", "zeroF"], "rowExtB")               # real row extent
    n("Greater", ["colExtF", "zeroF"], "colExtB")               # real col extent
    # canonical extents: crossExt on H, alongExt on W.
    #   non-T: cross=real rows, along=real cols.  T: swapped.
    n("Not", "isT_b", "notT_b")
    n("Reshape", ["rowExtB", "shpW"], "rowExtW")                # [1,1,1,30]
    n("Reshape", ["colExtB", "shpH"], "colExtH")                # [1,1,30,1]
    # crossExtH = notT?rowExt:colExt  (bool, via And/Or)
    n("And", ["notT_b", "rowExtB"], "ceh_n")
    n("And", ["isT_b", "colExtH"], "ceh_t")
    n("Or", ["ceh_n", "ceh_t"], "crossExtH")                   # [1,1,30,1] bool
    n("And", ["notT_b", "colExtB"], "aew_n")
    n("And", ["isT_b", "rowExtW"], "aew_t")
    n("Or", ["aew_n", "aew_t"], "alongExtW")                   # [1,1,1,30] bool
    n("And", ["crossExtH", "alongExtW"], "gridC")              # [1,1,30,30] bool

    # canonical side mask: left iff along(=W) < mid -- pure 1-D, no Or.
    n("Less", ["idxW", "mid"], "leftW_b")                       # [1,1,1,30] bool

    # ---- scalar seed colours colorL, colorR from the along colour profile ----
    n("Cast", "leftW_b", "leftPos", to=F16)
    n("Sub", ["c_one", "leftPos"], "rightPos")
    n("Mul", ["alongColW", "leftPos"], "cLmask")
    n("ReduceSum", "cLmask", "colorLf", axes=[3], keepdims=1)   # [1,1,1,1]
    n("Mul", ["alongColW", "rightPos"], "cRmask")
    n("ReduceSum", "cRmask", "colorRf", axes=[3], keepdims=1)
    n("Cast", "colorLf", "colorL", to=TensorProto.UINT8)
    n("Cast", "colorRf", "colorR", to=TensorProto.UINT8)

    # ---- canonical label map Lc ----
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v10", np.array(10, np.uint8), np.uint8)
    n("Where", ["gridC", "v0", "v10"], "Lc0")                  # in-grid bg=0 else 10
    n("Where", ["leftW_b", "colorL", "colorR"], "glyphCol")    # uint8 [1,1,30,30]
    n("Where", ["glyphM", "glyphCol", "Lc0"], "Lc")           # uint8 [1,1,30,30]

    # transpose the finished label iff the instance is transposed
    n("Transpose", ["Lc"], "LcT", perm=[0, 1, 3, 2])          # [1,1,30,30] uint8
    n("Where", ["isT_b", "LcT", "Lc"], "L")                   # [1,1,30,30] uint8

    init("chan10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan10"], "output")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task284", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
