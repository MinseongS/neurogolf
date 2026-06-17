"""task093 (ARC-AGI 4093f84a) — "pixels fall toward the gray horizon and stack".

Rule (from the generator, size=14 grid anchored top-left):
  * A solid GRAY(5) band ("horizon") spans the full width (or full height after
    transpose), thickness 2-5, at a fixed location.
  * Scattered single coloured pixels (one non-gray colour) sit off the band.
  * Each coloured pixel falls toward the band along the axis perpendicular to it
    and STACKS contiguously against the band edge.  Equivalently: per line
    perpendicular to the band, count the coloured pixels on each side; that many
    gray cells stack against the band on that side.
  * flip (swaps the two sides — symmetric, so a no-op for the rule) and xpose
    (rotates band horizontal<->vertical) may be applied.
  * OUTPUT colour is always GRAY: band + stacked cells -> 5, else 0.

Recovery (closed-form, reduction/broadcast-based, NO flood-fill):
  Work on a 14x14 canvas.  V = value plane (sum_k k*input_k).  G = gray(V==5),
  C = coloured (V>0 & V!=5).  Detect a fully-gray ROW (horizontal band) vs COL.
  HORIZONTAL: band rows [r0,r1].  Per-column counts na=sum_{r<r0}C,
  nb=sum_{r>r1}C.  gray(r,c) = band OR (1<=(r0-r)<=na[c]) OR (1<=(r-r1)<=nb[c]);
  each side is a broadcast of a per-row distance [1,1,14,1] against a per-column
  count [1,1,1,14] -> [1,1,14,14] (separable, no 2-D scan).  VERTICAL: rows<->cols.
  Select by a scalar horiz flag using bool And/Or (no fp Where).

  Entry value plane is one fp32 [1,1,30,30] (3600 B, irreducible 10->1 floor);
  everything downstream runs on fp16 [1,1,14,14] (392 B) or bool (196 B) planes.
  Output: L_u8 (5 where gray) -> Pad to 30x30 sentinel -> Equal([0..9]) -> BOOL.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

CW = 14  # working canvas side (grid always 14x14 top-left)


def build(task):
    inits, nodes = [], []

    def init(name, arr, npdtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=npdtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    F16 = TensorProto.FLOAT16
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    # ---- constants (fp16 for the small-canvas comparisons) ----
    init("colW", np.arange(10).reshape(1, 10, 1, 1), np.float32)   # value conv
    init("half", np.array(0.5, np.float16), np.float16)
    init("c45", np.array(4.5, np.float16), np.float16)
    init("c55", np.array(5.5, np.float16), np.float16)
    init("BIG", np.array(1e4, np.float16), np.float16)
    init("nBIG", np.array(-1e4, np.float16), np.float16)
    init("rowidx", np.arange(CW).reshape(1, 1, CW, 1).astype(np.float16), np.float16)
    init("colidx", np.arange(CW).reshape(1, 1, 1, CW).astype(np.float16), np.float16)
    # ramps along the CONTRACTED axis for masked-sum MatMuls
    init("rowidxC", np.arange(CW).reshape(1, 1, 1, CW).astype(np.float16), np.float16)  # row ramp on last axis
    init("colidxC", np.arange(CW).reshape(1, 1, CW, 1).astype(np.float16), np.float16)  # col ramp on row axis
    init("chan", np.arange(10).reshape(1, 10, 1, 1), np.uint8)
    init("c5u8", np.array(5, np.uint8), np.uint8)
    init("c0u8", np.array(0, np.uint8), np.uint8)
    init("crop", np.array([0, 0, 0, 0, 0, 0, CW - 30, CW - 30], np.int64), np.int64)
    init("padO", np.array([0, 0, 0, 0, 0, 0, 30 - CW, 30 - CW], np.int64), np.int64)
    init("sentU8", np.array(255, np.uint8), np.uint8)

    # ---- value plane V (one fp32 30x30 = the irreducible 3600B entry) ----
    n("Conv", ["input", "colW"], "V32")          # [1,1,30,30] fp32
    n("Pad", ["V32", "crop"], "Vc")              # [1,1,CW,CW] fp32
    n("Cast", ["Vc"], "V", to=F16)               # fp16 from here on

    # gray mask G (V==5), coloured mask C (V>0 & V!=5)
    n("Greater", ["V", "c45"], "g_lo")
    n("Less", ["V", "c55"], "g_hi")
    n("And", ["g_lo", "g_hi"], "Gb")             # gray bool
    n("Greater", ["V", "half"], "occB")          # any non-bg
    n("Not", ["Gb"], "notGb")
    n("And", ["occB", "notGb"], "Cb")            # coloured bool
    n("Cast", ["Cb"], "C", to=F16)               # fp16 {0,1}

    # ---- orientation: a fully-gray ROW has min-over-cols == 5 (all cells gray);
    #      any non-band row contains a bg(0) cell -> min < 5. Reuses V (no G plane). ----
    n("ReduceMin", ["V"], "rowVmin", axes=[3], keepdims=1)   # [1,1,CW,1]
    n("ReduceMin", ["V"], "colVmin", axes=[2], keepdims=1)   # [1,1,1,CW]
    n("Greater", ["rowVmin", "c45"], "rowFullB")   # min >= 5  => all gray
    n("Greater", ["colVmin", "c45"], "colFullB")
    n("Cast", ["rowFullB"], "rowFull", to=F16)
    n("ReduceMax", ["rowFull"], "horizScore", axes=[2, 3], keepdims=1)  # scalar
    n("Greater", ["horizScore", "half"], "horizB")          # scalar bool

    # ================= HORIZONTAL branch =================
    n("Where", ["rowFullB", "rowidx", "BIG"], "hr0_t")
    n("ReduceMin", ["hr0_t"], "hr0", axes=[2, 3], keepdims=1)   # scalar fp16
    n("Where", ["rowFullB", "rowidx", "nBIG"], "hr1_t")
    n("ReduceMax", ["hr1_t"], "hr1", axes=[2, 3], keepdims=1)
    # per-column counts via masked-sum MatMul (contract the row axis): no 2-D product plane
    n("Less", ["rowidxC", "hr0"], "h_aboveRowB")   # [1,1,1,CW] (row ramp on last axis)
    n("Cast", ["h_aboveRowB"], "h_aboveRow", to=F16)
    n("Greater", ["rowidxC", "hr1"], "h_belowRowB")
    n("Cast", ["h_belowRowB"], "h_belowRow", to=F16)
    n("MatMul", ["h_aboveRow", "C"], "h_na")       # [1,1,1,CW] = sum_r C[r,c]*1[r<r0]
    n("MatMul", ["h_belowRow", "C"], "h_nb")       # [1,1,1,CW]
    n("Add", ["h_na", "half"], "h_naH")            # [1,1,1,CW] tiny: na+0.5
    n("Add", ["h_nb", "half"], "h_nbH")
    n("Sub", ["hr0", "rowidx"], "h_distA")         # [1,1,CW,1] = r0-r (tiny)
    n("Greater", ["h_distA", "half"], "h_distApos")  # dist>=1
    n("Less", ["h_distA", "h_naH"], "h_aLE")       # dist<=na  -> bool plane direct
    n("And", ["h_distApos", "h_aLE"], "h_grayAboveB")
    n("Sub", ["rowidx", "hr1"], "h_distB")
    n("Greater", ["h_distB", "half"], "h_distBpos")
    n("Less", ["h_distB", "h_nbH"], "h_bLE")
    n("And", ["h_distBpos", "h_bLE"], "h_grayBelowB")
    n("Or", ["Gb", "h_grayAboveB"], "h_g1")
    n("Or", ["h_g1", "h_grayBelowB"], "h_grayB")   # [1,1,CW,CW] bool

    # ================= VERTICAL branch =================
    n("Where", ["colFullB", "colidx", "BIG"], "vc0_t")
    n("ReduceMin", ["vc0_t"], "vc0", axes=[2, 3], keepdims=1)
    n("Where", ["colFullB", "colidx", "nBIG"], "vc1_t")
    n("ReduceMax", ["vc1_t"], "vc1", axes=[2, 3], keepdims=1)
    # per-row counts via masked-sum MatMul (contract the col axis)
    n("Less", ["colidxC", "vc0"], "v_leftColB")    # [1,1,CW,1] (col ramp on row axis)
    n("Cast", ["v_leftColB"], "v_leftCol", to=F16)
    n("Greater", ["colidxC", "vc1"], "v_rightColB")
    n("Cast", ["v_rightColB"], "v_rightCol", to=F16)
    n("MatMul", ["C", "v_leftCol"], "v_nl")        # [1,1,CW,1] = sum_c C[r,c]*1[c<c0]
    n("MatMul", ["C", "v_rightCol"], "v_nr")       # [1,1,CW,1]
    n("Add", ["v_nl", "half"], "v_nlH")            # [1,1,CW,1] tiny
    n("Add", ["v_nr", "half"], "v_nrH")
    n("Sub", ["vc0", "colidx"], "v_distL")         # [1,1,1,CW] tiny
    n("Greater", ["v_distL", "half"], "v_distLpos")
    n("Less", ["v_distL", "v_nlH"], "v_lLE")       # bool plane direct
    n("And", ["v_distLpos", "v_lLE"], "v_grayLeftB")
    n("Sub", ["colidx", "vc1"], "v_distR")
    n("Greater", ["v_distR", "half"], "v_distRpos")
    n("Less", ["v_distR", "v_nrH"], "v_rLE")
    n("And", ["v_distRpos", "v_rLE"], "v_grayRightB")
    n("Or", ["Gb", "v_grayLeftB"], "v_g1")
    n("Or", ["v_g1", "v_grayRightB"], "v_grayB")

    # ================= select by scalar horiz flag (bool ops only) =====
    n("And", ["horizB", "h_grayB"], "sel_h")        # broadcast scalar over plane
    n("Not", ["horizB"], "vertB")
    n("And", ["vertB", "v_grayB"], "sel_v")
    n("Or", ["sel_h", "sel_v"], "grayB")            # [1,1,CW,CW] bool

    # ================= output: L_u8 -> Pad -> Equal =================
    n("Where", ["grayB", "c5u8", "c0u8"], "Lc")     # [1,1,CW,CW] uint8 (5 / 0)
    n("Pad", ["Lc", "padO", "sentU8"], "L", mode="constant")  # [1,1,30,30]
    n("Equal", ["L", "chan"], "output")              # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task093", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
