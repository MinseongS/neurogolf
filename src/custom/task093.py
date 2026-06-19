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

Recovery — OCCUPANCY-ONLY + ORIENTATION-EQUIVARIANCE (task341 lever) + CONTIGUOUS-SPAN.
  The actual gray value (5) is irrelevant: in the INPUT the band line is the ONLY
  fully-occupied line, the stack counts are occupied-pixel counts, and the OUTPUT
  gray region is one contiguous run per perpendicular line.  So we work purely on
  occupancy occ = (sum_k k*input_k > 0), a single bool/fp16 14x14 plane.

  The rule is transpose-equivariant, so canonicalise to a horizontal band:
  horiz = (some row is fully occupied);  C = horiz ? occ : occ^T.  Band rows
  [r0,r1] = rows with rowSum(C)==CW.  Per-column counts na=Sum_{r<r0}C[r,c],
  nb=Sum_{r>r1}C[r,c] via masked-sum MatMul (contract the row axis; no 2-D product
  plane).  KEY: above-stack [r0-na,r0-1], band [r0,r1], below-stack [r1+1,r1+nb]
  are CONTIGUOUS, so gray(r,c) = (r0-na[c]) <= r <= (r1+nb[c]) — just two compares
  against the row ramp gated by tiny column-indexed thresholds.  De-canonicalise
  (uint8 5/0 then horiz?LcH:LcH^T) and route to the FREE bool output via
  Pad(255 sentinel)+Equal([0..9]) — ch5=gray, ch0=in-grid bg, off-grid all-zero.

  ONLY fp16 14x14 planes are occ, occ^T, C (occupancy + canonicalisation); ORT has
  fp16 Where (NOT bool Where) under ORT_DISABLE_ALL.  Entry is one fp32 [1,1,30,30]
  Conv (3600B, irreducible 10->1 floor) + one fp32 14x14 crop (784B) + one uint8
  30x30 output carrier (900B).  ~15.97 pts (vs the two-branch value-plane net 15.67).
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

    # ---- constants ----
    init("colW", np.arange(10).reshape(1, 10, 1, 1), np.float32)   # value conv
    init("half", np.array(0.5, np.float16), np.float16)
    init("half32", np.array(0.5, np.float32), np.float32)
    init("cwm", np.array(CW - 0.5, np.float16), np.float16)  # CW-0.5: "row fully occupied"
    init("BIG", np.array(1e4, np.float16), np.float16)
    init("nBIG", np.array(-1e4, np.float16), np.float16)
    init("rowidx", np.arange(CW).reshape(1, 1, CW, 1).astype(np.float16), np.float16)
    # row ramp laid out on the LAST axis (for the contract-row MatMul)
    init("rowidxC", np.arange(CW).reshape(1, 1, 1, CW).astype(np.float16), np.float16)
    init("chan", np.arange(10).reshape(1, 10, 1, 1), np.uint8)
    init("c5u8", np.array(5, np.uint8), np.uint8)
    init("c0u8", np.array(0, np.uint8), np.uint8)
    init("crop", np.array([0, 0, 0, 0, 0, 0, CW - 30, CW - 30], np.int64), np.int64)
    init("padO", np.array([0, 0, 0, 0, 0, 0, 30 - CW, 30 - CW], np.int64), np.int64)
    init("sentU8", np.array(255, np.uint8), np.uint8)

    # ---- OCCUPANCY is all we need (gray value 5 is irrelevant): the band line is
    #      the only fully-occupied line in the INPUT, counts are occupied pixels,
    #      and the output gray span is a contiguous run.  occBool = (V32>0.5),
    #      cropped to the 14x14 grid in one Pad. ----
    n("Conv", ["input", "colW"], "V32")          # [1,1,30,30] fp32 (the 3600B entry)
    n("Pad", ["V32", "crop"], "Vc")              # [1,1,CW,CW] fp32
    n("Greater", ["Vc", "half32"], "occBool")    # [1,1,CW,CW] bool occupancy
    n("Cast", ["occBool"], "occF", to=F16)       # fp16 {0,1}

    # ---- orientation: the band is a fully-occupied ROW iff some row sums to CW. ----
    n("ReduceSum", ["occF"], "rowOccV", axes=[3], keepdims=1)   # [1,1,CW,1]
    n("ReduceMax", ["rowOccV"], "maxRowOcc", axes=[2, 3], keepdims=1)
    n("Greater", ["maxRowOcc", "cwm"], "horizB")               # scalar: band is a row?

    # ---- canonicalise occupancy: C = horiz ? occF : occF^T (band always a row).
    #      ORT has fp16 Where (NOT bool Where) under ORT_DISABLE_ALL. ----
    n("Transpose", ["occF"], "occFt", perm=[0, 1, 3, 2])
    n("Where", ["horizB", "occF", "occFt"], "C")  # [1,1,CW,CW] fp16 canonical occ

    # ---- band rows [r0,r1] on the canonical plane: a band row is fully occupied. ----
    n("ReduceSum", ["C"], "rowOcc", axes=[3], keepdims=1)   # [1,1,CW,1]
    n("Greater", ["rowOcc", "cwm"], "rowBandB")            # [1,1,CW,1] bool (==CW)
    n("Where", ["rowBandB", "rowidx", "BIG"], "r0_t")
    n("ReduceMin", ["r0_t"], "hr0", axes=[2, 3], keepdims=1)  # scalar fp16
    n("Where", ["rowBandB", "rowidx", "nBIG"], "r1_t")
    n("ReduceMax", ["r1_t"], "hr1", axes=[2, 3], keepdims=1)

    # per-column counts via masked-sum MatMul (contract the row axis)
    n("Less", ["rowidxC", "hr0"], "aboveRowB")   # [1,1,1,CW]
    n("Cast", ["aboveRowB"], "aboveRow", to=F16)
    n("Greater", ["rowidxC", "hr1"], "belowRowB")
    n("Cast", ["belowRowB"], "belowRow", to=F16)
    n("MatMul", ["aboveRow", "C"], "na")         # [1,1,1,CW] count above band
    n("MatMul", ["belowRow", "C"], "nb")         # [1,1,1,CW] count below band

    # KEY: above-stack [r0-na, r0-1], band [r0,r1], below-stack [r1+1, r1+nb] are
    # CONTIGUOUS, so the full gray span in column c is just  lo <= r <= hi  with
    # lo=r0-na[c], hi=r1+nb[c] (tiny vectors).  Only TWO full compares + one And.
    n("Sub", ["hr0", "na"], "lo")                # [1,1,1,CW] = r0 - na[c]
    n("Add", ["hr1", "nb"], "hi")                # [1,1,1,CW] = r1 + nb[c]
    n("Sub", ["lo", "half"], "loM")              # lo-0.5
    n("Add", ["hi", "half"], "hiP")              # hi+0.5
    n("Greater", ["rowidx", "loM"], "geLo")      # [1,1,CW,CW] bool: r >= lo
    n("Less", ["rowidx", "hiP"], "leHi")         # [1,1,CW,CW] bool: r <= hi
    n("And", ["geLo", "leHi"], "grayH")          # [1,1,CW,CW] canonical gray

    # ---- canonical L (uint8 5/0); de-canonicalise via uint8 Where (ORT OK) ----
    n("Where", ["grayH", "c5u8", "c0u8"], "LcH")     # [1,1,CW,CW] uint8 canonical
    n("Transpose", ["LcH"], "LcHt", perm=[0, 1, 3, 2])
    n("Where", ["horizB", "LcH", "LcHt"], "Lc")      # uint8 (5 / 0)

    # ---- output: Lc -> Pad(255) -> Equal([0..9]) -> BOOL ----
    n("Pad", ["Lc", "padO", "sentU8"], "L", mode="constant")  # [1,1,30,30]
    n("Equal", ["L", "chan"], "output")          # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task093", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
