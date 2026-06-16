"""task340 (ARC d687bc17): a coloured border frame (top=ec0, right=ec1,
bottom=ec2, left=ec3, corners background) plus, for every interior pixel whose
colour equals one of the four edge colours, a projection onto the inner ring:
  ec0 (top)    -> output[1][col]      ec1 (right)  -> output[row][W-2]
  ec2 (bottom) -> output[H-2][col]    ec3 (left)   -> output[row][1]
Garbage pixels (no edge-colour match) vanish; border is preserved.

Low-memory label-map (Tier B) design:
- NO 1x1 colour-index Conv: H/W and per-row/col occupancy come from
  ReduceMax(input, axes=[1,3]/[1,2]) (bg sets ch0=1 in-grid, 0 off-grid).
- ONE all-ones depthwise Conv per axis gives per-(row|col) channel counts
  (rowsum[1,10,30,1], colsum[1,10,1,30]); reshaped to [10,30] (rsr/csc).
- the 10-channel axis is contracted with small MATMULS instead of [1,10,30]
  Mul+ReduceSum planes: border-line channel counts = rsr@RselB[30,2] / csc@CselB,
  the per-edge dominant one-hot (max & >0) -> [10,2], edge colours = krow@oh,
  and projection per-line counts = ohT@(csc|rsr) -> [2,30]. "interior present"
  = count>=2 (the border line itself contributes exactly 1 to its own edge
  channel in every interior column/row).
- placement assembled with ONE packed MatMul A[30,8]@B[8,30]: cols 0..3 are
  row-selectors x col-value-vectors (column-indexed lines at rows 0,1,H-2,H-1),
  cols 4..7 are row-value-vectors x col-selectors (row-indexed lines at cols
  0,1,W-2,W-1); the 8 disjoint lines sum into one colour-index plane (no Add),
  masked to the in-grid rectangle, then Equal(L,arange) into the FREE bool
  output. All canvas-sized math is float16.
"""

import numpy as np
import onnx

from src.harness import DATA_TYPE, GRID_SHAPE, IR_VERSION

F16 = onnx.TensorProto.FLOAT16
F32 = onnx.TensorProto.FLOAT
I32 = onnx.TensorProto.INT32
U8 = onnx.TensorProto.UINT8


def build(task):
    nodes, inits = [], []

    def init(name, arr, dtype=None):
        a = np.ascontiguousarray(arr, dtype=dtype) if dtype is not None else np.ascontiguousarray(arr)
        inits.append(onnx.numpy_helper.from_array(a, name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(onnx.helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants (fp16) ----
    init("half", np.array(0.5, np.float16))
    init("one16", np.array(1.0, np.float16))
    init("two16", np.array(2.0, np.float16))
    init("c1p5", np.array(1.5, np.float16))   # threshold for count>=2
    init("zeroi", np.array(0, np.int32))
    init("arow", np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1))
    init("acol", np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30))
    # ---- occupancy / H,W (no colour-index plane) ----
    n("ReduceMax", ["input"], "rowocc32", axes=[1, 3], keepdims=1)   # [1,1,30,1] f32
    n("ReduceMax", ["input"], "colocc32", axes=[1, 2], keepdims=1)   # [1,1,1,30] f32
    n("Cast", ["rowocc32"], "rowocc", to=F16)
    n("Cast", ["colocc32"], "colocc", to=F16)
    n("ReduceSum", ["rowocc"], "H", axes=[2], keepdims=1)            # [1,1,1,1]
    n("ReduceSum", ["colocc"], "W", axes=[3], keepdims=1)
    n("Sub", ["H", "one16"], "Hm1")
    n("Sub", ["H", "two16"], "Hm2")
    n("Sub", ["W", "one16"], "Wm1")
    n("Sub", ["W", "two16"], "Wm2")

    # ---- edge colour SCALARS from fixed border cells of the one-hot input ----
    # Each border line is fully filled by its edge colour, so on that line the
    # edge channel's count is the maximum across channels (>> the corner-bg=2).
    # Derive ALL four edge channel one-hots from the per-row / per-col channel
    # counts (the two depthwise Convs below) -- no per-channel slice planes.

    # ---- per-axis presence Conv (all rows / all cols), depthwise ----
    Kcol = np.ones((10, 1, 30, 1), np.float32); init("Kcol", Kcol)
    n("Conv", ["input", "Kcol"], "colsum32", group=10)              # sum over rows -> [1,10,1,30]
    Krow = np.ones((10, 1, 1, 30), np.float32); init("Krow", Krow)
    n("Conv", ["input", "Krow"], "rowsum32", group=10)              # sum over cols -> [1,10,30,1]
    n("Cast", ["colsum32"], "colsum", to=F16)   # [1,10,1,30] per-column channel counts
    n("Cast", ["rowsum32"], "rowsum", to=F16)   # [1,10,30,1] per-row channel counts

    def line_eq(idx, target, tag):
        n("Sub", [idx, target], f"{tag}_d")
        n("Cast", [f"{tag}_d"], f"{tag}_di", to=I32)
        n("Equal", [f"{tag}_di", "zeroi"], f"{tag}_b")
        return f"{tag}_b"

    init("zeroh", np.array(0.0, np.float16))
    # row selectors (over rows axis=2): r==0 (top), r==H-1 (bottom)
    sr0_ = line_eq("arow", "zeroh", "selR0"); n("Cast", [sr0_], "selR0f", to=F16)  # [1,1,30,1]
    sbH1 = line_eq("arow", "Hm1", "selH1"); n("Cast", [sbH1], "selH1f", to=F16)
    # col selectors (over cols axis=3): c==0 (left), c==W-1 (right)
    sc0_ = line_eq("acol", "zeroh", "selC0"); n("Cast", [sc0_], "selC0f", to=F16)  # [1,1,1,30]
    scW1 = line_eq("acol", "Wm1", "selW1"); n("Cast", [scW1], "selW1f", to=F16)

    # ---- contract the 10-channel axis with MatMuls (no [1,10,30] intermediates) ----
    # reshape counts to [10,30]: rsr = rowsum (per-row counts), csc = colsum (per-col)
    init("shp10x30", np.array([10, 30], np.int64))
    n("Reshape", ["rowsum", "shp10x30"], "rsr")   # [10,30]  rsr[k,r]=#colour-k in row r
    n("Reshape", ["colsum", "shp10x30"], "csc")   # [10,30]  csc[k,c]=#colour-k in col c

    # border-line channel counts: select the two border rows/cols via [30,2] matrices
    init("shp30x2", np.array([30, 2], np.int64))
    n("Concat", ["selR0f", "selH1f"], "RselB4", axis=3)   # [1,1,30,2]
    n("Reshape", ["RselB4", "shp30x2"], "RselB")          # [30,2]
    n("Concat", ["selC0f", "selW1f"], "CselB4", axis=2)   # [1,1,2,30] -> need [30,2]
    init("shp2x30", np.array([2, 30], np.int64))
    n("Reshape", ["CselB4", "shp2x30"], "CselB2")         # [2,30]
    n("Transpose", ["CselB2"], "CselB", perm=[1, 0])      # [30,2]
    n("MatMul", ["rsr", "RselB"], "RbC")   # [10,2]: col0=top-border counts, col1=bottom
    n("MatMul", ["csc", "CselB"], "CbC")   # [10,2]: col0=left-border counts,  col1=right

    # per-edge one-hot over the 10-channel axis (dominant, >0). RbC/CbC are [10,2].
    def edges_oh(mat, tag):
        n("ReduceMax", [mat], f"{tag}_mx", axes=[0], keepdims=1)   # [1,2]
        n("Equal", [mat, f"{tag}_mx"], f"{tag}_eq")                # [10,2] bool
        n("Greater", [mat, "half"], f"{tag}_pos")
        n("And", [f"{tag}_eq", f"{tag}_pos"], f"{tag}_b")
        return n("Cast", [f"{tag}_b"], tag, to=F16)                # [10,2]
    edges_oh("RbC", "Roh")   # [10,2] top(col0)/bottom(col1) one-hots
    edges_oh("CbC", "Coh")   # [10,2] left(col0)/right(col1) one-hots

    # edge colour indices = k . onehot  -> [1,2] each
    init("krow", np.arange(10, dtype=np.float16).reshape(1, 10))   # [1,10]
    n("MatMul", ["krow", "Roh"], "Rcol")   # [1,2]: [topc, botc]
    n("MatMul", ["krow", "Coh"], "Ccol")   # [1,2]: [leftc, rightc]
    # split to scalars [1,1,1,1] for broadcasting
    init("zi", np.array([0], np.int64)); init("ti", np.array([1], np.int64)); init("twoi", np.array([2], np.int64)); init("ax1", np.array([1], np.int64))
    def scal(src, lo, hi, tag):
        n("Slice", [src, lo, hi, "ax1"], f"{tag}_s")              # [1,1]
        return n("Reshape", [f"{tag}_s", "shp1111"], tag)         # [1,1,1,1]
    init("shp1111", np.array([1, 1, 1, 1], np.int64))
    topc = scal("Rcol", "zi", "ti", "topc")
    botc = scal("Rcol", "ti", "twoi", "botc")
    leftc = scal("Ccol", "zi", "ti", "leftc")
    rightc = scal("Ccol", "ti", "twoi", "rightc")

    # ---- projection presence per column/row (interior pixels of the edge col) ----
    # top/bottom per-COLUMN counts: [2,10]@[10,30]=[2,30]; rows=[top,bot]
    n("Transpose", ["Roh"], "RohT", perm=[1, 0])   # [2,10]
    n("Transpose", ["Coh"], "CohT", perm=[1, 0])   # [2,10]
    n("MatMul", ["RohT", "csc"], "TB")   # [2,30]: row0=top per-col, row1=bottom per-col
    n("MatMul", ["CohT", "rsr"], "LR")   # [2,30]: row0=left per-row, row1=right per-row
    # presence: count >= 2 (border line itself contributes exactly 1)
    n("Greater", ["TB", "c1p5"], "TB_b"); n("Cast", ["TB_b"], "TBp", to=F16)  # [2,30]
    n("Greater", ["LR", "c1p5"], "LR_b"); n("Cast", ["LR_b"], "LRp", to=F16)  # [2,30]
    # split rows to per-direction vectors and shape for placement
    init("shp1130", np.array([1, 1, 1, 30], np.int64))
    init("shp1301", np.array([1, 1, 30, 1], np.int64))
    init("ax0", np.array([0], np.int64))
    def rowvec(src, lo, hi, shp, tag):
        n("Slice", [src, lo, hi, "ax0"], f"{tag}_s")   # [1,30]
        return n("Reshape", [f"{tag}_s", shp], tag)
    rowvec("TBp", "zi", "ti", "shp1130", "tp")   # [1,1,1,30] top presence per col
    rowvec("TBp", "ti", "twoi", "shp1130", "bp")
    rowvec("LRp", "zi", "ti", "shp1301", "lp")   # [1,1,30,1] left presence per row
    rowvec("LRp", "ti", "twoi", "shp1301", "rp")

    # value lines for projections (colour where present)
    n("Mul", ["tp", topc], "tval")       # [1,1,1,30]
    n("Mul", ["bp", botc], "bval")
    n("Mul", ["lp", leftc], "lval")      # [1,1,30,1]
    n("Mul", ["rp", rightc], "rval")

    # ---- border line value vectors (in interior extent: 1 <= idx <= dim-2) ----
    init("neg_half", np.array(-0.5, np.float16))
    # column interior: 1 <= c <= W-2
    n("Greater", ["acol", "half"], "ci_lo")               # c>=1
    n("Sub", ["Wm2", "acol"], "ci_hd2"); n("Greater", ["ci_hd2", "neg_half"], "ci_hi2")  # c<=W-2
    n("And", ["ci_lo", "ci_hi2"], "colint_b"); n("Cast", ["colint_b"], "colint", to=F16)  # [1,1,1,30]
    # row interior: 1 <= r <= H-2
    n("Greater", ["arow", "half"], "ri_lo")
    n("Sub", ["Hm2", "arow"], "ri_hd"); n("Greater", ["ri_hd", "neg_half"], "ri_hi")
    n("And", ["ri_lo", "ri_hi"], "rowint_b"); n("Cast", ["rowint_b"], "rowint", to=F16)   # [1,1,30,1]

    n("Mul", [topc, "colint"], "topline")    # [1,1,1,30]
    n("Mul", [botc, "colint"], "botline")
    n("Mul", [leftc, "rowint"], "leftline")  # [1,1,30,1]
    n("Mul", [rightc, "rowint"], "rightline")

    # ---- assemble all 8 lines via ONE packed MatMul A[30,8] @ B[8,30] ----
    # k=0..3: column-indexed lines  -> A[r,k]=(r==R_k)  , B[k,c]=colvalvec_k(c)
    #   (rows 0,1,H-2,H-1 carry topline, tval, bval, botline)
    # k=4..7: row-indexed lines     -> A[r,k]=rowvalvec_k(r), B[k,c]=(c==C_k)
    #   (cols 0,1,W-2,W-1 carry leftline, lval, rval, rightline)
    init("shpA", np.array([30, 8], np.int64))
    init("shpB", np.array([8, 30], np.int64))
    # row selectors: r==0 (selR0f, reuse), r==1, r==H-2, r==H-1 (selH1f, reuse)
    sr1 = line_eq("arow", "one16", "r1"); n("Cast", [sr1], "r1f", to=F16)
    srH2 = line_eq("arow", "Hm2", "rH2"); n("Cast", [srH2], "rH2f", to=F16)
    # A columns (over rows): row-selectors then row-value-vectors  -> [1,1,30,8]
    n("Concat", ["selR0f", "r1f", "rH2f", "selH1f", "leftline", "lval", "rval", "rightline"],
      "Acol", axis=3)
    n("Reshape", ["Acol", "shpA"], "Amat")                          # [30,8]
    # col selectors: c==0 (selC0f, reuse), c==1, c==W-2, c==W-1 (selW1f, reuse)
    sc1 = line_eq("acol", "one16", "c1"); n("Cast", [sc1], "c1f", to=F16)
    scW2 = line_eq("acol", "Wm2", "cW2"); n("Cast", [scW2], "cW2f", to=F16)
    # B rows (over cols): col-value-vectors then col-selectors      -> [1,1,8,30]
    n("Concat", ["topline", "tval", "bval", "botline", "selC0f", "c1f", "cW2f", "selW1f"],
      "Brow", axis=2)
    n("Reshape", ["Brow", "shpB"], "Bmat")                          # [8,30]
    n("MatMul", ["Amat", "Bmat"], "og")                             # [30,30] f16 colour index

    # ---- in-grid mask (row<H & col<W) ----
    n("Sub", ["H", "arow"], "H_r"); n("Greater", ["H_r", "half"], "rin_b")
    n("Sub", ["W", "acol"], "W_c"); n("Greater", ["W_c", "half"], "cin_b")
    n("And", ["rin_b", "cin_b"], "gm_b")                            # [1,1,30,30] bool

    # ---- label map L (uint8) then Equal -> free bool output ----
    n("Cast", ["og"], "og_u8", to=U8)
    init("v10", np.array(10, np.uint8), np.uint8)
    n("Where", ["gm_b", "og_u8", "v10"], "L")                       # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                             # free BOOL

    x = onnx.helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)
    graph = onnx.helper.make_graph(nodes, "task340", [x], [y], inits)
    return onnx.helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[onnx.helper.make_opsetid("", 11)])
