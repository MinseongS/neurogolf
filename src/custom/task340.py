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
- placement assembled with ONE packed 4-D MatMul Acol[1,1,30,10]@Brow[1,1,10,30]
  (batched contraction of the packed-line axis -> NO [30,*]/[*,30] reshapes):
  lines 0..3 are row-selectors x col-value-vectors (column-indexed lines at rows
  0,1,H-2,H-1), lines 4..7 are row-value-vectors x col-selectors (row-indexed
  lines at cols 0,1,W-2,W-1); the 8 disjoint lines sum into one colour-index
  plane. Lines 8..9 fold the OFF-GRID sentinel straight into the same MatMul
  (col 8 = 10*(r>=H) x all-ones, col 9 = all-ones x 10*(c>=W)) so og>=10 off-grid
  and no separate in-grid mask / Where plane materialises.
- Equal(og_fp16, channel-ramp) routes the 10-channel one-hot expansion into the
  FREE bool output (fp16 Equal is exact for integer operands under ORT_DISABLE_ALL,
  no extra cast plane). Line selectors are direct fp16 Equal(idx,target) (integer-
  exact, no int32 cast/Sub planes). Projection presence(count>=2) AND per-direction
  edge colour fuse into ONE Where each. All canvas-sized math is float16.
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
        # idx and target are integer-valued fp16 -> Equal is exact (no int32 cast,
        # no Sub plane).
        n("Equal", [idx, target], f"{tag}_b")
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
    # FUSE presence(count>=2) AND per-direction edge colour in ONE Where each:
    #   TBval[d,c] = (TB[d,c]>1.5) ? Rcol[d] : 0    (Rcol=[topc,botc] as [2,1])
    n("Reshape", ["Rcol", "shp21"], "Rcol2")            # [2,1]
    n("Reshape", ["Ccol", "shp21"], "Ccol2")            # [2,1]
    init("shp21", np.array([2, 1], np.int64))
    init("zeroh2", np.array(0.0, np.float16))
    n("Greater", ["TB", "c1p5"], "TB_b")
    n("Where", ["TB_b", "Rcol2", "zeroh2"], "TBval")    # [2,30]
    n("Greater", ["LR", "c1p5"], "LR_b")
    n("Where", ["LR_b", "Ccol2", "zeroh2"], "LRval")    # [2,30]
    # split rows -> per-direction value vectors (already carrying the colour)
    init("shp1130", np.array([1, 1, 1, 30], np.int64))
    init("shp1301", np.array([1, 1, 30, 1], np.int64))
    init("ax0", np.array([0], np.int64))
    def rowvec(src, lo, hi, shp, tag):
        n("Slice", [src, lo, hi, "ax0"], f"{tag}_s")   # [1,30]
        return n("Reshape", [f"{tag}_s", shp], tag)
    rowvec("TBval", "zi", "ti", "shp1130", "tval")   # [1,1,1,30] top projection values
    rowvec("TBval", "ti", "twoi", "shp1130", "bval")
    rowvec("LRval", "zi", "ti", "shp1301", "lval")   # [1,1,30,1] left projection values
    rowvec("LRval", "ti", "twoi", "shp1301", "rval")

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
    # Pack the off-grid sentinel directly into the colour-index MatMul so no
    # separate in-grid mask plane / Where ever materialises:
    #   extra col 8 of A = 10*(r>=H), B row 8 = all-ones  -> +10 on off-grid rows
    #   extra col 9 of A = all-ones,  B row 9 = 10*(c>=W)  -> +10 on off-grid cols
    # so og >= 10 anywhere off-grid (Equal-to-channel false => background), while
    # in-grid cells keep their colour index 0..9.
    init("ten16", np.array(10.0, np.float16))
    # row selectors: r==0 (selR0f, reuse), r==1, r==H-2, r==H-1 (selH1f, reuse)
    sr1 = line_eq("arow", "one16", "r1"); n("Cast", [sr1], "r1f", to=F16)
    srH2 = line_eq("arow", "Hm2", "rH2"); n("Cast", [srH2], "rH2f", to=F16)
    # off-grid row vector: 10*(r>=H) == 10*(H-r < 0.5)
    n("Sub", ["H", "arow"], "H_r"); n("Less", ["H_r", "half"], "roff_b")
    n("Where", ["roff_b", "ten16", "zeroh"], "roff")               # [1,1,30,1]
    # off-grid col vector: 10*(c>=W)
    n("Sub", ["W", "acol"], "W_c"); n("Less", ["W_c", "half"], "coff_b")
    n("Where", ["coff_b", "ten16", "zeroh"], "coff")               # [1,1,1,30]
    init("ones30r", np.ones((1, 1, 30, 1), np.float16))
    init("ones30c", np.ones((1, 1, 1, 30), np.float16))
    # A columns (over rows): row-selectors, row-value-vectors, then 2 off-grid cols
    n("Concat", ["selR0f", "r1f", "rH2f", "selH1f", "leftline", "lval", "rval", "rightline",
                 "roff", "ones30r"], "Acol", axis=3)                 # [1,1,30,10]
    # col selectors: c==0 (selC0f, reuse), c==1, c==W-2, c==W-1 (selW1f, reuse)
    sc1 = line_eq("acol", "one16", "c1"); n("Cast", [sc1], "c1f", to=F16)
    scW2 = line_eq("acol", "Wm2", "cW2"); n("Cast", [scW2], "cW2f", to=F16)
    # B rows (over cols): col-value-vectors, col-selectors, then 2 off-grid rows
    n("Concat", ["topline", "tval", "bval", "botline", "selC0f", "c1f", "cW2f", "selW1f",
                 "ones30c", "coff"], "Brow", axis=2)                 # [1,1,10,30]
    # batched 4-D MatMul contracts the packed-line axis directly -> no reshapes
    n("MatMul", ["Acol", "Brow"], "og")                             # [1,1,30,30] f16

    # ---- Equal(og, channel-ramp) straight into the FREE bool output ----
    # og is fp16 with integer values 0..9 in-grid and >=10 off-grid; Equal
    # accepts fp16 under ORT_DISABLE_ALL with no extra counted cast plane.
    init("chan", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1))
    n("Equal", ["og", "chan"], "output")                            # free BOOL

    x = onnx.helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, GRID_SHAPE)
    graph = onnx.helper.make_graph(nodes, "task340", [x], [y], inits)
    return onnx.helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[onnx.helper.make_opsetid("", 11)])
