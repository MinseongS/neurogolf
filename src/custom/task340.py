"""task340 (ARC-AGI d687bc17) — "shoot each interior pixel to its matching wall".

Rule (reconstructed exactly from 266 stored examples, 0 mismatches):
  The grid is an H x W rectangle (H,W in [10,20]) anchored at (0,0) of the 30x30 canvas;
  everything beyond is background 0. Its 4 borders are solid one-colour walls of DISTINCT
  colours: top=tc (row 0), bottom=bc (row H-1), left=lc (col 0), right=rc (col W-1). Interior
  cells hold scattered single coloured pixels. Each interior pixel of colour v:
     v==tc -> row 1 (same col) ;  v==bc -> row H-2 (same col)
     v==lc -> col 1 (same row) ;  v==rc -> col W-2 (same row)
     v matches no wall -> VANISHES.  Walls kept; interior otherwise cleared.

Encoding (ONE fp16 index plane -> Equal -> FREE bool output; no [1,10,30,30] full plane):
  Hidx=H-1, Widx=W-1 from ramp*occupancy ReduceMax. Wall colours tc/bc/lc/rc =
  chramp.(per-channel max of the relevant border row/col). Per-channel column/row pixel
  COUNTS (ReduceSum of the FREE input); a wall colour's count is exactly 1 on its own wall
  line, so "column c carries an INTERIOR tc pixel"  ==  colcount[tc,c] > 1  (the +1 wall
  contribution cancels) -- no interior masking nor any variable-row slice needed.
  ONE packed outer-product MatMul Acol[1,1,30,10] @ Brow[1,1,10,30] sums all 8 lines AND
  folds an off-grid +10 sentinel into the same plane:
    k0..3 (rows 0,1,H-2,H-1): row-selectors x col-value-vectors {tc-wall, tc-topline,
           bc-botline, bc-wall}
    k4..7 (cols 0,1,W-2,W-1): row-value-vectors {lc-wall, lc-leftline, rc-rightline,
           rc-wall} x col-selectors
    k8 = 10*(r>Hidx) x ones ; k9 = ones x 10*(c>Widx)  -> og>=10 off-grid -> Equal false.
  output one-hot = Equal(og_fp16, chramp[1,10,1,1]) -> BOOL (fp16 Equal exact for ints).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from src.harness import IR_VERSION, DATA_TYPE, GRID_SHAPE

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64
N = 30


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype=None):
        a = np.ascontiguousarray(arr, dtype=dtype) if dtype is not None else np.ascontiguousarray(arr)
        inits.append(numpy_helper.from_array(a, name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, list(ins), [out], **attrs))
        return out

    # ---- ramps / constants -------------------------------------------------
    init("arow", np.arange(N, dtype=np.float16).reshape(1, 1, N, 1))   # [1,1,30,1]
    init("acol", np.arange(N, dtype=np.float16).reshape(1, 1, 1, N))   # [1,1,1,30]
    init("chramp", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    init("one16", np.array(1.0, np.float16))
    init("half16", np.array(0.5, np.float16))
    init("ten16", np.array(10.0, np.float16))
    init("zero16", np.array(0.0, np.float16))
    init("c1p5", np.array(1.5, np.float32))   # threshold for count>=2 (fp32 counts)

    # ---- grid extent scalars Hidx (=H-1), Widx (=W-1) ----------------------
    n("ReduceMax", ["input"], "rowocc", axes=[1, 3], keepdims=1)   # [1,1,30,1] f32
    n("ReduceMax", ["input"], "colocc", axes=[1, 2], keepdims=1)   # [1,1,1,30] f32
    n("Cast", ["rowocc"], "rowocc16", to=F16)
    n("Cast", ["colocc"], "colocc16", to=F16)
    n("Mul", ["rowocc16", "arow"], "rowx")
    n("Mul", ["colocc16", "acol"], "colx")
    n("ReduceMax", ["rowx"], "Hidx", axes=[2, 3], keepdims=1)      # [1,1,1,1] = H-1 (f16)
    n("ReduceMax", ["colx"], "Widx", axes=[2, 3], keepdims=1)      # [1,1,1,1] = W-1
    n("Sub", ["Hidx", "one16"], "Hm1")    # H-2
    n("Sub", ["Widx", "one16"], "Wm1")    # W-2

    init("sc1shape", np.array([1], np.int64))  # 1-element shape (avoids 0-dim params trap)
    n("Cast", ["Hidx"], "Hidx_i", to=I64)
    n("Cast", ["Widx"], "Widx_i", to=I64)
    n("Reshape", ["Hidx_i", "sc1shape"], "Hidx_s")  # [1]
    n("Reshape", ["Widx_i", "sc1shape"], "Widx_s")  # [1]

    # ---- per-channel column / row pixel counts (fp32) ----------------------
    n("ReduceSum", ["input"], "colcount", axes=[2], keepdims=1)    # [1,10,1,30] f32
    n("ReduceSum", ["input"], "rowcount", axes=[3], keepdims=1)    # [1,10,30,1] f32

    # ---- per-channel wall one-hots derived FROM the counts (no 10-ch slice) -
    # A wall is a full line; the wall colour's count on that line is W-2/H-2 (large),
    # every other channel ~0 there, so argmax over channels identifies it.
    #   top  wall colour: rowcount at row 0 ;  bottom: rowcount at row Hidx
    #   left wall colour: colcount at col 0 ;  right:  colcount at col Widx
    init("s_first", np.array([0, 0, 0, 0], np.int64))
    init("e_rowct0", np.array([1, 10, 1, 1], np.int64))   # rowcount[:, :, 0:1, :]
    init("ax_all", np.array([0, 1, 2, 3], np.int64))
    n("Slice", ["rowcount", "s_first", "e_rowct0", "ax_all"], "topw")  # [1,10,1,1]
    init("e_colct0", np.array([1, 10, 1, 1], np.int64))   # colcount[:, :, :, 0:1]
    n("Slice", ["colcount", "s_first", "e_colct0", "ax_all"], "leftw")  # [1,10,1,1]
    n("Gather", ["rowcount", "Hidx_s"], "botw_a", axis=2)  # [1,10,1,1] (row Hidx)
    n("Gather", ["colcount", "Widx_s"], "rightw_a", axis=3)  # [1,10,1,1] (col Widx)

    # colour scalar = sum_k k * argmax_onehot_k
    def wall_color(src, tag):
        n("ReduceMax", [src], tag + "_mx", axes=[1], keepdims=1)   # [1,1,1,1]
        n("Equal", [src, tag + "_mx"], tag + "_oh")                # [1,10,1,1] bool
        n("Cast", [tag + "_oh"], tag + "_ohf", to=F32)
        n("Mul", [tag + "_ohf", "chramp"], tag + "_kc")
        n("ReduceSum", [tag + "_kc"], tag + "_c32", axes=[1], keepdims=1)  # [1,1,1,1]
        n("Cast", [tag + "_c32"], tag + "_c", to=F16)
        return tag + "_c"
    tc = wall_color("topw", "topw")
    bc = wall_color("botw_a", "botw")
    lc = wall_color("leftw", "leftw")
    rc = wall_color("rightw_a", "rightw")

    # channel indices for gathering the count line (index shape [1])
    def chan_scalar(colf, tag):
        n("Cast", [colf], tag + "_i", to=I64)            # [1,1,1,1] int64
        n("Reshape", [tag + "_i", "sc1shape"], tag + "_s")  # [1]
        return tag + "_s"
    tci = chan_scalar(tc, "tci")
    bci = chan_scalar(bc, "bci")
    lci = chan_scalar(lc, "lci")
    rci = chan_scalar(rc, "rci")

    # ---- interior-presence per column/row of each wall colour --------------
    n("Gather", ["colcount", tci], "tc_cc", axis=1)   # [1,1,1,30] f32
    n("Gather", ["colcount", bci], "bc_cc", axis=1)
    n("Gather", ["rowcount", lci], "lc_rr", axis=1)   # [1,1,30,1] f32
    n("Gather", ["rowcount", rci], "rc_rr", axis=1)
    n("Greater", ["tc_cc", "c1p5"], "topcol_b")       # [1,1,1,30] bool
    n("Greater", ["bc_cc", "c1p5"], "botcol_b")
    n("Greater", ["lc_rr", "c1p5"], "leftrow_b")      # [1,1,30,1] bool
    n("Greater", ["rc_rr", "c1p5"], "rightrow_b")
    n("Cast", ["topcol_b"], "topcol", to=F16)
    n("Cast", ["botcol_b"], "botcol", to=F16)
    n("Cast", ["leftrow_b"], "leftrow", to=F16)
    n("Cast", ["rightrow_b"], "rightrow", to=F16)

    # ---- interior extent vectors (1 <= idx <= dim-2) -----------------------
    init("neg_half", np.array(-0.5, np.float16))
    n("Greater", ["acol", "half16"], "ci_lo")                 # c>=1
    n("Sub", ["Wm1", "acol"], "ci_hd")                        # (W-2)-c
    n("Greater", ["ci_hd", "neg_half"], "ci_hi")              # c<=W-2
    n("And", ["ci_lo", "ci_hi"], "colint_b")
    n("Cast", ["colint_b"], "colint", to=F16)                 # [1,1,1,30]
    n("Greater", ["arow", "half16"], "ri_lo")
    n("Sub", ["Hm1", "arow"], "ri_hd")
    n("Greater", ["ri_hd", "neg_half"], "ri_hi")
    n("And", ["ri_lo", "ri_hi"], "rowint_b")
    n("Cast", ["rowint_b"], "rowint", to=F16)                 # [1,1,30,1]

    # ---- row / col selectors (integer Equal, fp16 exact) -------------------
    def sel(ramp, target, tag):
        n("Equal", [ramp, target], tag + "_b")
        return n("Cast", [tag + "_b"], tag, to=F16)
    sel("arow", "zero16", "rs0")     # r==0
    sel("arow", "one16", "rs1")      # r==1
    sel("arow", "Hm1", "rsHm2")      # r==H-2
    sel("arow", "Hidx", "rsHm1")     # r==H-1
    sel("acol", "zero16", "cs0")     # c==0
    sel("acol", "one16", "cs1")      # c==1
    sel("acol", "Wm1", "csWm2")      # c==W-2
    sel("acol", "Widx", "csWm1")     # c==W-1

    # off-grid sentinel vectors: 10 on rows r>Hidx / cols c>Widx.
    # reuse the row/col occupancy (1 in-grid, 0 off-grid): roff = 10*(1-occ).
    n("Sub", ["one16", "rowocc16"], "row_off")   # [1,1,30,1] 1 off-grid
    n("Mul", ["row_off", "ten16"], "roff")
    n("Sub", ["one16", "colocc16"], "col_off")   # [1,1,1,30]
    n("Mul", ["col_off", "ten16"], "coff")
    init("ones_r", np.ones((1, 1, N, 1), np.float16))
    init("ones_c", np.ones((1, 1, 1, N), np.float16))

    # ---- col-value vectors (Brow rows 0..3) and row-value vectors (Acol cols 4..7)
    n("Mul", [tc, "colint"], "tc_wall")     # top wall row   [1,1,1,30]
    n("Mul", [tc, "topcol"], "tc_line")     # top line row
    n("Mul", [bc, "botcol"], "bc_line")     # bottom line row
    n("Mul", [bc, "colint"], "bc_wall")     # bottom wall row
    n("Mul", [lc, "rowint"], "lc_wall")     # left wall col  [1,1,30,1]
    n("Mul", [lc, "leftrow"], "lc_line")    # left line col
    n("Mul", [rc, "rightrow"], "rc_line")   # right line col
    n("Mul", [rc, "rowint"], "rc_wall")     # right wall col

    # ---- ONE packed MatMul: Acol[1,1,30,10] @ Brow[1,1,10,30] --------------
    # Acol columns (over rows): k0..3 row-selectors, k4..7 row-value-vecs, k8 off-row, k9 ones
    n("Concat", ["rs0", "rs1", "rsHm2", "rsHm1",
                 "lc_wall", "lc_line", "rc_line", "rc_wall",
                 "roff", "ones_r"], "Acol", axis=3)        # [1,1,30,10]
    # Brow rows (over cols): k0..3 col-value-vecs, k4..7 col-selectors, k8 ones, k9 off-col
    n("Concat", ["tc_wall", "tc_line", "bc_line", "bc_wall",
                 "cs0", "cs1", "csWm2", "csWm1",
                 "ones_c", "coff"], "Brow", axis=2)        # [1,1,10,30]
    n("MatMul", ["Acol", "Brow"], "og")                    # [1,1,30,30] f16

    # ---- Equal(og, channel-ramp) straight into the FREE bool output --------
    init("chan", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1))
    n("Equal", ["og", "chan"], "output")                   # [1,10,30,30] bool (FREE)

    x = helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", BOOL, GRID_SHAPE)
    graph = helper.make_graph(nodes, "task340", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
