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
  H = #occupied rows = Sum(rowocc16) (grid is a SOLID origin-anchored rect, no gaps),
  Hidx=H-1, Widx=W-1.  Wall colour = Sum_k k*(border-line count_k > 0): a solid wall line
  holds ONLY its own colour (interior/garbage pixels are >=2 cells from every wall), so the
  weighted sum recovers the wall's k directly (no ReduceMax/Equal argmax).  Per-channel
  column/row pixel COUNTS (ReduceSum of FREE input); a wall colour's count is exactly 1 on
  its own wall line, so "column c carries an INTERIOR tc pixel" == colcount[tc,c] > 1 (the
  +1 wall contribution cancels) -- no interior masking nor any variable-row slice.
  ONE packed outer-product MatMul Acol[1,1,30,9] @ Brow[1,1,9,30] sums 8 lines AND folds a
  uniform in-grid +1 into the same plane:
    k0..3 (rows 0,1,H-2,H-1): row-selectors x col-value-vectors {tc-wall, tc-topline,
           bc-botline, bc-wall}
    k4..7 (cols 0,1,W-2,W-1): row-value-vectors {lc-wall, lc-leftline, rc-rightline,
           rc-wall} x col-selectors
    k8 = rowocc16 x colocc16  -> +1 on every in-grid cell.
  Then output one-hot = Equal(og, colour+1 ramp) -> BOOL: in-grid bg og=1 -> ch0; coloured
  cell colour k -> og=k+1 -> ch k; off-grid og=0 -> matches no channel (ramp starts at 1)
  -> all-false (the empty off-grid one-hot the generator emits).  Value vectors are built
  with Where(mask_bool, colour, 0) so no fp16 mask planes are materialised.
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
    init("one16", np.array(1.0, np.float16))
    init("half16", np.array(0.5, np.float16))
    init("zero16", np.array(0.0, np.float16))
    init("c1p5", np.array(1.5, np.float32))   # threshold for count>=2 (fp32 counts)

    # ---- grid extent scalars Hidx (=H-1), Widx (=W-1) ----------------------
    # The grid is a SOLID H x W rectangle anchored at origin, so rows 0..H-1 and
    # cols 0..W-1 are all occupied (no gaps) -> H = #occupied rows = Sum(rowocc16),
    # Hidx = H-1.  This drops the rowx/colx (rowocc*ramp) Mul planes entirely.
    n("ReduceMax", ["input"], "rowocc", axes=[1, 3], keepdims=1)   # [1,1,30,1] f32
    n("ReduceMax", ["input"], "colocc", axes=[1, 2], keepdims=1)   # [1,1,1,30] f32
    n("Cast", ["rowocc"], "rowocc16", to=F16)
    n("Cast", ["colocc"], "colocc16", to=F16)
    n("ReduceSum", ["rowocc16"], "Hcnt", axes=[2, 3], keepdims=1)  # [1,1,1,1] = H (f16)
    n("ReduceSum", ["colocc16"], "Wcnt", axes=[2, 3], keepdims=1)  # [1,1,1,1] = W
    n("Sub", ["Hcnt", "one16"], "Hidx")   # H-1
    n("Sub", ["Wcnt", "one16"], "Widx")   # W-1
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

    # colour scalar = sum_k k * (count_k > 0).  A solid wall line contains ONLY
    # its own colour (garbage/interior pixels are >=2 cells from every wall), so
    # exactly one channel has count>0 there -> the weighted sum recovers its k.
    # Everything past the bool test runs in fp16 (the 10-ch [1,10,1,1] working
    # planes halve to 20B; values are 0-9 so fp16 is exact).
    init("zero32", np.array(0.0, np.float32))
    init("chramp16", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1))
    def wall_color(src, tag):
        n("Greater", [src, "zero32"], tag + "_pos")               # [1,10,1,1] bool
        n("Where", [tag + "_pos", "chramp16", "zero16"], tag + "_kc")  # k where present
        n("ReduceSum", [tag + "_kc"], tag + "_c", axes=[1], keepdims=1)  # [1,1,1,1] f16
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

    # ---- interior-presence per column/row of each wall colour (kept BOOL) --
    n("Gather", ["colcount", tci], "tc_cc", axis=1)   # [1,1,1,30] f32
    n("Gather", ["colcount", bci], "bc_cc", axis=1)
    n("Gather", ["rowcount", lci], "lc_rr", axis=1)   # [1,1,30,1] f32
    n("Gather", ["rowcount", rci], "rc_rr", axis=1)
    n("Greater", ["tc_cc", "c1p5"], "topcol_b")       # [1,1,1,30] bool
    n("Greater", ["bc_cc", "c1p5"], "botcol_b")
    n("Greater", ["lc_rr", "c1p5"], "leftrow_b")      # [1,1,30,1] bool
    n("Greater", ["rc_rr", "c1p5"], "rightrow_b")

    # ---- interior extent masks (kept BOOL) --------------------------------
    # c<=W-2  ==  c<W-1  ==  Less(c, Widx);  no Sub-vector plane needed.
    n("Greater", ["acol", "half16"], "ci_lo")                 # c>=1
    n("Less", ["acol", "Widx"], "ci_hi")                      # c<=W-2
    n("And", ["ci_lo", "ci_hi"], "colint_b")                  # [1,1,1,30] bool
    n("Greater", ["arow", "half16"], "ri_lo")
    n("Less", ["arow", "Hidx"], "ri_hi")                      # r<=H-2
    n("And", ["ri_lo", "ri_hi"], "rowint_b")                  # [1,1,30,1] bool

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

    # ---- in-grid indicator term (replaces both off-grid sentinels) ---------
    # og gets a uniform +1 on every in-grid cell (rank-1: rowocc16 x colocc16,
    # both already computed = the in-grid occupancy).  Then chan = colour+1, so
    # in-grid bg -> og=1 -> ch0; coloured cell colour k -> og=k+1 -> ch k;
    # off-grid -> og=0 -> matches no channel (chan starts at 1) -> all-false.

    # ---- value vectors via Where(mask_bool, colour, 0): no fp16 mask casts --
    n("Where", ["colint_b", tc, "zero16"], "tc_wall")    # top wall row   [1,1,1,30]
    n("Where", ["topcol_b", tc, "zero16"], "tc_line")    # top line row
    n("Where", ["botcol_b", bc, "zero16"], "bc_line")    # bottom line row
    n("Where", ["colint_b", bc, "zero16"], "bc_wall")    # bottom wall row
    n("Where", ["rowint_b", lc, "zero16"], "lc_wall")    # left wall col  [1,1,30,1]
    n("Where", ["leftrow_b", lc, "zero16"], "lc_line")   # left line col
    n("Where", ["rightrow_b", rc, "zero16"], "rc_line")  # right line col
    n("Where", ["rowint_b", rc, "zero16"], "rc_wall")    # right wall col

    # ---- ONE packed MatMul: Acol[1,1,30,9] @ Brow[1,1,9,30] ----------------
    # Acol cols (over rows): k0..3 row-selectors, k4..7 row-value-vecs, k8 in-grid row
    n("Concat", ["rs0", "rs1", "rsHm2", "rsHm1",
                 "lc_wall", "lc_line", "rc_line", "rc_wall",
                 "rowocc16"], "Acol", axis=3)              # [1,1,30,9]
    # Brow rows (over cols): k0..3 col-value-vecs, k4..7 col-selectors, k8 in-grid col
    n("Concat", ["tc_wall", "tc_line", "bc_line", "bc_wall",
                 "cs0", "cs1", "csWm2", "csWm1",
                 "colocc16"], "Brow", axis=2)              # [1,1,9,30]
    n("MatMul", ["Acol", "Brow"], "og")                    # [1,1,30,30] f16

    # ---- Equal(og, colour+1 ramp) straight into the FREE bool output -------
    init("chan", (np.arange(10, dtype=np.float16) + 1).reshape(1, 10, 1, 1))
    n("Equal", ["og", "chan"], "output")                   # [1,10,30,30] bool (FREE)

    x = helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", BOOL, GRID_SHAPE)
    graph = helper.make_graph(nodes, "task340", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
