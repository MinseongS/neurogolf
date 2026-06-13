"""task340 (ARC d687bc17): each interior pixel slides to the border edge whose
color equals its own (top->row1, right->col W-2, bottom->row H-2, left->col1);
garbage pixels vanish; the border is preserved.

Low-memory design (no [1,10,30,30] float32 intermediates, f16 spatial math):
- 4 depthwise Convs over the one-hot `input` (free) give per-channel
  directional pixel counts as tiny tensors. "skip row0/col0" kernels exclude
  the near border; the far (variable-position) border is removed by subtracting
  the interior-span indicator.
- the 4 edge color values come from a 1x1-Conv integer color grid `cg`.
- per-edge: select the edge channel (one-hot via Equal), reduce to a 1-D
  presence vector, scale by the color -> a 1-D value line.
- two small MatMuls turn the four 1-D value lines into the placement grid
  (top/bottom share columns, left/right share rows), add the masked border,
  mask outside the grid to -1, one-hot expand (Equal vs [0..9]) into `output`.
All spatial arithmetic is float16 (values are integers <= 30, exact in f16),
halving every canvas-sized intermediate.
"""

import numpy as np
import onnx

from src.harness import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

F16 = onnx.TensorProto.FLOAT16
I32 = onnx.TensorProto.INT32


def build(task):
    nodes, inits = [], []

    def init(name, arr, dtype=None):
        a = np.ascontiguousarray(arr, dtype=dtype) if dtype is not None else np.ascontiguousarray(arr)
        inits.append(onnx.numpy_helper.from_array(a, name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(onnx.helper.make_node(op, ins, [out], **attrs))
        return out

    # float16 constants
    init("one", np.array(1.0, np.float16))
    init("half", np.array(0.5, np.float16))
    init("neg_half", np.array(-0.5, np.float16))
    init("zeroi", np.array(0, np.int32))
    init("arow", np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1))
    init("acol", np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30))
    init("kveci", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1))

    # --- integer color grid cg32 [1,1,30,30] (f32; only small reductions of
    # it are kept, then cast to f16 -> no f16 canvas copy of the grid) ---
    init("kvec", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    n("Conv", ["input", "kvec"], "cg32", kernel_shape=[1, 1])

    # --- H, W via small max-color-per-row/col tensors ---
    n("ReduceMax", ["cg32"], "rowmaxc32", axes=[3], keepdims=1)
    n("Cast", ["rowmaxc32"], "rowmaxc", to=F16)              # [1,1,30,1]
    n("ReduceMax", ["cg32"], "colmaxc32", axes=[2], keepdims=1)
    n("Cast", ["colmaxc32"], "colmaxc", to=F16)              # [1,1,1,30]
    n("Greater", ["rowmaxc", "zeroh"], "rocc_b"); n("Cast", ["rocc_b"], "rowocc", to=F16)
    n("Greater", ["colmaxc", "zeroh"], "cocc_b"); n("Cast", ["cocc_b"], "colocc", to=F16)
    init("zeroh", np.array(0.0, np.float16))
    n("ReduceSum", ["rowocc"], "H", axes=[2], keepdims=1)
    n("ReduceSum", ["colocc"], "W", axes=[3], keepdims=1)
    n("Sub", ["H", "one"], "Hm1")
    n("Sub", ["H", "two"], "Hm2"); init("two", np.array(2.0, np.float16))
    n("Sub", ["W", "one"], "Wm1")
    n("Sub", ["W", "two"], "Wm2")

    # --- interior-span indicators (1-D, f16) ---
    def span(idx, hi, tag):
        n("Greater", [idx, "half"], f"{tag}_ge")
        n("Sub", [hi, idx], f"{tag}_hd")
        n("Greater", [f"{tag}_hd", "neg_half"], f"{tag}_le")
        n("And", [f"{tag}_ge", f"{tag}_le"], f"{tag}_b")
        return n("Cast", [f"{tag}_b"], f"{tag}", to=F16)

    rowint = span("arow", "Hm2", "rowint")   # [1,1,30,1]
    colint = span("acol", "Wm2", "colint")   # [1,1,1,30]

    # --- edge color scalars ---
    init("r0_s", np.array([0], np.int64)); init("r0_e", np.array([1], np.int64))
    init("r_ax", np.array([2], np.int64))
    n("Slice", ["cg32", "r0_s", "r0_e", "r_ax"], "row0")
    n("ReduceMax", ["row0"], "topc32", axes=[2, 3], keepdims=1)
    n("Cast", ["topc32"], "topc", to=F16)
    init("c0_s", np.array([0], np.int64)); init("c0_e", np.array([1], np.int64))
    init("c_ax", np.array([3], np.int64))
    n("Slice", ["cg32", "c0_s", "c0_e", "c_ax"], "col0")
    n("ReduceMax", ["col0"], "leftc32", axes=[2, 3], keepdims=1)
    n("Cast", ["leftc32"], "leftc", to=F16)

    def line_eq(idx, target, tag):
        n("Sub", [idx, target], f"{tag}_d")
        n("Cast", [f"{tag}_d"], f"{tag}_di", to=I32)
        n("Equal", [f"{tag}_di", "zeroi"], f"{tag}_b")
        return f"{tag}_b"

    # bottomc = max color in row H-1; rightc = max color in col W-1
    sbrow = line_eq("arow", "Hm1", "rb")
    n("Cast", [sbrow], "rbf", to=F16)
    n("Mul", ["rowmaxc", "rbf"], "botsel")
    n("ReduceSum", ["botsel"], "bottomc", axes=[2, 3], keepdims=1)
    srcol = line_eq("acol", "Wm1", "cr")
    n("Cast", [srcol], "crf", to=F16)
    n("Mul", ["colmaxc", "crf"], "rsel")
    n("ReduceSum", ["rsel"], "rightc", axes=[2, 3], keepdims=1)

    # --- edge channel one-hots [1,10,1,1] (f16) ---
    def ch_onehot(colorc, tag):
        n("Cast", [colorc], f"{tag}_ci", to=I32)
        n("Equal", ["kveci", f"{tag}_ci"], f"{tag}_eb")
        return n("Cast", [f"{tag}_eb"], f"{tag}_oh", to=F16)

    tch = ch_onehot("topc", "tc")
    bch = ch_onehot("bottomc", "bc")
    lch = ch_onehot("leftc", "lc")
    rch = ch_onehot("rightc", "rc")

    # --- directional per-channel counts via depthwise Conv on input (free) ---
    Kcol_skip0 = np.zeros((10, 1, 30, 1), np.float32); Kcol_skip0[:, 0, 1:, 0] = 1.0
    init("Kcol_skip0", Kcol_skip0)
    n("Conv", ["input", "Kcol_skip0"], "colsum_s0_32", group=10)
    n("Cast", ["colsum_s0_32"], "colsum_s0", to=F16)
    Kcol_all = np.ones((10, 1, 30, 1), np.float32); init("Kcol_all", Kcol_all)
    n("Conv", ["input", "Kcol_all"], "colsum_all_32", group=10)
    n("Cast", ["colsum_all_32"], "colsum_all", to=F16)
    Krow_skip0 = np.zeros((10, 1, 1, 30), np.float32); Krow_skip0[:, 0, 0, 1:] = 1.0
    init("Krow_skip0", Krow_skip0)
    n("Conv", ["input", "Krow_skip0"], "rowsum_s0_32", group=10)
    n("Cast", ["rowsum_s0_32"], "rowsum_s0", to=F16)
    Krow_all = np.ones((10, 1, 1, 30), np.float32); init("Krow_all", Krow_all)
    n("Conv", ["input", "Krow_all"], "rowsum_all_32", group=10)
    n("Cast", ["rowsum_all_32"], "rowsum_all", to=F16)

    # --- select edge channel, reduce to 1-D ---
    def sel_vec(counts, choh, tag):
        n("Mul", [counts, choh], f"{tag}_m")
        return n("ReduceSum", [f"{tag}_m"], f"{tag}_v", axes=[1], keepdims=1)

    tv = sel_vec("colsum_s0", tch, "tv")     # [1,1,1,30]
    bv = sel_vec("colsum_all", bch, "bv")    # incl bottom-border row
    lv = sel_vec("rowsum_s0", lch, "lv")     # [1,1,30,1]
    rv = sel_vec("rowsum_all", rch, "rv")    # incl right-border col

    n("Sub", [bv, "colint"], "bv2")
    n("Sub", [rv, "rowint"], "rv2")
    n("Greater", [tv, "half"], "tp_b"); n("Cast", ["tp_b"], "tp", to=F16)
    n("Greater", ["bv2", "half"], "bp_b"); n("Cast", ["bp_b"], "bp", to=F16)
    n("Greater", [lv, "half"], "lp_b"); n("Cast", ["lp_b"], "lp", to=F16)
    n("Greater", ["rv2", "half"], "rp_b"); n("Cast", ["rp_b"], "rp", to=F16)
    n("Mul", ["tp", "topc"], "tval")         # [1,1,1,30]
    n("Mul", ["bp", "bottomc"], "bval")
    n("Mul", ["lp", "leftc"], "lval")        # [1,1,30,1]
    n("Mul", ["rp", "rightc"], "rval")

    # border-line value vectors (full edge lines reconstructed via the MatMuls)
    n("Mul", ["topc", "colint"], "topline")     # [1,1,1,30] top border row
    n("Mul", ["bottomc", "colint"], "botline")
    n("Mul", ["leftc", "rowint"], "leftline")   # [1,1,30,1] left border col
    n("Mul", ["rightc", "rowint"], "rightline")

    # --- placement+border grids via two [30,4]@[4,30] MatMuls (f16) ---
    init("shpA", np.array([30, 4], np.int64))
    init("shpB", np.array([4, 30], np.int64))
    # column grid: rows {0,1,H-2,H-1} <- {topline, tval, bval, botline}
    sr0 = line_eq("arow", "zeroh", "r0"); n("Cast", [sr0], "r0f", to=F16)
    sr1 = line_eq("arow", "one", "r1"); n("Cast", [sr1], "r1f", to=F16)
    srH2 = line_eq("arow", "Hm2", "rH2"); n("Cast", [srH2], "rH2f", to=F16)
    srH1 = line_eq("arow", "Hm1", "rH1"); n("Cast", [srH1], "rH1f", to=F16)
    n("Concat", ["r0f", "r1f", "rH2f", "rH1f"], "Acol", axis=3)   # [1,1,30,4]
    n("Reshape", ["Acol", "shpA"], "Amat")                        # [30,4]
    n("Concat", ["topline", "tval", "bval", "botline"], "Bcol", axis=2)  # [1,1,4,30]
    n("Reshape", ["Bcol", "shpB"], "Bmat")                        # [4,30]
    n("MatMul", ["Amat", "Bmat"], "colMM")                        # [30,30]
    # row grid: cols {0,1,W-2,W-1} <- {leftline, lval, rval, rightline}
    sc0 = line_eq("acol", "zeroh", "c0"); n("Cast", [sc0], "c0f", to=F16)
    sc1 = line_eq("acol", "one", "c1"); n("Cast", [sc1], "c1f", to=F16)
    scW2 = line_eq("acol", "Wm2", "cW2"); n("Cast", [scW2], "cW2f", to=F16)
    scW1 = line_eq("acol", "Wm1", "cW1"); n("Cast", [scW1], "cW1f", to=F16)
    n("Concat", ["leftline", "lval", "rval", "rightline"], "Arow", axis=3)  # [1,1,30,4]
    n("Reshape", ["Arow", "shpA"], "Amat2")                       # [30,4]
    n("Concat", ["c0f", "c1f", "cW2f", "cW1f"], "Brow", axis=2)   # [1,1,4,30]
    n("Reshape", ["Brow", "shpB"], "Bmat2")                       # [4,30]
    n("MatMul", ["Amat2", "Bmat2"], "rowMM")                      # [30,30]

    # --- combine (disjoint per cell) ---
    n("Add", ["colMM", "rowMM"], "og")

    # --- mask outside grid to -1: ogm = og + gm - 1 (og==0 outside) ---
    n("Sub", ["H", "arow"], "H_r"); n("Greater", ["H_r", "half"], "rin_b")
    n("Sub", ["W", "acol"], "W_c"); n("Greater", ["W_c", "half"], "cin_b")
    n("And", ["rin_b", "cin_b"], "gm_b")
    n("Cast", ["gm_b"], "gm", to=F16)
    n("Add", ["og", "gm"], "ogp")
    n("Sub", ["ogp", "one"], "ogm")

    # --- one-hot expand into output ---
    n("Cast", ["ogm"], "ogm_i", to=I32)
    init("chvec", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1))
    n("Equal", ["ogm_i", "chvec"], "oh_b")
    n("Cast", ["oh_b"], "output", to=DATA_TYPE)

    x = onnx.helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = onnx.helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = onnx.helper.make_graph(nodes, "task340", [x], [y], inits)
    return onnx.helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
