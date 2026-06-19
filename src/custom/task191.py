"""task191 (ARC-AGI 7df24a62) — template-match a small sprite under all 8 dihedral
orientations across a yellow-noise field, stamp a blue box (bbox dilated by 1) at every
match, keep the yellow dots.

Rule (from generator task_7df24a62.py, size=23 fixed):
  Input: a blue square (channel 1 = blue frame) enclosing a small yellow (channel 4) pattern
  of shape tall x wide, tall in {1,2,3}, wide in {2,3}, pattern touches all 4 bbox edges and
  has exactly max(tall,wide) yellow cells; PLUS scattered yellow noise dots over the grid.
  Output: for every grid position (mrow,mcol) and every dihedral orientation (4 rot x 2 xpose)
  where the yellow noise EXACTLY equals the oriented pattern (every pattern-yellow present AND
  no extra yellow inside the oriented bbox), draw a blue box = oriented-bbox dilated by 1.
  Then overlay the yellow dots on top. (The reference sprite region reproduces itself.)

Encoding (template-matching as a stacked Conv; verified 500/500 fresh, mem 31276, pts 14.62):
  - Y = channel 4 sliced AND cropped to the fixed 23x23 active grid in ONE Slice (fp32 2116B).
  - Extract the sprite: brow=(min blue row)+1, bcol=(min blue col)+1 and tall/wide from the blue
    bbox extent — all recovered as SCALARS from two no-pad blue-channel profile Convs (per-row /
    per-col blue counts), no full blue plane. K3 = 3x3 yellow window at (brow,bcol) gathered from
    Y (gather indices CLAMPED to [0,22] since the window can run off the 23x23 grid at bottom/right;
    the clamped duplicate row/col is masked out by rows>=tall / cols>=wide). npat = sum(K3).
  - 8 oriented 3x3 kernels Ko = FIXED index-permutations of K3's 9 flat elements (PERMS).
    mask3_o = bbox-fill of Ko (triangular-MatMul prefix/suffix-OR per axis, outer-product).
  - COMBINED, BIASED match Conv (ONE plane each for scores+indicator):
      combk encodes "pattern-yellow present AND no extra yellow in the oriented bbox"; corr<=npat
      with equality exactly at a match. We BIAS the Conv by -(npat-0.5) (runtime per-channel bias)
      so its output is corr-(npat-0.5) = +0.5 at a match, <=-0.5 otherwise; ONE Relu yields the
      fp16 {0,0.5} match indicator M directly — no separate Equal-bool nor fp16 Cast plane.
  - box: forward grouped-SUM Conv stamps mask3_o (spatially flipped) at every match anchor and
    collapses the 8 orientations into ONE [1,1,23,23] plane; one 3x3 MaxPool dilates by 1.
  - output: ONE uint8 colour-index plane (yellow=4 > blue=1 > bg=0) built with two uint8 Where ops
    at 23x23, Pad-99 to 30x30 (uint8 Pad; off-grid sentinel 99 matches no channel -> all-off), then
    Equal(colidx, arange[1,10,1,1]) routes the 10-ch one-hot into the FREE bool output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64
U8 = TensorProto.UINT8

H = 30
G = 23  # fixed active grid size (generator size=23; grid is the top-left 23x23 of the 30x30 canvas)
BVAL = 100.0  # extra-yellow penalty in the combined match kernel (fp16-exact, < 2048)

# orientation permutations of flattened 3x3 (row-major), Ko[i] = K3[perm[i]]
PERMS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 3, 6, 1, 4, 7, 2, 5, 8],
    [2, 5, 8, 1, 4, 7, 0, 3, 6],
    [2, 1, 0, 5, 4, 3, 8, 7, 6],
    [8, 7, 6, 5, 4, 3, 2, 1, 0],
    [8, 5, 2, 7, 4, 1, 6, 3, 0],
    [6, 3, 0, 7, 4, 1, 8, 5, 2],
    [6, 7, 8, 3, 4, 5, 0, 1, 2],
]


def build(task):
    inits, nodes = [], []

    np_of = {F32: np.float32, F16: np.float16, BOOL: np.bool_, I64: np.int64,
             U8: np.uint8}

    def init(name, arr, dtype):
        npd = np_of.get(dtype, dtype)
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=npd), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ===== yellow plane: slice channel 4 AND crop to the fixed 23x23 active grid in ONE Slice =====
    # (the grid is always the top-left 23x23, so off-grid yellow never exists -> 23x23 is exact).
    init("y_s", np.array([4, 0, 0], np.int64), np.int64)     # ch4, row0, col0
    init("y_e", np.array([5, G, G], np.int64), np.int64)     # ch5, row23, col23
    init("y_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "y_s", "y_e", "y_ax"], "Y")         # [1,1,23,23] fp32

    # ===== brow, bcol from blue bbox =====
    # rowhas[1,1,30,1], colhas[1,1,1,30] via no-pad profile Convs that pick the BLUE channel and
    # collapse one spatial axis in ONE op — avoids slicing the full [1,1,30,30] blue plane (3600B).
    wrow = np.zeros((1, 10, 1, H), np.float32)
    wrow[0, 1, 0, :] = 1.0      # sum over columns of the blue channel -> per-row count
    wcol = np.zeros((1, 10, H, 1), np.float32)
    wcol[0, 1, :, 0] = 1.0      # per-col count
    init("wrow", wrow, F32)
    init("wcol", wcol, F32)
    n("Conv", ["input", "wrow"], "rowcnt")   # [1,1,30,1] fp32
    n("Conv", ["input", "wcol"], "colcnt")   # [1,1,1,30] fp32
    n("Greater", ["rowcnt", "zero1111b"], "rowhas_b")
    n("Greater", ["colcnt", "zero1111b"], "colhas_b")
    init("zero1111b", np.array([0.0], np.float32).reshape(1, 1, 1, 1), F32)
    n("Cast", ["rowhas_b"], "rowhas", to=F32)   # [1,1,30,1] {0,1}
    n("Cast", ["colhas_b"], "colhas", to=F32)   # [1,1,1,30]
    # min index with blue = min over r of (r if has else BIG)
    ramp_r = np.arange(H, dtype=np.float32).reshape(1, 1, H, 1)
    ramp_c = np.arange(H, dtype=np.float32).reshape(1, 1, 1, H)
    init("rampr", ramp_r, F32)
    init("rampc", ramp_c, F32)
    init("BIG", np.array([999.0], np.float32).reshape(1, 1, 1, 1), F32)
    # val = rowhas*ramp + (1-rowhas)*BIG
    n("Mul", ["rowhas", "rampr"], "rr_a")
    init("one", np.array([1.0], np.float32).reshape(1, 1, 1, 1), F32)
    n("Sub", ["one", "rowhas"], "rr_n")
    n("Mul", ["rr_n", "BIG"], "rr_b")
    n("Add", ["rr_a", "rr_b"], "rr_v")
    n("ReduceMin", ["rr_v"], "minr", axes=[2], keepdims=0)   # [1,1,1]
    n("Mul", ["colhas", "rampc"], "cc_a")
    n("Sub", ["one", "colhas"], "cc_n")
    n("Mul", ["cc_n", "BIG"], "cc_b")
    n("Add", ["cc_a", "cc_b"], "cc_v")
    n("ReduceMin", ["cc_v"], "minc", axes=[3], keepdims=0)   # [1,1,1]
    # max index with blue = max over r of (r*rowhas)
    n("ReduceMax", ["rr_a"], "maxr", axes=[2], keepdims=0)   # [1,1,1]
    n("ReduceMax", ["cc_a"], "maxc", axes=[3], keepdims=0)   # [1,1,1]
    # tall = maxr - minr - 1 ; wide = maxc - minc - 1
    n("Sub", ["maxr", "minr"], "tw_r")
    n("Sub", ["maxc", "minc"], "tw_c")
    init("one11", np.array([1.0], np.float32).reshape(1, 1, 1), F32)
    n("Sub", ["tw_r", "one11"], "tall")   # [1,1,1]
    n("Sub", ["tw_c", "one11"], "wide")   # [1,1,1]
    # brow = minr+1, bcol = minc+1; build index vectors [brow,brow+1,brow+2]
    n("Reshape", ["minr", "scalar1"], "minr_s")
    n("Reshape", ["minc", "scalar1"], "minc_s")
    init("scalar1", np.array([1], np.int64), np.int64)
    init("off012", np.array([1.0, 2.0, 3.0], np.float32), F32)  # +1 for brow then 0,1,2
    n("Add", ["minr_s", "off012"], "ridx_f0")   # [3]
    n("Add", ["minc_s", "off012"], "cidx_f0")   # [3]
    # Clamp gather indices to [0, G-1]: Y is the 23x23 grid; when the 3x3 window's last row/col
    # would exceed the grid (sprite near the bottom/right edge, tall<3 or wide<3), that out-of-range
    # row/col is masked OUT below (rows>=tall / cols>=wide) so clamping to a duplicate edge is safe.
    init("gmax", np.array([G - 1.0], np.float32), F32)
    init("gzero", np.array([0.0], np.float32), F32)
    n("Clip", ["ridx_f0", "gzero", "gmax"], "ridx_f")
    n("Clip", ["cidx_f0", "gzero", "gmax"], "cidx_f")
    n("Cast", ["ridx_f"], "ridx", to=I64)
    n("Cast", ["cidx_f"], "cidx", to=I64)

    # ===== K3 = 3x3 yellow window at (brow,bcol) =====
    # Gather rows then cols from Y[1,1,30,30]
    n("Gather", ["Y", "ridx"], "Yr", axis=2)   # [1,1,3,30]
    n("Gather", ["Yr", "cidx"], "K3raw", axis=3)  # [1,1,3,3]
    # mask out rows>=tall and cols>=wide (3x3 window can exceed the tall x wide pattern,
    # capturing adjacent noise yellows just outside the sprite frame).
    a3r = np.arange(3, dtype=np.float32).reshape(1, 1, 3, 1)
    a3c = np.arange(3, dtype=np.float32).reshape(1, 1, 1, 3)
    init("a3r", a3r, F32)
    init("a3c", a3c, F32)
    # tall/wide are [1,1,1]; reshape to [1,1,1,1] to broadcast against [1,1,3,1]
    n("Reshape", ["tall", "sh1111"], "tall4")
    n("Reshape", ["wide", "sh1111"], "wide4")
    init("sh1111", np.array([1, 1, 1, 1], np.int64), np.int64)
    n("Less", ["a3r", "tall4"], "rkeep_b")   # [1,1,3,1] bool
    n("Less", ["a3c", "wide4"], "ckeep_b")   # [1,1,1,3]
    n("Cast", ["rkeep_b"], "rkeep", to=F32)
    n("Cast", ["ckeep_b"], "ckeep", to=F32)
    n("Mul", ["K3raw", "rkeep"], "K3a")
    n("Mul", ["K3a", "ckeep"], "K3")         # [1,1,3,3] masked pattern
    n("ReduceSum", ["K3"], "npat", axes=[2, 3], keepdims=1)  # [1,1,1,1]
    # flatten K3 to [9]
    n("Reshape", ["K3", "sh9"], "k3flat")      # [9]
    init("sh9", np.array([9], np.int64), np.int64)

    # ===== 8 oriented kernels via gather permutation =====
    perm_arr = np.array(PERMS, np.int64).reshape(-1)  # [72]
    init("perm", perm_arr, I64)
    n("Gather", ["k3flat", "perm"], "ko_flat", axis=0)  # [72]
    # reshape to [8,1,3,3]
    n("Reshape", ["ko_flat", "sh8133"], "Kconv")
    init("sh8133", np.array([8, 1, 3, 3], np.int64), np.int64)

    # ===== mask3 (bbox-fill) for each orientation =====
    # rowhas3 = ReduceMax over cols -> [8,1,3,1]; colhas3 over rows -> [8,1,1,3]
    n("ReduceMax", ["Kconv"], "rh3", axes=[3], keepdims=1)   # [8,1,3,1]
    n("ReduceMax", ["Kconv"], "ch3", axes=[2], keepdims=1)   # [8,1,1,3]
    # bbox fill along axis: prefixOR AND suffixOR. With 3 elems, use triangular MatMul.
    # rowfill[i] = (exists j<=i with rh3) AND (exists j>=i) -> use cummax both directions.
    # cumulative-or via MatMul with lower/upper triangular {0,1}
    tril = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], np.float32)  # prefix (incl)
    triu = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], np.float32)  # suffix
    init("tril", tril, F32)
    init("triu", triu, F32)
    # rh3 is [8,1,3,1]; reshape to [8,3] for matmul over the 3-axis
    n("Reshape", ["rh3", "sh83"], "rh2")        # [8,3]
    init("sh83", np.array([8, 3], np.int64), np.int64)
    n("MatMul", ["rh2", "tril"], "rpre")        # [8,3] prefix sums (>0 => any earlier)
    n("MatMul", ["rh2", "triu"], "rsuf")        # suffix sums
    init("zero", np.array([0.0], np.float32).reshape(1, 1), F32)
    n("Greater", ["rpre", "zero"], "rpre_b")
    n("Greater", ["rsuf", "zero"], "rsuf_b")
    n("And", ["rpre_b", "rsuf_b"], "rfill_b")   # [8,3] bool
    n("Cast", ["rfill_b"], "rfill", to=F32)
    n("Reshape", ["rfill", "sh8131"], "rfill4")  # [8,1,3,1]
    init("sh8131", np.array([8, 1, 3, 1], np.int64), np.int64)
    n("Reshape", ["ch3", "sh83b"], "ch2")
    init("sh83b", np.array([8, 3], np.int64), np.int64)
    n("MatMul", ["ch2", "tril"], "cpre")
    n("MatMul", ["ch2", "triu"], "csuf")
    n("Greater", ["cpre", "zero"], "cpre_b")
    n("Greater", ["csuf", "zero"], "csuf_b")
    n("And", ["cpre_b", "csuf_b"], "cfill_b")
    n("Cast", ["cfill_b"], "cfill", to=F32)
    n("Reshape", ["cfill", "sh8113"], "cfill4")  # [8,1,1,3]
    init("sh8113", np.array([8, 1, 1, 3], np.int64), np.int64)
    n("Mul", ["rfill4", "cfill4"], "Mconv")      # [8,1,3,3] mask3 kernels

    # ===== COMBINED match kernel: fold the "no-extra-yellow" check into ONE Conv =====
    # combk = Ko*(1+B) - B*mask3 : a pattern-yellow contributes (1+B), an extra yellow inside
    # the oriented bbox contributes -B, outside-bbox 0.  match  iff  corr == npat  (all npat
    # pattern cells present AND zero extras).  This removes the separate `tot` Conv (one fewer
    # [1,8,25,25] fp16 plane).
    init("Bp1", np.array([1.0 + BVAL], np.float32).reshape(1, 1, 1, 1), F32)
    init("Bneg", np.array([BVAL], np.float32).reshape(1, 1, 1, 1), F32)
    n("Mul", ["Kconv", "Bp1"], "Kw")
    n("Mul", ["Mconv", "Bneg"], "Mw")
    n("Sub", ["Kw", "Mw"], "combk")              # [8,1,3,3]

    # ===== Y is already the 23x23 active grid; cast to fp16 for the match Conv =====
    n("Cast", ["Y"], "Yg", to=F16)                 # [1,1,23,23] fp16
    n("Cast", ["combk"], "combk16", to=F16)
    # Match WITHOUT a separate bool + fp16 plane: bias the Conv by -(npat-0.5) so its output is
    # already corr-(npat-0.5) (=+0.5 exactly at a match since corr<=npat is integer-valued, <=-0.5
    # otherwise), then ONE Relu yields the fp16 {0,0.5} indicator directly. This replaces the
    # corr(8.5KB)+Equal-bool(4.2KB)+Cast-fp16(8.5KB) trio with corr'(8.5KB)+Relu(8.5KB).
    # Conv accepts a runtime per-output-channel bias [8]; build it as (0.5 - npat) broadcast to 8.
    init("half", np.array([0.5], np.float32).reshape(1, 1, 1, 1), F32)
    n("Sub", ["half", "npat"], "negnpat")            # [1,1,1,1] = 0.5 - npat
    n("Reshape", ["negnpat", "scalar1"], "negnpat1")  # [1]
    init("ones8", np.ones((8,), np.float32), F32)
    n("Mul", ["negnpat1", "ones8"], "bias8_f")        # [8] broadcast
    n("Cast", ["bias8_f"], "bias8", to=F16)
    # SAME-pad Conv directly on the 23x23 crop (pads=[1,1,1,1]) avoids a separate padded Yp plane.
    n("Conv", ["Yg", "combk16", "bias8"], "corrm", pads=[1, 1, 1, 1])  # [1,8,23,23] fp16
    n("Relu", ["corrm"], "M")                          # fp16 {0, 0.5}; >0 iff match

    # spatially-flip Mconv [8,1,3,3] -> reshape to [1,8,3,3] sum-Conv weight (fp16).
    init("flip_s", np.array([2, 2], np.int64), np.int64)        # start last idx (size3)
    init("flip_e", np.array([-4, -4], np.int64), np.int64)      # to before 0
    init("flip_step", np.array([-1, -1], np.int64), np.int64)
    init("flip_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["Mconv", "flip_s", "flip_e", "flip_ax", "flip_step"], "Mconv_f")
    init("sh1833", np.array([1, 8, 3, 3], np.int64), np.int64)
    n("Reshape", ["Mconv_f", "sh1833"], "Mconv_flip")
    n("Cast", ["Mconv_flip"], "sumk16", to=F16)

    # ===== stamp each oriented content bbox at every match anchor, collapse, dilate by 1 =====
    # A forward grouped-SUM Conv collapses the 8 orientation channels into ONE [1,1,23,23] plane
    # (SUM is fine — we only test >0). It reproduces the old per-orientation ConvTranspose stamp by
    # spatially flipping the mask (forward Conv correlates with the un-flipped kernel) and SAME pad=1
    # so the stamp stays centred on the anchor. Then ONE 3x3 MaxPool dilates the bbox by 1 = the box.
    n("Conv", ["M", "sumk16"], "placed1", pads=[1, 1, 1, 1])  # [1,1,23,23] fp16
    n("MaxPool", ["placed1"], "boxsum", kernel_shape=[3, 3], pads=[1, 1, 1, 1],
      strides=[1, 1])                          # dilate by 1

    # ===== assemble output entirely at 23x23, then Pad-99 to 30x30 (uint8 Pad keeps off-grid off) =====
    init("zero16", np.array([0.0], np.float16).reshape(1, 1, 1, 1), F16)
    init("zero1111", np.array([0.0], np.float32).reshape(1, 1, 1, 1), F32)
    # box (bool) and yellow (bool) both at 23x23 — no full-30x30 fp16 plane needed.
    n("Greater", ["boxsum", "zero16"], "box23")   # [1,1,23,23] bool
    n("Greater", ["Y", "zero1111"], "yel23")      # [1,1,23,23] bool (Y is the 23x23 yellow slice)

    # Route the 10-ch output via ONE uint8 colour-index plane + a final Equal into the FREE output
    # (Equal & Where both run on uint8 under ORT_DISABLE_ALL). Priority yellow(4) > blue(1) > bg(0).
    init("c0u", np.array([0], np.uint8).reshape(1, 1, 1, 1), U8)
    init("c1u", np.array([1], np.uint8).reshape(1, 1, 1, 1), U8)
    init("c4u", np.array([4], np.uint8).reshape(1, 1, 1, 1), U8)
    n("Where", ["box23", "c1u", "c0u"], "idx_bb")     # blue or bg [1,1,23,23] uint8
    n("Where", ["yel23", "c4u", "idx_bb"], "colidx23")  # yellow overrides [1,1,23,23] uint8
    # Pad to 30x30 with sentinel 99 -> off-grid cells match no channel (required all-off one-hot).
    init("pad99", np.array([0, 0, 0, 0, 0, 0, H - G, H - G], np.int64), np.int64)
    init("cv99", np.array(99, np.uint8), U8)
    n("Pad", ["colidx23", "pad99", "cv99"], "colidx", mode="constant")  # [1,1,30,30] uint8
    arange_ch = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("arange_ch", arange_ch, U8)
    n("Equal", ["colidx", "arange_ch"], "output")       # [1,10,30,30] bool (FREE output)

    graph = helper.make_graph(nodes, "task191", [
        helper.make_tensor_value_info("input", F32, [1, 10, H, H])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, H, H])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
