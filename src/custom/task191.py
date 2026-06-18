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

Encoding (template-matching as a stacked Conv; verified 500/500 fresh, pts 13.77):
  - Y = input channel 4 (yellow presence); B = channel 1 (blue).
  - Extract the sprite: brow=(min blue row)+1, bcol=(min blue col)+1; tall/wide from the blue
    bbox extent.  K3raw = 3x3 yellow window at (brow,bcol); MASK rows>=tall, cols>=wide
    (the 3x3 window can otherwise capture a noise yellow just outside the t x w sprite) -> K3.
    npat = sum(K3) (= max(tall,wide)).
  - 8 oriented 3x3 kernels Ko = FIXED index-permutations of K3's 9 flat elements
    (rot90(K3,k) optionally transposed). mask3_o = bbox-fill of Ko (triangular-MatMul
    prefix/suffix-OR per axis, outer-product).  Stack both as Conv weights [8,1,3,3].
  - COMBINED match kernel folds the "no-extra-yellow" test into ONE Conv:
      combk = Ko*(1+B) - B*mask3   (pattern-yellow -> 1+B, extra-yellow-in-bbox -> -B)
      match M = Equal(Conv(Yp, combk), npat)   ([1,8,Hm,Hm] bool) -- corr==npat iff all npat
      pattern cells present AND zero extras.  This removes the separate `tot` Conv plane.
  - box: ConvTranspose(M, mask3 [8,1,3,3], group=8) stamps each oriented content bbox at its
    window; ReduceMax over the 8 orientations -> 1 plane; one 3x3 MaxPool dilates by 1.
  - Working planes (Yp/corr/M/placed) are fp16 (fp16-exact for these small integers); the Y
    conv canvas is cropped to the fixed 23x23 active grid first.
  - output: ch4 = yellow; ch1 = box AND NOT yellow; ch0 = ingrid AND NOT box AND NOT yellow.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

H = 30
G = 23  # fixed active grid size
PAD = 2  # all-sides pad
CROP0 = 2  # boxsum grid-coord offset (placed q-PAD)
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

    np_of = {F32: np.float32, F16: np.float16, BOOL: np.bool_, I64: np.int64}

    def init(name, arr, dtype):
        npd = np_of.get(dtype, dtype)
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=npd), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ===== yellow / blue planes =====
    init("c4_s", np.array([4], np.int64), np.int64)
    init("c4_e", np.array([5], np.int64), np.int64)
    init("ax1", np.array([1], np.int64), np.int64)
    n("Slice", ["input", "c4_s", "c4_e", "ax1"], "Y")        # [1,1,30,30] fp32

    init("c1_s", np.array([1], np.int64), np.int64)
    init("c1_e", np.array([2], np.int64), np.int64)
    n("Slice", ["input", "c1_s", "c1_e", "ax1"], "B")        # [1,1,30,30] blue fp32

    # ===== brow, bcol from blue bbox =====
    # rowhas[1,1,30,1], colhas[1,1,1,30]
    n("ReduceMax", ["B"], "rowhas", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["B"], "colhas", axes=[1, 2], keepdims=1)  # [1,1,1,30]
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
    n("Add", ["minr_s", "off012"], "ridx_f")   # [3]
    n("Add", ["minc_s", "off012"], "cidx_f")   # [3]
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

    # ===== crop Y to the fixed 23x23 active grid, pad, single Conv =====
    init("gs", np.array([0, 0], np.int64), np.int64)
    init("ge", np.array([G, G], np.int64), np.int64)
    init("gax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["Y", "gs", "ge", "gax"], "Yg")   # [1,1,23,23]
    init("padY", np.array([0, 0, PAD, PAD, 0, 0, PAD, PAD], np.int64), np.int64)
    n("Pad", ["Yg", "padY"], "Yp32")   # [1,1,23+2PAD,23+2PAD]
    n("Cast", ["Yp32"], "Yp", to=F16)
    n("Cast", ["combk"], "combk16", to=F16)
    n("Cast", ["Mconv"], "Mconv16", to=F16)
    n("Cast", ["npat"], "npat16", to=F16)
    n("Conv", ["Yp", "combk16"], "corr")   # [1,8,Hp-2,Hp-2] fp16
    n("Equal", ["corr", "npat16"], "Mb")   # [1,8,Hm,Hm] bool
    n("Cast", ["Mb"], "M", to=F16)

    # ===== stamp the (undilated) content bbox per orientation, collapse, dilate ONCE =====
    # ConvTranspose places mask3 (3x3) at each anchor -> [1,8,Hm+2,Hm+2]; ReduceMax over the
    # 8 orientations -> [1,1,Hm+2,Hm+2]; then a single 3x3 MaxPool dilates by 1 (the box border).
    n("ConvTranspose", ["M", "Mconv16"], "placed", group=8,
      kernel_shape=[3, 3], strides=[1, 1])     # [1,8,Hm+2,Hm+2] fp16
    n("ReduceMax", ["placed"], "placed1", axes=[1], keepdims=1)  # [1,1,Hm+2,Hm+2]
    n("MaxPool", ["placed1"], "boxsum", kernel_shape=[3, 3], pads=[1, 1, 1, 1],
      strides=[1, 1])                          # dilate by 1

    # boxsum coordinates: anchor a in [0..Hm-1] maps to padded-Y coord a, stamp top-left a-1.
    # ConvTranspose output index = a + (kernel index). With stamp top-left meaning offset 0..4,
    # and we want absolute padded coord = (a-1)+i. ConvTranspose gives out[a+i].
    # So out index o corresponds to padded coord o-1? We must align: padded coord p = o + ?.
    # We'll Slice the final box to recover the 30x30 grid below after determining offset.

    # ===== assemble output =====
    # box>0 indicator on the boxsum plane; need to crop to 30x30 grid coords.
    # We'll handle cropping with a Slice (offsets determined empirically; see CROP below).
    n("Greater", ["boxsum", "zero16"], "box_b")  # [1,1,Hb,Hb] bool
    init("zero16", np.array([0.0], np.float16).reshape(1, 1, 1, 1), F16)
    init("zero1111", np.array([0.0], np.float32).reshape(1, 1, 1, 1), F32)
    # crop box_b to the 23x23 grid then pad to 30x30 (off-grid box = 0)
    init("cs_s", np.array([CROP0, CROP0], np.int64), np.int64)
    init("cs_e", np.array([CROP0 + G, CROP0 + G], np.int64), np.int64)
    init("cs_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["box_b", "cs_s", "cs_e", "cs_ax"], "box23")  # [1,1,23,23] bool
    n("Cast", ["box23"], "box23f", to=F16)
    init("padbox", np.array([0, 0, 0, 0, 0, 0, H - G, H - G], np.int64), np.int64)
    n("Pad", ["box23f", "padbox"], "box30f")   # [1,1,30,30] fp16
    n("Greater", ["box30f", "zero16"], "box30")  # bool

    # yellow mask bool
    n("Greater", ["Y", "zero1111"], "yel_b")   # [1,1,30,30]
    # in-grid mask: off-grid cells are all-zero one-hot (no channel set)
    n("ReduceMax", ["input"], "ingrid_f", axes=[1], keepdims=1)  # [1,1,30,30]
    n("Greater", ["ingrid_f", "zero1111"], "ingrid")            # bool
    # box must be in-grid (matches only occur in-grid)
    n("And", ["box30", "ingrid"], "box_ig")
    # blue out = box AND NOT yellow
    n("Not", ["yel_b"], "nyel")
    n("And", ["box_ig", "nyel"], "blue_out")     # ch1
    # bg out = in-grid AND NOT box AND NOT yellow
    n("Not", ["box_ig"], "nbox")
    n("And", ["nbox", "nyel"], "bg_a")
    n("And", ["bg_a", "ingrid"], "bg_out")       # ch0

    # build 10-channel output: ch0=bg, ch1=blue, ch4=yellow, others 0
    # output = And(chsel[1,10,1,1], ...) approach: use Where chain via concat of bools.
    # Simpler: make each channel plane then Concat along channel axis -> [1,10,30,30] bool.
    init("false30", np.zeros((1, 1, H, H), np.bool_), BOOL)
    # ch2,3 = false; ch5..9 = false
    n("Concat", ["bg_out", "blue_out", "false30", "false30",
                 "yel_b", "false30", "false30", "false30", "false30", "false30"],
      "output", axis=1)

    graph = helper.make_graph(nodes, "task191", [
        helper.make_tensor_value_info("input", F32, [1, 10, H, H])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, H, H])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
