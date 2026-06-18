"""task201 (ARC-AGI 846bdb03) — reassemble a bordered box with two de-flipped sprites.

Rule (from generator task_846bdb03.py, square grid, here size=13):
  INPUT contains two disjoint objects on a black canvas:
    1. a hollow BOX of size h x w (h=max(rows)+3, w=2*(max(cols)+2)) with YELLOW(4)
       corners, its left column = colors[0], its right column = colors[1].
    2. a SPRITE CLUSTER: two conway sprites side-by-side (sprite idx0 drawn in
       colors[0], idx1 in colors[1]); the whole cluster is horizontally FLIPPED
       inside its own (h-2)x(w-2) bounding box iff flip==1.
  OUTPUT is exactly h x w: the box's border (corners yellow, left col colors[0],
  right col colors[1], top/bottom rows otherwise background) with its interior
  (rows 1..h-2, cols 1..w-2) filled by the sprite cluster, DE-FLIPPED so colors[0]'s
  sprite is on the left and colors[1]'s on the right.

  Verified exactly (500/500 numpy):
    - YELLOW bounding box gives (r0,c0) and the output size (h,w) exactly (the 4
      yellow pixels ARE the corners).
    - colors[0]=colf[r0+1,c0], colors[1]=colf[r0+1,c0+w-1].
    - sprite cluster = everything nonzero OUTSIDE the box bbox; its bbox is exactly
      (h-2)x(w-2); top-left (sr0,sc0).
    - flip iff (mean col of colors[0] pixels in the cluster) > (mean col of colors[1]
      pixels) -> colors[0] sits on the RIGHT, so the cluster is mirrored.
    - output interior[i-1,j-1] = colf[sr0+(i-1), sc0 + (flip? w-2-j : j-1)].

Encoding (all working planes tiny; final 10-ch expansion into FREE output):
  - colf = sum_k k*input_k via one Conv[1,10,1,1] -> [1,1,30,30] f32, sliced to 13x13.
  - all geometry scalars recovered from 1-D ReduceMax profiles + ArgMax / ramp-dot.
  - build a FIXED 6x8 colour-index plane OUT by Gather-ing colf for the interior and
    overlaying the border + an off-output sentinel(99) via a tiny Where chain.
  - Pad OUT (sentinel 99) to 30x30 uint8, output = Equal(OUT_u8, arange[0..9]) -> BOOL.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

G = 13          # active input grid (square)
OH, OW = 8, 8   # max output height / width (scored set tops at 7x8; margin for safety)


def build(task):
    inits, nodes = [], []

    NP = {F32: np.float32, F16: np.float16, I64: np.int64, U8: np.uint8, BOOL: np.bool_}

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=NP[dtype]), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, list(ins), [out], **attrs))
        return out

    # ---------- colour-index plane colf (sum_k k*input_k), cropped to GxG -------
    init("colw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), F32)
    n("Conv", ["input", "colw"], "colf30")                      # [1,1,30,30] f32
    init("cs", np.array([0, 0], np.int64), I64)
    init("ce", np.array([G, G], np.int64), I64)
    init("cax", np.array([2, 3], np.int64), I64)
    n("Slice", ["colf30", "cs", "ce", "cax"], "colf")           # [1,1,G,G] f32
    n("Cast", ["colf"], "colf16", to=F16)                       # [1,1,G,G] f16 (0..9 exact)

    # ---------- yellow plane + 1-D profiles -------------------------------------
    init("ys", np.array([0, 4, 0, 0], np.int64), I64)
    init("ye", np.array([1, 5, G, G], np.int64), I64)
    init("yax", np.array([0, 1, 2, 3], np.int64), I64)
    n("Slice", ["input", "ys", "ye", "yax"], "Y")               # [1,1,G,G] f32
    n("ReduceMax", ["Y"], "Yrow", axes=[3], keepdims=0)         # [1,1,G]
    n("ReduceMax", ["Y"], "Ycol", axes=[2], keepdims=0)         # [1,1,G]

    # row/col ramps (length G) as f32
    init("rampG", np.arange(G, dtype=np.float32).reshape(1, 1, G), F32)
    init("BIG", np.array(1000.0, np.float32), F32)

    # r0 = min row with yellow ; r1 = max row with yellow
    #   min: argmin of (ramp + (1-Yrow)*BIG) -> but easier: r0 via masked-min.
    #   present rows: Yrow==1. value = ramp where present else +BIG -> ReduceMin -> r0
    #                 value = ramp where present else -BIG -> ReduceMax -> r1
    def bbox(profile, tag):
        # profile [1,1,G] in {0,1}
        n("Sub", ["one1", profile], f"np_{tag}")                # 1-present
        n("Mul", [f"np_{tag}", "BIG"], f"pen_{tag}")            # BIG where absent
        n("Add", ["rampG", f"pen_{tag}"], f"lo_{tag}")          # ramp + penalty
        n("ReduceMin", [f"lo_{tag}"], f"mn_{tag}", axes=[2], keepdims=0)   # [1,1]
        n("Mul", [f"pen_{tag}", "negone"], f"penn_{tag}")       # -BIG where absent
        n("Add", ["rampG", f"penn_{tag}"], f"hi_{tag}")
        n("ReduceMax", [f"hi_{tag}"], f"mx_{tag}", axes=[2], keepdims=0)   # [1,1]
        return f"mn_{tag}", f"mx_{tag}"

    init("one1", np.array(1.0, np.float32).reshape(1, 1, 1), F32)
    init("negone", np.array(-1.0, np.float32), F32)
    r0, r1 = bbox("Yrow", "yr")          # [1,1]
    c0, c1 = bbox("Ycol", "yc")

    # h = r1-r0+1 ; w = c1-c0+1   (scalars [1,1])
    init("one11", np.array(1.0, np.float32).reshape(1, 1), F32)
    n("Sub", [r1, r0], "hm1")            # h-1
    n("Add", ["hm1", "one11"], "hh")     # h
    n("Sub", [c1, c0], "wm1")            # w-1
    n("Add", ["wm1", "one11"], "ww")     # w

    # ---------- sprite cluster: nonzero outside the box bbox --------------------
    # occupancy occ = (colf>0)  [1,1,G,G]
    init("z0", np.array(0.0, np.float16), F16)
    n("Greater", ["colf16", "z0"], "occ_b")                     # bool [1,1,G,G]
    # box-region mask: rows in [r0,r1] AND cols in [c0,c1]
    init("rampGr", np.arange(G, dtype=np.float32).reshape(1, 1, G, 1), F32)
    init("rampGc", np.arange(G, dtype=np.float32).reshape(1, 1, 1, G), F32)
    # rin = (r>=r0)&(r<=r1)
    n("Reshape", [r0, "to_r"], "r0r")
    n("Reshape", [r1, "to_r"], "r1r")
    n("Reshape", [c0, "to_c"], "c0c")
    n("Reshape", [c1, "to_c"], "c1c")
    init("to_r", np.array([1, 1, 1, 1], np.int64), I64)
    init("to_c", np.array([1, 1, 1, 1], np.int64), I64)
    n("Not", [n("Less", ["rampGr", "r0r"], "lt_r0")], "ge_r0")
    n("Not", [n("Less", ["r1r", "rampGr"], "gt_r1")], "le_r1")
    n("And", ["ge_r0", "le_r1"], "rin")                         # [1,1,G,1]
    n("Not", [n("Less", ["rampGc", "c0c"], "lt_c0")], "ge_c0")
    n("Not", [n("Less", ["c1c", "rampGc"], "gt_c1")], "le_c1")
    n("And", ["ge_c0", "le_c1"], "cin")                         # [1,1,1,G]
    n("And", ["rin", "cin"], "boxmask_b")                       # [1,1,G,G]
    # sprite occupancy = occ_b AND NOT boxmask
    n("Not", ["boxmask_b"], "nbox")
    n("And", ["occ_b", "nbox"], "sp_b")                         # [1,1,G,G] bool
    n("Cast", ["sp_b"], "sp", to=F32)
    n("ReduceMax", ["sp"], "sprow", axes=[3], keepdims=0)       # [1,1,G]
    n("ReduceMax", ["sp"], "spcol", axes=[2], keepdims=0)       # [1,1,G]
    sr0, sr1 = bbox("sprow", "sr")
    sc0, sc1 = bbox("spcol", "sc")

    # ---------- colours: colf[r0+1, c0] and colf[r0+1, c1] ----------------------
    # gather row (r0+1) from colf -> [1,1,1,G]; then gather col c0 / c1
    n("Cast", [n("Add", [r0, "one11"], "r0p1")], "r0p1_i", to=I64)   # [1,1] int
    n("Reshape", ["r0p1_i", "to1"], "r0p1_s")                       # scalar [1]
    init("to1", np.array([1], np.int64), I64)
    n("Squeeze", ["r0p1_s"], "r0p1_sc", axes=[0])                   # scalar
    n("Gather", ["colf16", "r0p1_sc"], "rowL", axis=2)             # [1,1,G] f16
    n("Cast", [c0], "c0_i", to=I64)
    n("Reshape", ["c0_i", "to1"], "c0_s")
    n("Squeeze", ["c0_s"], "c0_sc", axes=[0])
    n("Cast", [c1], "c1_i", to=I64)
    n("Reshape", ["c1_i", "to1"], "c1_s")
    n("Squeeze", ["c1_s"], "c1_sc", axes=[0])
    n("Gather", ["rowL", "c0_sc"], "col0_v", axis=2)              # [1,1] colors[0]
    n("Gather", ["rowL", "c1_sc"], "col1_v", axis=2)              # [1,1] colors[1]

    # ---------- flip: mean col of color0 px > mean col of color1 px -------------
    # sprite-region value plane masked (fp16 — values/indices well below fp16 limit)
    n("Cast", ["sp_b"], "sp16", to=F16)
    n("Mul", ["colf16", "sp16"], "spv")                          # [1,1,G,G] f16
    # is_col0 / is_col1 over sprite region
    n("Cast", [n("Reshape", ["col0_v", "to_c"], "col0_r32")], "col0_r", to=F16)  # [1,1,1,1] f16
    n("Cast", [n("Reshape", ["col1_v", "to_c"], "col1_r32")], "col1_r", to=F16)
    n("Equal", ["spv", "col0_r"], "is0_b")                       # bg(0) cells: spv=0, col!=0 -> safe
    n("And", ["is0_b", "sp_b"], "m0_b")                          # restrict to sprite region
    n("Equal", ["spv", "col1_r"], "is1_b")
    n("And", ["is1_b", "sp_b"], "m1_b")
    n("Cast", ["m0_b"], "m0", to=F16)
    n("Cast", ["m1_b"], "m1", to=F16)
    # mean col = sum(col*mask)/sum(mask)   (fp16 ramp, sums < few hundred -> exact)
    init("rampGc16", np.arange(G, dtype=np.float16).reshape(1, 1, 1, G), F16)
    n("Mul", ["m0", "rampGc16"], "m0c")
    n("ReduceSum", ["m0c"], "s0c", axes=[2, 3], keepdims=0)      # [1,1] f16
    n("ReduceSum", ["m0"], "s0", axes=[2, 3], keepdims=0)
    n("Mul", ["m1", "rampGc16"], "m1c")
    n("ReduceSum", ["m1c"], "s1c", axes=[2, 3], keepdims=0)
    n("ReduceSum", ["m1"], "s1", axes=[2, 3], keepdims=0)
    # flip iff s0c/s0 > s1c/s1  <=> s0c*s1 > s1c*s0
    n("Mul", ["s0c", "s1"], "lhs")
    n("Mul", ["s1c", "s0"], "rhs")
    n("Greater", ["lhs", "rhs"], "flip_b")                       # [1,1] bool

    # ---------- build output row/col source-index vectors -----------------------
    # output rows i=0..OH-1 ; interior src_row = sr0 + (i-1) (clamped 0..G-1)
    init("rampOH", np.arange(OH, dtype=np.float32).reshape(OH), F32)   # [OH]
    init("rampOW", np.arange(OW, dtype=np.float32).reshape(OW), F32)   # [OW]
    n("Reshape", [sr0, "to1"], "sr0_1")
    n("Reshape", [sc0, "to1"], "sc0_1")
    n("Reshape", ["ww", "to1"], "ww_1")
    # src_row = sr0 + (i-1)
    init("oneOH", np.array(1.0, np.float32).reshape(1), F32)
    n("Sub", ["rampOH", "oneOH"], "im1")                         # i-1  [OH]
    n("Add", ["im1", "sr0_1"], "srcrow_f")                       # [OH]
    # clamp 0..G-1
    init("z0v", np.array(0.0, np.float32).reshape(1), F32)
    init("gm1v", np.array(float(G - 1), np.float32).reshape(1), F32)
    n("Max", ["srcrow_f", "z0v"], "srcrow_a")
    n("Min", ["srcrow_a", "gm1v"], "srcrow_c")
    n("Cast", ["srcrow_c"], "srcrow_i", to=I64)                  # [OH]

    # src_col: no-flip = sc0 + (j-1) ; flip = sc0 + (w-2-j)
    n("Sub", ["rampOW", "oneOH"], "jm1")                         # j-1 [OW]
    n("Add", ["jm1", "sc0_1"], "srccol_nf")                      # [OW]
    # flip: w-2-j + sc0
    init("two1", np.array(2.0, np.float32).reshape(1), F32)
    n("Sub", ["ww_1", "two1"], "wm2")                            # w-2
    n("Sub", ["wm2", "rampOW"], "wm2j")                          # w-2-j  [OW]
    n("Add", ["wm2j", "sc0_1"], "srccol_f")                      # [OW]
    # select by flip
    n("Reshape", ["flip_b", "to1"], "flip_1")                    # [1]
    n("Where", ["flip_1", "srccol_f", "srccol_nf"], "srccol_sel")  # [OW]
    n("Max", ["srccol_sel", "z0v"], "srccol_a")
    n("Min", ["srccol_a", "gm1v"], "srccol_c")
    n("Cast", ["srccol_c"], "srccol_i", to=I64)                  # [OW]

    # ---------- gather interior from colf16 (values 0..9 exact in f16) ----------
    n("Gather", ["colf16", "srcrow_i"], "gr", axis=2)            # [1,1,OH,G] f16
    n("Gather", ["gr", "srccol_i"], "interior", axis=3)          # [1,1,OH,OW] f16

    # ---------- overlay border + sentinel on a fixed OH x OW grid ----------------
    rOH = init("rOH", np.arange(OH, dtype=np.float32).reshape(1, 1, OH, 1), F32)
    rOW = init("rOW", np.arange(OW, dtype=np.float32).reshape(1, 1, 1, OW), F32)
    n("Reshape", ["hh", "to_r"], "hh4")          # [1,1,1,1]
    n("Reshape", ["ww", "to_r"], "ww4")
    n("Reshape", ["hm1", "to_r"], "hm14")        # h-1
    n("Reshape", ["wm1", "to_r"], "wm14")        # w-1
    n("Cast", [n("Reshape", ["col0_v", "to_r"], "c0val32")], "c0val", to=F16)
    n("Cast", [n("Reshape", ["col1_v", "to_r"], "c1val32")], "c1val", to=F16)

    # masks (bool) over OH x OW
    n("Less", [rOH, "hh4"], "i_lt_h")            # i<h  [1,1,OH,1]
    n("Less", [rOW, "ww4"], "j_lt_w")            # j<w  [1,1,1,OW]
    n("And", ["i_lt_h", "j_lt_w"], "in_out")     # [1,1,OH,OW]
    init("z04", np.array(0.0, np.float32).reshape(1, 1, 1, 1), F32)
    n("Greater", [rOH, "z04"], "i_gt0")          # i>0
    n("Greater", [rOW, "z04"], "j_gt0")          # j>0
    n("Less", [rOH, "hm14"], "i_lt_hm1")         # i<h-1
    n("Less", [rOW, "wm14"], "j_lt_wm1")         # j<w-1
    n("Equal", [rOH, "z04"], "i_eq0")
    n("Equal", [rOW, "z04"], "j_eq0")
    n("Equal", [rOH, "hm14"], "i_eqhm1")
    n("Equal", [rOW, "wm14"], "j_eqwm1")
    # interior region: i_gt0 & i_lt_hm1 & j_gt0 & j_lt_wm1
    n("And", ["i_gt0", "i_lt_hm1"], "i_inner")
    n("And", ["j_gt0", "j_lt_wm1"], "j_inner")
    n("And", ["i_inner", "j_inner"], "inner_b")  # [1,1,OH,OW]
    # left edge: j==0 & i_inner
    n("And", ["j_eq0", "i_inner"], "left_b")
    # right edge: j==w-1 & i_inner
    n("And", ["j_eqwm1", "i_inner"], "right_b")
    # corner: (i==0 | i==h-1) & (j==0 | j==w-1)
    n("Or", ["i_eq0", "i_eqhm1"], "i_edge")
    n("Or", ["j_eq0", "j_eqwm1"], "j_edge")
    n("And", ["i_edge", "j_edge"], "corner_b")

    # value chain: start background 0, then interior, left, right, corner
    init("BG", np.array(0.0, np.float16).reshape(1, 1, 1, 1), F16)
    init("YEL", np.array(4.0, np.float16).reshape(1, 1, 1, 1), F16)
    init("SENT99", np.array(99.0, np.float16).reshape(1, 1, 1, 1), F16)
    n("Where", ["inner_b", "interior", "BG"], "v1")
    n("Where", ["left_b", "c0val", "v1"], "v2")
    n("Where", ["right_b", "c1val", "v2"], "v3")
    n("Where", ["corner_b", "YEL", "v3"], "v4")
    # off-output -> sentinel 99
    n("Where", ["in_out", "v4", "SENT99"], "OUTv")              # [1,1,OH,OW] f32
    n("Cast", ["OUTv"], "OUT_u8", to=U8)

    # ---------- pad to 30x30 (sentinel 99) + Equal-expand into FREE output -------
    init("Lpads", np.array([0, 0, 0, 0, 0, 0, 30 - OH, 30 - OW], np.int64), I64)
    init("SENTU", np.array(99, np.uint8), U8)
    n("Pad", ["OUT_u8", "Lpads", "SENTU"], "OUT30", mode="constant")  # [1,1,30,30] u8
    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), U8)
    n("Equal", ["OUT30", "arange"], "output")                   # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task201", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
