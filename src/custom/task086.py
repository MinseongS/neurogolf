"""task086 (ARC-AGI 3befdf3e) — concentric "flower" frame around each filled square.

Rule (from generator task_3befdf3e.py):
  An instance has 1-2 non-overlapping squares.  Each square has half-size L in {1,2}.
  In the INPUT a square is an (L+2)x(L+2) block: a 1-px border of color1 (c1) around an
  LxL inner of color0 (c0).  `colors=[c0,c1]` is GLOBAL per instance.  In the INPUT c1
  (border, (L+2)^2 - L^2 cells) outnumbers c0 (inner, L^2 cells) so:
     c1 = most-frequent non-bg color, c0 = the other non-bg color.

  OUTPUT per square (a fixed, L-parametric concentric stamp at the same location):
    SQ    = the (L+2)^2 solid block (= the input non-bg block).
    er    = erode3(SQ)            = the LxL inner.
    out0  = SQ - er               = the (L+2)^2 PERIMETER ring         -> color c0
    arms  = cross arms of length L extending L cells outward from SQ along the block's
            row-span (vertical arms, cols = block cols) and col-span (horizontal arms),
            NO corners.  = (dilate_v(SQ,2L+1) | dilate_h(SQ,2L+1)) minus the block.
    out1  = er | arms             = inner LxL + cross arms              -> color c1
    everything else unchanged (background).

  Variable L is handled WITHOUT a scalar: split SQ into L=1 / L=2 cells and dilate each
  with its own kernel.  A cell is in an L=2 (4x4 solid) block via a 4x4-sum==16 top-left
  detector dilated back over its 4x4 footprint (in4).  L=1 cells are the rest.
    cross = dil_v(SQ1,3) | dil_h(SQ1,3) | dil_v(SQ2,5) | dil_h(SQ2,5)
    out1  = cross AND NOT out0          (out0 already carved out the perimeter ring)

  Verified 500/500 exact in numpy and ISOLATED fresh 200/200 against the generator.

Encoding (route the 10-ch expansion into the FREE output; never materialize [1,10,H,W]):
  Work on a W=12 canvas (grid sizes are 10-12; squares always sit in-grid).  All masks are
  [1,1,W,W] fp16 (288B) Conv responses; the only fp32 tensor is the [1,10] channel-count
  vector used to recover the two colour one-hots.  Output assembled as
    output = Where(out1_30, c1_oh, Where(out0_30, c0_oh, input))   (FREE [1,10,30,30]).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

W = 12  # active canvas (grid is always 10-12; squares + arms stay in-grid)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- 1-D in-grid occupancy profiles, sliced to the WxW window -----------
    #   rowany[r]=1 iff some channel set in row r (in-grid bg rows have ch0=1).
    n("ReduceMax", ["input"], "rowany_f", axes=[1, 3], keepdims=1)  # [1,1,30,1] f32
    n("ReduceMax", ["input"], "colany_f", axes=[1, 2], keepdims=1)  # [1,1,1,30] f32
    init("ZH", np.array(0.0, np.float16), np.float16)
    n("Cast", ["rowany_f"], "rowany", to=F16)
    n("Cast", ["colany_f"], "colany", to=F16)
    # slice the 1-D profiles to the active window (W rows / W cols).
    init("rw_s", np.array([0], np.int64), np.int64)
    init("rw_e", np.array([W], np.int64), np.int64)
    init("rw_ax", np.array([2], np.int64), np.int64)
    init("cw_ax", np.array([3], np.int64), np.int64)
    n("Slice", ["rowany", "rw_s", "rw_e", "rw_ax"], "rowany_w")   # [1,1,W,1] f16
    n("Slice", ["colany", "rw_s", "rw_e", "cw_ax"], "colany_w")   # [1,1,1,W] f16
    n("Greater", ["rowany_w", "ZH"], "rowin_w")                  # [1,1,W,1] bool
    n("Greater", ["colany_w", "ZH"], "colin_w")                  # [1,1,1,W] bool
    n("And", ["rowin_w", "colin_w"], "ingrid_wb")                # [1,1,W,W] bool (broadcast)
    n("Cast", ["ingrid_wb"], "ingrid_w", to=F16)                 # [1,1,W,W] f16

    # ---- SQ (colored cells) on the WxW active window ------------------------
    # ch0_w = bg channel slice; colored = (1 - ch0) AND in-grid (off-grid has
    # ch0=0 too, so AND ingrid removes the spurious off-grid band).
    init("c0_s", np.array([0, 0, 0], np.int64), np.int64)
    init("c0_e", np.array([1, W, W], np.int64), np.int64)
    init("w_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "c0_s", "c0_e", "w_ax"], "ch0_w")       # [1,1,W,W] f32
    n("Cast", ["ch0_w"], "ch0_wh", to=F16)
    init("ONEH", np.array(1.0, np.float16), np.float16)
    n("Sub", ["ONEH", "ch0_wh"], "notbg_w")                      # [1,1,W,W] f16 {0,1}
    n("Mul", ["notbg_w", "ingrid_w"], "SQ")                      # [1,1,W,W] f16 {0,1}

    # ---- erode3(SQ): 3x3 sum == 9 -> inner LxL ------------------------------
    init("k3", np.ones((1, 1, 3, 3), np.float16), np.float16)
    n("Conv", ["SQ", "k3"], "s3", pads=[1, 1, 1, 1])             # [1,1,W,W] f16
    init("C9", np.array(9.0, np.float16), np.float16)
    n("Equal", ["s3", "C9"], "er_b")                            # bool inner
    n("Cast", ["er_b"], "er", to=F16)                           # [1,1,W,W] f16

    # ---- out0 = SQ - er  (perimeter ring -> colour c0) ----------------------
    n("Sub", ["SQ", "er"], "out0f")                             # f16 {0,1}
    n("Greater", ["out0f", "ZH"], "out0_b")                     # bool perimeter

    # ---- in4(SQ): cells in a 4x4 solid block (L=2) --------------------------
    # tl: 4x4 window summed with top-left anchored at (i,j) (pad bottom/right 3),
    #     == 16  -> top-left of a 4x4 solid.
    init("k4", np.ones((1, 1, 4, 4), np.float16), np.float16)
    n("Conv", ["SQ", "k4"], "w4tl", pads=[0, 0, 3, 3])          # [1,1,W,W] f16
    init("C16", np.array(16.0, np.float16), np.float16)
    n("Equal", ["w4tl", "C16"], "tl_b")
    n("Cast", ["tl_b"], "tl", to=F16)                           # [1,1,W,W] f16
    # cover: dilate tl back over its 4x4 footprint (pad top/left 3) > 0.
    n("Conv", ["tl", "k4"], "cov", pads=[3, 3, 0, 0])           # [1,1,W,W] f16
    n("Greater", ["cov", "ZH"], "L2_b")                         # bool: cell in L=2 block
    n("Cast", ["L2_b"], "L2", to=F16)

    # ---- split SQ into L=2 (SQ2) and L=1 (SQ1) cells ------------------------
    n("Mul", ["SQ", "L2"], "SQ2")                               # f16
    n("Sub", ["SQ", "SQ2"], "SQ1")                              # f16  (= SQ AND NOT L2)

    # ---- cross = dilate each part with its own kernel -----------------------
    # vertical kernel (2L+1)x1, horizontal 1x(2L+1).  SAME padding = (n-1)/2.
    init("kv3", np.ones((1, 1, 3, 1), np.float16), np.float16)
    init("kh3", np.ones((1, 1, 1, 3), np.float16), np.float16)
    init("kv5", np.ones((1, 1, 5, 1), np.float16), np.float16)
    init("kh5", np.ones((1, 1, 1, 5), np.float16), np.float16)
    n("Conv", ["SQ1", "kv3"], "dv1", pads=[1, 0, 1, 0])
    n("Conv", ["SQ1", "kh3"], "dh1", pads=[0, 1, 0, 1])
    n("Conv", ["SQ2", "kv5"], "dv2", pads=[2, 0, 2, 0])
    n("Conv", ["SQ2", "kh5"], "dh2", pads=[0, 2, 0, 2])
    n("Sum", ["dv1", "dh1", "dv2", "dh2"], "crossf")           # f16, >0 where any arm/block
    n("Greater", ["crossf", "ZH"], "cross_b")                  # bool

    # ---- out1 = cross AND NOT out0  (inner LxL + arms; perimeter excluded) ---
    n("Not", ["out0_b"], "not_out0")
    n("And", ["cross_b", "not_out0"], "out1_b")                # bool

    # ---- carry both masks to fp16 0/1 on the WxW canvas ---------------------
    n("Cast", ["out0_b"], "out0_h", to=F16)                    # [1,1,W,W] f16 {0,1}
    n("Cast", ["out1_b"], "out1_h", to=F16)                    # [1,1,W,W] f16 {0,1}

    # ---- recover SCALAR colour indices c0idx / c1idx from channel counts ----
    # cnt[k] = #cells of colour k (ch0 huge).  Mask ch0 out, then:
    #   c1 = argmax of nonzero counts (most frequent colour = border)
    #   c0 = the other nonzero count channel.
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    notbg = np.ones((1, 10, 1, 1), np.float32)
    notbg[0, 0, 0, 0] = 0.0
    init("notbg", notbg, np.float32)
    n("Mul", ["cnt", "notbg"], "cnt_fg")                       # [1,10,1,1] f32
    n("ReduceMax", ["cnt_fg"], "cmax", axes=[1], keepdims=1)   # [1,1,1,1] f32
    n("Equal", ["cnt_fg", "cmax"], "c1_oh")                    # [1,10,1,1] bool
    init("ZF", np.array(0.0, np.float32), np.float32)
    n("Greater", ["cnt_fg", "ZF"], "fg_oh")                    # bool nonzero fg channels
    n("Not", ["c1_oh"], "not_c1")
    n("And", ["fg_oh", "not_c1"], "c0_oh")                     # bool c0 channel
    # scalar index = sum_k k * onehot_k
    ar10 = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("ar10", ar10, np.float32)
    n("Cast", ["c1_oh"], "c1_ohf", to=F32)
    n("Cast", ["c0_oh"], "c0_ohf", to=F32)
    n("Mul", ["c1_ohf", "ar10"], "c1_w")
    n("Mul", ["c0_ohf", "ar10"], "c0_w")
    n("ReduceSum", ["c1_w"], "c1idx_f", axes=[1], keepdims=1)  # [1,1,1,1] f32
    n("ReduceSum", ["c0_w"], "c0idx_f", axes=[1], keepdims=1)  # [1,1,1,1] f32
    n("Cast", ["c1idx_f"], "c1idx", to=F16)                    # [1,1,1,1] f16
    n("Cast", ["c0idx_f"], "c0idx", to=F16)                    # [1,1,1,1] f16

    # ---- colour-index plane Lcol on WxW (0=bg, c0 on out0, c1 on out1) ------
    # out0 / out1 disjoint, so a weighted sum carries each region's colour.
    n("Mul", ["out0_h", "c0idx"], "Lc0")                       # [1,1,W,W] f16
    n("Mul", ["out1_h", "c1idx"], "Lc1")                       # [1,1,W,W] f16
    n("Add", ["Lc0", "Lc1"], "Lcol")                           # [1,1,W,W] f16
    # off-grid cells INSIDE the WxW window -> sentinel -1 (so Equal misses) ---
    init("NEG1", np.array(-1.0, np.float16), np.float16)
    n("Where", ["ingrid_wb", "Lcol", "NEG1"], "Lcol_m")        # [1,1,W,W] f16
    # pad up to 30x30 with the SAME -1 sentinel (the [W:30] band is off-grid) -
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    n("Pad", ["Lcol_m", "pads", "NEG1"], "L", mode="constant")  # [1,1,30,30] f16

    # ---- output = Equal(L, arange[1,10,1,1]) : FREE bool [1,10,30,30] --------
    ar10h = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("ar10h", ar10h, np.float16)
    n("Equal", ["L", "ar10h"], "output")                       # bool one-hot

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task086", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
