"""task364 (ARC-AGI e509e548) — recolour each green L/U/H box by its SHAPE.

Rule (from generator task_e509e548.py):
  1+max(H,W)//3 non-overlapping boxes (gap>=1) are drawn in GREEN (colour 3) on a
  black canvas; grid is H in [10,20], W = H+[-2,2] (so <= 20 rows x 22 cols).  Each box
  is one of three sprite skeletons (in any of 8 dihedral orientations via transpose+gravity):
    - "el"    (L-shape): one full column + one full perpendicular row  -> OUTPUT colour 1 (blue)
    - "aitch" (H-shape): full left col + half right col + a middle cross-row -> OUTPUT colour 2 (red)
    - "you"   (U-shape): two full parallel cols + one connecting row   -> OUTPUT colour 6 (pink)
  Output recolours every green pixel by its box's shape class; background stays 0.

  CLOSED-FORM DISCRIMINATOR (verified 0/500 fresh, NO unique-label flood / ScatterND):
  compute 4-neighbour degree on the green mask, then THREE local seeds:
    J    = deg>=3                       (a T-junction)   -> only H has one
    vend = deg==1 and the lone nbr is VERTICAL  (above/below)
    hend = deg==1 and the lone nbr is HORIZONTAL (left/right)
  Flood each seed (MAX-pool, re-gated to the mask, 8 iters >= max box graph-diameter 8;
  kernel=3 only, since the inter-box gap can be 1 -> a bigger kernel would leak) so every
  pixel of a box carries its box's J/V/H presence flags.  Classify:
    isH = Jf>0
    isU = (not H) and (Vf>0) XOR (Hf>0)     # U's two tips point the SAME way -> only one type
    isL = (not H) and (Vf>0) AND (Hf>0)     # L's two tips are perpendicular -> both types
  (L and U both have exactly 2 endpoints & no junction; the XOR-vs-AND of endpoint
  orientation is the unique separator — verified clean across 1700 boxes.)

  Encoding: slice input to the 20x22 max-grid box (every plane measured there, not 30x30),
  flood in fp16, build colour-index L in fp16, route the 10-ch expansion into the FREE
  bool output via Equal(L_u8, arange).  No int64, no ScatterND, no histogram.
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

GH, GW = 20, 22          # max possible grid (H<=20, W<=22)
ITERS = 8                # >= max box graph-diameter (h-1)+(w-1) <= 8


def build(task):
    inits, nodes, vinfo = [], [], []

    NP = {I64: np.int64, F16: np.float16, F32: np.float32, U8: np.uint8, BOOL: np.bool_}

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=NP[dtype]), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfo.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # ---- slice input to the active 20x22 region -----------------------------
    init("s_g", np.array([0, 3, 0, 0], np.int64), I64)     # green channel (3)
    init("e_g", np.array([1, 4, GH, GW], np.int64), I64)
    init("s_b", np.array([0, 0, 0, 0], np.int64), I64)     # bg channel (0)
    init("e_b", np.array([1, 1, GH, GW], np.int64), I64)
    init("ax4", np.array([0, 1, 2, 3], np.int64), I64)
    n("Slice", ["input", "s_g", "e_g", "ax4"], "green_f32"); vi("green_f32", F32, [1, 1, GH, GW])
    n("Slice", ["input", "s_b", "e_b", "ax4"], "bg_f32");    vi("bg_f32", F32, [1, 1, GH, GW])
    n("Cast", ["green_f32"], "m", to=F16); vi("m", F16, [1, 1, GH, GW])  # mask {0,1} f16

    # ---- 4-neighbour planes via Pad+Slice (zero outside) --------------------
    # vert = up-nbr + down-nbr, horiz = left-nbr + right-nbr, via two 3x3 Convs
    # (SAME pad).  deg = (vert+horiz)*mask.
    init("ZF16", np.array(0.0, np.float16), F16)
    Kv = np.zeros((1, 1, 3, 3), np.float16); Kv[0, 0, 0, 1] = 1; Kv[0, 0, 2, 1] = 1
    Kh = np.zeros((1, 1, 3, 3), np.float16); Kh[0, 0, 1, 0] = 1; Kh[0, 0, 1, 2] = 1
    init("Kv", Kv, F16)
    init("Kh", Kh, F16)
    n("Conv", ["m", "Kv"], "vert", pads=[1, 1, 1, 1]); vi("vert", F16, [1, 1, GH, GW])
    n("Conv", ["m", "Kh"], "horiz", pads=[1, 1, 1, 1]); vi("horiz", F16, [1, 1, GH, GW])
    n("Add", ["vert", "horiz"], "deg_all"); vi("deg_all", F16, [1, 1, GH, GW])
    n("Mul", ["deg_all", "m"], "deg"); vi("deg", F16, [1, 1, GH, GW])

    # ---- the three seeds ----------------------------------------------------
    init("F1", np.array(1.0, np.float16), F16)
    init("F2", np.array(2.0, np.float16), F16)
    mb = n("Cast", ["m"], "mbool", to=BOOL); vi("mbool", BOOL, [1, 1, GH, GW])

    # J = deg>=3  (==  deg>2)
    n("Greater", ["deg", "F2"], "J_b"); vi("J_b", BOOL, [1, 1, GH, GW])
    # deg==1
    n("Equal", ["deg", "F1"], "deg1_b"); vi("deg1_b", BOOL, [1, 1, GH, GW])
    # lone neighbour is vertical: vert==1 (and deg==1)
    n("Equal", ["vert", "F1"], "vert1_b"); vi("vert1_b", BOOL, [1, 1, GH, GW])
    n("Equal", ["horiz", "F1"], "horiz1_b"); vi("horiz1_b", BOOL, [1, 1, GH, GW])
    n("And", ["deg1_b", "vert1_b"], "vend_b"); vi("vend_b", BOOL, [1, 1, GH, GW])
    n("And", ["deg1_b", "horiz1_b"], "hend_b"); vi("hend_b", BOOL, [1, 1, GH, GW])

    # cast seeds to fp16 for MaxPool flood
    n("Cast", ["J_b"], "J0", to=F16); vi("J0", F16, [1, 1, GH, GW])
    n("Cast", ["vend_b"], "V0", to=F16); vi("V0", F16, [1, 1, GH, GW])
    n("Cast", ["hend_b"], "H0", to=F16); vi("H0", F16, [1, 1, GH, GW])

    # ---- flood-MAX each seed through the mask (kernel 3) ---------------------
    # kernel must be 3 (radius 1): inter-box gap can be 1, so a bigger kernel would
    # leak a seed across the gap into a neighbouring box.  Per-seed iter counts =
    # the measured max BFS reach from that seed type to any box cell:
    #   junction (H boxes) reach <= 6 ; vertical/horizontal endpoints reach <= 8.
    def flood(seed, iters):
        cur = seed
        for i in range(iters):
            pooled = f"{seed}_p{i}"
            nodes.append(helper.make_node("MaxPool", [cur], [pooled],
                                          kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
            vi(pooled, F16, [1, 1, GH, GW])
            if i == iters - 1:
                # last step: skip the re-gate.  Cross-box leak can only land on the
                # 1-wide gap cell (non-green) which is discarded by the green-gated
                # Where chain downstream, so the classify is still exact on green cells.
                return pooled
            gated = f"{seed}_g{i}"
            n("Mul", [pooled, "m"], gated); vi(gated, F16, [1, 1, GH, GW])
            cur = gated
        return cur

    Jf = flood("J0", 6)
    Vf = flood("V0", 8)
    Hf = flood("H0", 8)

    # ---- classify -----------------------------------------------------------
    n("Greater", [Jf, "ZF16"], "isH"); vi("isH", BOOL, [1, 1, GH, GW])
    n("Greater", [Vf, "ZF16"], "hasV"); vi("hasV", BOOL, [1, 1, GH, GW])
    n("Greater", [Hf, "ZF16"], "hasH"); vi("hasH", BOOL, [1, 1, GH, GW])

    # U = exactly one endpoint type (XOR).  No need to exclude H here: the H Where is
    # applied LAST (highest priority) in the colour chain, so it overwrites any H cell
    # that XOR also flagged (WHERE-CHAIN PRIORITY).  Drops the notH plane.
    n("Xor", ["hasV", "hasH"], "xorVH"); vi("xorVH", BOOL, [1, 1, GH, GW])
    # the final flood step is un-gated, so isH/xorVH may be true on the 1-wide gap cell
    # (non-green) -> gate both by the green mask before the colour Where chain.
    n("And", ["isH", "mbool"], "isHg"); vi("isHg", BOOL, [1, 1, GH, GW])
    n("And", ["xorVH", "mbool"], "isU"); vi("isU", BOOL, [1, 1, GH, GW])
    # isL = green pixel that is neither H nor U  (handled by the Where chain default)

    # ---- colour-index plane L (uint8) via a uint8 Where priority chain -------
    #   off-grid -> 255 (matches no colour, all-zero output)
    #   bg in-grid -> 0
    #   green & H -> 2 ; green & U -> 6 ; green & (L) -> 1
    n("Cast", ["bg_f32"], "bg_b", to=BOOL); vi("bg_b", BOOL, [1, 1, GH, GW])
    n("Or", ["bg_b", "mbool"], "ingrid"); vi("ingrid", BOOL, [1, 1, GH, GW])
    init("U255", np.array(255, np.uint8).reshape(1, 1, 1, 1), U8)
    init("U0", np.array(0, np.uint8).reshape(1, 1, 1, 1), U8)
    init("U1", np.array(1, np.uint8).reshape(1, 1, 1, 1), U8)
    init("U2", np.array(2, np.uint8).reshape(1, 1, 1, 1), U8)
    init("U6", np.array(6, np.uint8).reshape(1, 1, 1, 1), U8)
    # base: in-grid ? (green?1:0) : 255   -> green default colour is 1 (L)
    n("Where", ["ingrid", "U0", "U255"], "Lbase"); vi("Lbase", U8, [1, 1, GH, GW])
    n("Where", ["mbool", "U1", "Lbase"], "Lg"); vi("Lg", U8, [1, 1, GH, GW])   # green->1 default
    n("Where", ["isU", "U6", "Lg"], "Lu"); vi("Lu", U8, [1, 1, GH, GW])        # U->6
    n("Where", ["isHg", "U2", "Lu"], "L_u8"); vi("L_u8", U8, [1, 1, GH, GW])   # H->2

    # ---- pad L back to 30x30 (off-grid sentinel 255 -> matches no colour) ----
    init("L_pads", np.array([0, 0, 0, 0, 0, 0, 30 - GH, 30 - GW], np.int64), I64)
    init("S255", np.array(255, np.uint8), U8)
    n("Pad", ["L_u8", "L_pads", "S255"], "L30", mode="constant"); vi("L30", U8, [1, 1, 30, 30])

    # ---- route 10-ch expansion into the FREE bool output --------------------
    init("arange10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), U8)
    n("Equal", ["L30", "arange10"], "output")   # [1,10,30,30] bool == graph output

    graph = helper.make_graph(
        nodes, "task364",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])],
        inits, value_info=vinfo,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
