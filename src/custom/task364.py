"""task364 (ARC-AGI e509e548) — recolour each green L/U/H box by its SHAPE.

Rule (from generator):
  1+max(H,W)//3 non-overlapping boxes (bbox gap >= 1) are drawn in GREEN(3) on a
  black canvas; grid is H in [10,20], W = H+[-2,2] (so <= 20 rows x 22 cols).  Each
  box is one of three sprite skeletons (any of 8 dihedral orientations):
    - "el"    (L): one full arm + one perpendicular arm meeting at a corner -> blue(1)
    - "you"   (U): two full parallel arms + one connecting bar  (3-sided)   -> pink(6)
    - "aitch" (H): left bar + half right bar + a middle cross-bar            -> red(2)
  Output recolours every green pixel by its box's shape class; background stays 0.

CLOSED-FORM SHAPE DISCRIMINATOR (verified 0 bad / fresh thousands; no flood-fill of
unique labels, no NonZero/ScatterND):
  Compute the 4-neighbour bit-code on the green mask via ONE 3x3 Conv
    code = 1*up + 2*down + 4*left + 8*right   (presence of each green 4-neighbour)
  then three LOCAL feature seeds (gated to green):
    v_end  = code in {1,2}        (degree-1 cell whose lone neighbour is vertical)
    h_end  = code in {4,8}        (degree-1 cell whose lone neighbour is horizontal)
    junc   = code in {7,11} or >=13   (a degree>=3 junction)  -> only H has one
  Each box's pixels need their box's flag.  Because every box pixel is within
  geodesic distance <= 7 of a feature cell along the thin line, a SHORT mask-gated
  3x3 MaxPool flood propagates the flags exactly (kernel must be 3: inter-box gap
  can be 1).  Two complementary MAX-floods recover all three flags without losing
  the L/U count:
    pos_code  (junc=4, v_end=2, h_end=1)  -> max-flood; a box's max code reveals
              the HIGHEST-priority feature present.
    inv_code  (v_end=2, h_end=4)          -> max-flood; distinguishes an L box
              (has BOTH endpoint orientations -> inv reaches 4) from a U box
              (one orientation only -> inv stays 2).
  Decode:
    isH = pos > 2            (junction present)
    isL = (pos == 2) AND (inv > 2)   (no junction, has a v_end, AND an h_end)
    isU = green AND not(isH or isL)

Encoding:  slice to the 20x22 max grid so every plane is measured there (not 30x30);
  flood in UINT8 (1 byte/plane, opset>=12 supports uint8 MaxPool) -> half the fp16
  cost; build the colour index in uint8 via a Where priority chain; Pad back to
  30x30 with an off-grid sentinel (255, matches no colour channel); the 10-channel
  expansion lands in the FREE bool output via Equal(L, arange).

Result (src.harness.evaluate): pts=14.607  mem=32580  params=61  pass 266/266
  stored, 0/3000 fresh.  Beats the deployed kojimar flood net (14.542, 34320 B) by
  +0.06 only -> MARGINAL.  The cost is dominated by the flood (2 complementary
  uint8 floods x 7 iters = ~11.9 KB) which is irreducible: 7 iters is the measured
  minimum geodesic reach, a kernel-3 mask re-gate is required every step (inter-box
  gap can be 1 cell), and TWO floods are genuinely needed (a single max-flood cannot
  count 1-vs-2 endpoint orientations to separate L from U).  This is the documented
  flood-at-floor regime; the deployed net is already a tuned flood near the same floor.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

GH, GW = 20, 22       # max grid (H<=20, W<=22)
ITERS_P = 7           # >= max geodesic reach of any seed to a box cell (measured 7)
ITERS_I = 7
OPSET = 12            # uint8 MaxPool requires opset >= 12


def build(task):
    inits, nodes, vinfo = [], [], []
    NP = {I64: np.int64, F32: np.float32, U8: np.uint8, BOOL: np.bool_}

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=NP[dt]), name))
        return name

    def n(op, ins, out, **a):
        nodes.append(helper.make_node(op, ins, [out], **a))
        return out

    def vi(name, dt, shape):
        vinfo.append(helper.make_tensor_value_info(name, dt, shape))
        return name

    # ---- slice the active 20x22 region: green(ch3) and bg(ch0) ----------------
    init("s_g", np.array([0, 3, 0, 0], np.int64), I64)
    init("e_g", np.array([1, 4, GH, GW], np.int64), I64)
    init("s_b", np.array([0, 0, 0, 0], np.int64), I64)
    init("e_b", np.array([1, 1, GH, GW], np.int64), I64)
    init("ax4", np.array([0, 1, 2, 3], np.int64), I64)
    n("Slice", ["input", "s_g", "e_g", "ax4"], "green_f32"); vi("green_f32", F32, [1, 1, GH, GW])
    n("Slice", ["input", "s_b", "e_b", "ax4"], "bg_f32");    vi("bg_f32", F32, [1, 1, GH, GW])
    # green_f32 is already {0,1}; use it directly as the mask (no extra f32 plane).
    n("Cast", ["green_f32"], "mbool", to=BOOL); vi("mbool", BOOL, [1, 1, GH, GW])

    # ---- neighbour bit-code via ONE 3x3 Conv (f32) ----------------------------
    Kc = np.zeros((1, 1, 3, 3), np.float32)
    Kc[0, 0, 0, 1] = 1.0   # up
    Kc[0, 0, 2, 1] = 2.0   # down
    Kc[0, 0, 1, 0] = 4.0   # left
    Kc[0, 0, 1, 2] = 8.0   # right
    init("Kc", Kc, F32)
    n("Conv", ["green_f32", "Kc"], "code", pads=[1, 1, 1, 1]); vi("code", F32, [1, 1, GH, GW])

    # feature masks (3 bool planes).  code is integer-valued on the f32 conv plane.
    #   v_end: code in {1,2}     -> 0.5 < code < 2.5
    #   h_end: code in {4,8}
    #   junc : code in {7,11,13,14,15}  (degree>=3)  ->  code>=7 AND code not in {8,9,10,12}
    init("c05", np.array(0.5, np.float32), F32)
    init("c25", np.array(2.5, np.float32), F32)
    init("c4", np.array(4.0, np.float32), F32)
    init("c8", np.array(8.0, np.float32), F32)
    init("c7", np.array(7.0, np.float32), F32)
    init("c12", np.array(12.0, np.float32), F32)
    init("c105", np.array(10.5, np.float32), F32)
    # v_end
    n("Greater", ["code", "c05"], "vg"); vi("vg", BOOL, [1, 1, GH, GW])
    n("Less", ["code", "c25"], "vl"); vi("vl", BOOL, [1, 1, GH, GW])
    n("And", ["vg", "vl"], "vend"); vi("vend", BOOL, [1, 1, GH, GW])
    # h_end
    n("Equal", ["code", "c4"], "hl"); vi("hl", BOOL, [1, 1, GH, GW])
    n("Equal", ["code", "c8"], "hr"); vi("hr", BOOL, [1, 1, GH, GW])
    n("Or", ["hl", "hr"], "hend"); vi("hend", BOOL, [1, 1, GH, GW])
    # junc: degree>=3 codes are {7,11,13,14,15}.  code 12 (left+right) is a
    #   horizontal-through deg-2 cell, NOT a junction, so it must be excluded.
    #   junc = (code==7) OR (code>=11 AND code!=12)
    n("Equal", ["code", "c7"], "j7"); vi("j7", BOOL, [1, 1, GH, GW])
    n("Greater", ["code", "c105"], "j11p"); vi("j11p", BOOL, [1, 1, GH, GW])  # code>=11
    n("Equal", ["code", "c12"], "j12"); vi("j12", BOOL, [1, 1, GH, GW])       # code==12
    n("Not", ["j12"], "n12"); vi("n12", BOOL, [1, 1, GH, GW])
    n("And", ["j11p", "n12"], "j11"); vi("j11", BOOL, [1, 1, GH, GW])
    n("Or", ["j7", "j11"], "junc"); vi("junc", BOOL, [1, 1, GH, GW])

    # ---- build the two uint8 flood codes (gate by green at the end) ------------
    init("u0", np.array(0, np.uint8).reshape(1, 1, 1, 1), U8)
    init("u1", np.array(1, np.uint8).reshape(1, 1, 1, 1), U8)
    init("u2", np.array(2, np.uint8).reshape(1, 1, 1, 1), U8)
    init("u4", np.array(4, np.uint8).reshape(1, 1, 1, 1), U8)
    # pos_code: h_end=1, v_end=2, junc=4 (priority via Where order); then green-gate
    n("Where", ["hend", "u1", "u0"], "p_h"); vi("p_h", U8, [1, 1, GH, GW])
    n("Where", ["vend", "u2", "p_h"], "p_hv"); vi("p_hv", U8, [1, 1, GH, GW])
    n("Where", ["junc", "u4", "p_hv"], "p_raw"); vi("p_raw", U8, [1, 1, GH, GW])
    n("Where", ["mbool", "p_raw", "u0"], "pos_code"); vi("pos_code", U8, [1, 1, GH, GW])
    # inv_code: v_end=2, h_end=4; then green-gate
    n("Where", ["vend", "u2", "u0"], "i_v"); vi("i_v", U8, [1, 1, GH, GW])
    n("Where", ["hend", "u4", "i_v"], "i_raw"); vi("i_raw", U8, [1, 1, GH, GW])
    n("Where", ["mbool", "i_raw", "u0"], "inv_code"); vi("inv_code", U8, [1, 1, GH, GW])

    # ---- mask-gated uint8 MaxPool flood (kernel 3) ----------------------------
    def flood(seed, iters):
        cur = seed
        for i in range(iters):
            p = f"{seed}_p{i}"
            nodes.append(helper.make_node(
                "MaxPool", [cur], [p], kernel_shape=[3, 3],
                pads=[1, 1, 1, 1], strides=[1, 1]))
            vi(p, U8, [1, 1, GH, GW])
            if i == iters - 1:
                # last step un-gated: a 1-wide-gap leak lands only on non-green
                # cells, discarded by the green-gated colour chain downstream.
                return p
            g = f"{seed}_g{i}"
            n("Where", ["mbool", p, "u0"], g); vi(g, U8, [1, 1, GH, GW])
            cur = g
        return cur

    posf = flood("pos_code", ITERS_P)
    invf = flood("inv_code", ITERS_I)

    # ---- decode ---------------------------------------------------------------
    n("Greater", [posf, "u2"], "is2"); vi("is2", BOOL, [1, 1, GH, GW])       # H
    n("Equal", [posf, "u2"], "pos_eq2"); vi("pos_eq2", BOOL, [1, 1, GH, GW])
    n("Greater", [invf, "u2"], "inv_gt2"); vi("inv_gt2", BOOL, [1, 1, GH, GW])
    n("And", ["pos_eq2", "inv_gt2"], "is1"); vi("is1", BOOL, [1, 1, GH, GW])  # L
    n("Or", ["is2", "is1"], "not6"); vi("not6", BOOL, [1, 1, GH, GW])
    n("Not", ["not6"], "maybe6"); vi("maybe6", BOOL, [1, 1, GH, GW])
    n("And", ["mbool", "maybe6"], "is6"); vi("is6", BOOL, [1, 1, GH, GW])     # U
    # gate H by mask (flood last step un-gated may flag a 1-wide gap cell).
    # L needs no gate: it is the colour-chain default for green cells.
    n("And", ["mbool", "is2"], "is2g"); vi("is2g", BOOL, [1, 1, GH, GW])

    # ---- colour-index plane (uint8) via Where priority chain ------------------
    #   off-grid -> 255 (matches no colour); in-grid bg -> 0; green default L=1.
    n("Cast", ["bg_f32"], "bg_b", to=BOOL); vi("bg_b", BOOL, [1, 1, GH, GW])
    n("Or", ["bg_b", "mbool"], "ingrid"); vi("ingrid", BOOL, [1, 1, GH, GW])
    init("U255", np.array(255, np.uint8).reshape(1, 1, 1, 1), U8)
    init("U6", np.array(6, np.uint8).reshape(1, 1, 1, 1), U8)
    n("Where", ["ingrid", "u0", "U255"], "Lbase"); vi("Lbase", U8, [1, 1, GH, GW])
    n("Where", ["mbool", "u1", "Lbase"], "Lg"); vi("Lg", U8, [1, 1, GH, GW])      # green->1
    n("Where", ["is6", "U6", "Lg"], "L6"); vi("L6", U8, [1, 1, GH, GW])           # U->6
    n("Where", ["is2g", "u2", "L6"], "L_u8"); vi("L_u8", U8, [1, 1, GH, GW])      # H->2

    # ---- pad back to 30x30 (off-grid sentinel 255) ----------------------------
    init("L_pads", np.array([0, 0, 0, 0, 0, 0, 30 - GH, 30 - GW], np.int64), I64)
    init("S255", np.array(255, np.uint8), U8)
    n("Pad", ["L_u8", "L_pads", "S255"], "L30", mode="constant"); vi("L30", U8, [1, 1, 30, 30])

    # ---- route 10-channel expansion into the FREE bool output -----------------
    init("arange10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), U8)
    n("Equal", ["L30", "arange10"], "output")   # [1,10,30,30] bool == graph output

    graph = helper.make_graph(
        nodes, "task364",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])],
        inits, value_info=vinfo,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    return model
