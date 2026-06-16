"""task268 (ARC-AGI aba27056) — "yellow fountain" from a colored box.

Rule (from the generator, fully deterministic in 8 params size,wide,tall,col,row,
color,flip,xpose):
A single hollow rectangular box of one random colour sits near one grid edge with
a NOTCH (a gap) on the edge facing the grid interior.  The output draws the box
frame in its colour AND a yellow(4) "fountain" that erupts from the notched edge:
a yellow interior fill of the box, a vertical yellow band running from the box to
the far edge through the inner columns, and two yellow diagonal "arms" spreading
outward as they travel away from the box.

All geometry is a closed form of the box bounding box + the notch direction, with
flip/transpose orientation.  In the CANONICAL frame (notch on top, box at bottom),
with box bbox rows [r0,r1] cols [c0,c1]:
    YELLOW(rr,cc) = band  OR interior OR left-arm OR right-arm
        band     : c0+2 <= cc <= c1-2  AND  rr <= r1-1
        interior : r0+1<=rr<=r1-1  AND  c0+1<=cc<=c1-1
        left-arm : cc-rr == c0+2-r0  AND  rr <= r0-1
        right-arm: cc+rr == c1-2+r0  AND  rr <= r0-1
    FRAME(rr,cc) = box-perimeter AND NOT YELLOW
The observed orientation maps canonical->observed by (flip rows within the size
window) then (transpose); dir->(flip,xpose): top(0,0) bot(1,0) left(0,1) right(1,1).

Implementation: recover size (extent of any-channel occupancy — IN-GRID bg cells
carry ch0=1, OFF-GRID cells are all-zero, so the grid extent is recoverable!),
box bbox (occupancy of colf=Sum_k k*input_k>0), colour, and the notch direction
(the box edge whose occupancy count is < its length).  Build EFFECTIVE canonical
coordinate planes ER,EC per observed cell (folding flip+transpose into the ramps),
so all canonical predicates evaluate directly in the observed frame with NO plane
transpose.  Work on a 10x10 window (max grid size 10), Pad to 30x30 at the end.

Output one-hot routed into the FREE output via a Where priority chain
(yellow -> colour -> background), so no [1,10,H,W] plane is materialized.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
B = TensorProto.BOOL

W = 10  # grids are at most 10x10 at the top-left


def build(task):
    inits, nodes, vis = [], [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, list(ins), [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vis.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # ----------------------------------------------------------------- crop 10x10
    init("s0", np.array([0, 0], np.int64), np.int64)
    init("sWW", np.array([W, W], np.int64), np.int64)
    init("ax23", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "s0", "sWW", "ax23"], "win")            # [1,10,W,W] f32

    # any-channel occupancy (in-grid mask)  and  box value plane colf
    n("ReduceMax", ["win"], "anyc32", axes=[1], keepdims=1)      # [1,1,W,W] f32 (0/1)
    vi("anyc32", F32, [1, 1, W, W])
    # colf = sum_k k*input_k  via a 1x1 Conv (weight = arange) -> [1,1,W,W], no [1,10,W,W]
    convw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("convw", convw, np.float32)
    n("Conv", ["win", "convw"], "colf32")                        # [1,1,W,W] f32
    vi("colf32", F32, [1, 1, W, W])
    n("Cast", ["colf32"], "colf", to=F16); vi("colf", F16, [1, 1, W, W])
    n("Cast", ["anyc32"], "anyc", to=F16); vi("anyc", F16, [1, 1, W, W])
    n("Greater", ["colf", "z0f"], "occ")                         # [1,1,W,W] bool (box)
    init("z0f", np.array(0.0, np.float16), np.float16)
    vi("occ", B, [1, 1, W, W])
    n("Cast", ["occ"], "occf", to=F16)
    vi("occf", F16, [1, 1, W, W])

    # ----------------------------------------------------------------- scalars
    rr = np.arange(W, dtype=np.float16).reshape(1, 1, W, 1)
    cc = np.arange(W, dtype=np.float16).reshape(1, 1, 1, W)
    init("rr", rr, np.float16)
    init("cc", cc, np.float16)
    init("BIG", np.array(99.0, np.float16), np.float16)
    init("NBIG", np.array(-99.0, np.float16), np.float16)
    init("half", np.array(0.5, np.float16), np.float16)

    # size = 1 + max index (row/col) where anyc>0
    n("ReduceMax", ["anyc"], "anyrow", axes=[3], keepdims=1)      # [1,1,W,1]
    vi("anyrow", F16, [1, 1, W, 1])
    n("ReduceMax", ["anyc"], "anycol", axes=[2], keepdims=1)      # [1,1,1,W]
    vi("anycol", F16, [1, 1, 1, W])
    n("Greater", ["anyrow", "half"], "anyrb"); vi("anyrb", B, [1, 1, W, 1])
    n("Greater", ["anycol", "half"], "anycb"); vi("anycb", B, [1, 1, 1, W])
    n("Where", ["anyrb", "rr", "NBIG"], "szr_src"); vi("szr_src", F16, [1, 1, W, 1])
    n("ReduceMax", ["szr_src"], "maxr", axes=[2], keepdims=1); vi("maxr", F16, [1, 1, 1, 1])
    n("Where", ["anycb", "cc", "NBIG"], "szc_src"); vi("szc_src", F16, [1, 1, 1, W])
    n("ReduceMax", ["szc_src"], "maxc", axes=[3], keepdims=1); vi("maxc", F16, [1, 1, 1, 1])
    n("Max", ["maxr", "maxc"], "szm"); vi("szm", F16, [1, 1, 1, 1])  # size-1
    init("one_f", np.array(1.0, np.float16), np.float16)

    # box bbox R0,R1,C0,C1 from occ
    n("ReduceMax", ["occf"], "occrow", axes=[3], keepdims=1); vi("occrow", F16, [1, 1, W, 1])
    n("ReduceMax", ["occf"], "occcol", axes=[2], keepdims=1); vi("occcol", F16, [1, 1, 1, W])
    n("Greater", ["occrow", "half"], "occrb"); vi("occrb", B, [1, 1, W, 1])
    n("Greater", ["occcol", "half"], "occcb"); vi("occcb", B, [1, 1, 1, W])
    n("Where", ["occrb", "rr", "BIG"], "R0src"); vi("R0src", F16, [1, 1, W, 1])
    n("ReduceMin", ["R0src"], "R0", axes=[2], keepdims=1); vi("R0", F16, [1, 1, 1, 1])
    n("Where", ["occrb", "rr", "NBIG"], "R1src"); vi("R1src", F16, [1, 1, W, 1])
    n("ReduceMax", ["R1src"], "R1", axes=[2], keepdims=1); vi("R1", F16, [1, 1, 1, 1])
    n("Where", ["occcb", "cc", "BIG"], "C0src"); vi("C0src", F16, [1, 1, 1, W])
    n("ReduceMin", ["C0src"], "C0", axes=[3], keepdims=1); vi("C0", F16, [1, 1, 1, 1])
    n("Where", ["occcb", "cc", "NBIG"], "C1src"); vi("C1src", F16, [1, 1, 1, W])
    n("ReduceMax", ["C1src"], "C1", axes=[3], keepdims=1); vi("C1", F16, [1, 1, 1, 1])

    # colour scalar + one-hot [1,10,1,1]
    n("ReduceMax", ["colf"], "colmax", axes=[2, 3], keepdims=1); vi("colmax", F16, [1, 1, 1, 1])
    init("chvec", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)

    # ----------------------------------------------- notch direction (flip,xpose)
    # edge fullness: count occ along each bbox edge vs its length.
    # top row R0 mask, bottom row R1, left col C0, right col C1 within bbox span.
    # rowsel_R0[r] = (rr==R0); colsel within [C0,C1]; etc. Use occ counts.
    # length of horizontal edge = C1-C0+1 ; vertical = R1-R0+1
    n("Sub", ["C1", "C0"], "wm1"); vi("wm1", F16, [1, 1, 1, 1])
    n("Add", ["wm1", "one_f"], "wide"); vi("wide", F16, [1, 1, 1, 1])
    n("Sub", ["R1", "R0"], "tm1"); vi("tm1", F16, [1, 1, 1, 1])
    n("Add", ["tm1", "one_f"], "tall"); vi("tall", F16, [1, 1, 1, 1])
    # row/col boundary selectors. only bot/left/right edges needed (top is the
    # default canonical orientation; flip=bot|rgt, xpose=lft|rgt).
    n("Equal", ["rr", "R1"], "is_r1"); vi("is_r1", B, [1, 1, W, 1])
    n("Equal", ["cc", "C0"], "is_c0"); vi("is_c0", B, [1, 1, 1, W])
    n("Equal", ["cc", "C1"], "is_c1"); vi("is_c1", B, [1, 1, 1, W])
    n("And", ["occ", "is_r1"], "e_bot"); vi("e_bot", B, [1, 1, W, W])
    n("And", ["occ", "is_c0"], "e_lft"); vi("e_lft", B, [1, 1, W, W])
    n("And", ["occ", "is_c1"], "e_rgt"); vi("e_rgt", B, [1, 1, W, W])
    n("Cast", ["e_bot"], "e_botf", to=F16); vi("e_botf", F16, [1, 1, W, W])
    n("Cast", ["e_lft"], "e_lftf", to=F16); vi("e_lftf", F16, [1, 1, W, W])
    n("Cast", ["e_rgt"], "e_rgtf", to=F16); vi("e_rgtf", F16, [1, 1, W, W])
    n("ReduceSum", ["e_botf"], "n_bot", axes=[2, 3], keepdims=1); vi("n_bot", F16, [1, 1, 1, 1])
    n("ReduceSum", ["e_lftf"], "n_lft", axes=[2, 3], keepdims=1); vi("n_lft", F16, [1, 1, 1, 1])
    n("ReduceSum", ["e_rgtf"], "n_rgt", axes=[2, 3], keepdims=1); vi("n_rgt", F16, [1, 1, 1, 1])
    # notch edge: count < length. horizontal edges compare to wide, vertical to tall.
    n("Less", ["n_bot", "wide"], "notch_bot"); vi("notch_bot", B, [1, 1, 1, 1])
    n("Less", ["n_lft", "tall"], "notch_lft"); vi("notch_lft", B, [1, 1, 1, 1])
    n("Less", ["n_rgt", "tall"], "notch_rgt"); vi("notch_rgt", B, [1, 1, 1, 1])
    # flip = notch in {bot,right}; xpose = notch in {left,right}
    n("Or", ["notch_bot", "notch_rgt"], "flip"); vi("flip", B, [1, 1, 1, 1])
    n("Or", ["notch_lft", "notch_rgt"], "xpose"); vi("xpose", B, [1, 1, 1, 1])

    # ------------------------------------------------ canonical bbox (un-transform)
    # invert: un-xpose (swap r<->c) then un-flip (r -> size-1-r).
    # If xpose: (r0,r1,c0,c1) = (C0,C1,R0,R1) else (R0,R1,C0,C1).
    n("Where", ["xpose", "C0", "R0"], "a0"); vi("a0", F16, [1, 1, 1, 1])
    n("Where", ["xpose", "C1", "R1"], "a1"); vi("a1", F16, [1, 1, 1, 1])
    n("Where", ["xpose", "R0", "C0"], "cc0"); vi("cc0", F16, [1, 1, 1, 1])
    n("Where", ["xpose", "R1", "C1"], "cc1"); vi("cc1", F16, [1, 1, 1, 1])
    # un-flip rows: if flip, r0' = size-1-a1, r1' = size-1-a0
    n("Sub", ["szm", "a1"], "fr0"); vi("fr0", F16, [1, 1, 1, 1])   # size-1-a1
    n("Sub", ["szm", "a0"], "fr1"); vi("fr1", F16, [1, 1, 1, 1])   # size-1-a0
    n("Where", ["flip", "fr0", "a0"], "r0"); vi("r0", F16, [1, 1, 1, 1])
    n("Where", ["flip", "fr1", "a1"], "r1"); vi("r1", F16, [1, 1, 1, 1])
    n("Identity", ["cc0"], "c0"); vi("c0", F16, [1, 1, 1, 1])
    n("Identity", ["cc1"], "c1"); vi("c1", F16, [1, 1, 1, 1])

    # ------------------------------------- effective canonical coordinate planes
    # For observed cell (or,oc):  a = xpose? oc : or ;  b = xpose? or : oc
    #   ER = flip ? size-1-a : a ;  EC = b
    # Build a-plane and b-plane (full [1,1,W,W]).
    # or-plane = rr broadcast, oc-plane = cc broadcast.
    # a = Where(xpose, cc, rr) broadcast; b = Where(xpose, rr, cc)
    n("Where", ["xpose", "cc", "rr"], "aplane"); vi("aplane", F16, [1, 1, W, W])
    n("Where", ["xpose", "rr", "cc"], "bplane"); vi("bplane", F16, [1, 1, W, W])
    # ER = flip ? szm - aplane : aplane
    n("Sub", ["szm", "aplane"], "afl"); vi("afl", F16, [1, 1, W, W])
    n("Where", ["flip", "afl", "aplane"], "ER"); vi("ER", F16, [1, 1, W, W])
    n("Identity", ["bplane"], "EC"); vi("EC", F16, [1, 1, W, W])

    # ------------------------------------------- canonical predicates on ER,EC
    # band: c0+2 <= EC <= c1-2 AND ER <= r1-1
    init("two_f", np.array(2.0, np.float16), np.float16)
    n("Add", ["c0", "two_f"], "c0p2"); vi("c0p2", F16, [1, 1, 1, 1])
    n("Sub", ["c1", "two_f"], "c1m2"); vi("c1m2", F16, [1, 1, 1, 1])
    n("Sub", ["r1", "one_f"], "r1m1"); vi("r1m1", F16, [1, 1, 1, 1])
    # EC >= c0p2  == Not(EC < c0p2)
    n("Less", ["EC", "c0p2"], "lt_a"); vi("lt_a", B, [1, 1, W, W])
    n("Less", ["c1m2", "EC"], "gt_a"); vi("gt_a", B, [1, 1, W, W])
    n("Or", ["lt_a", "gt_a"], "band_c_out"); vi("band_c_out", B, [1, 1, W, W])
    n("Not", ["band_c_out"], "band_c"); vi("band_c", B, [1, 1, W, W])
    # ER <= r1-1
    n("Less", ["r1m1", "ER"], "band_r_gt"); vi("band_r_gt", B, [1, 1, W, W])
    n("Not", ["band_r_gt"], "band_r"); vi("band_r", B, [1, 1, W, W])
    n("And", ["band_c", "band_r"], "band"); vi("band", B, [1, 1, W, W])

    # interior: r0+1<=ER<=r1-1 AND c0+1<=EC<=c1-1
    n("Add", ["r0", "one_f"], "r0p1"); vi("r0p1", F16, [1, 1, 1, 1])
    n("Add", ["c0", "one_f"], "c0p1"); vi("c0p1", F16, [1, 1, 1, 1])
    n("Sub", ["c1", "one_f"], "c1m1"); vi("c1m1", F16, [1, 1, 1, 1])
    n("Less", ["ER", "r0p1"], "i_rlt"); vi("i_rlt", B, [1, 1, W, W])
    n("Less", ["r1m1", "ER"], "i_rgt"); vi("i_rgt", B, [1, 1, W, W])
    n("Less", ["EC", "c0p1"], "i_clt"); vi("i_clt", B, [1, 1, W, W])
    n("Less", ["c1m1", "EC"], "i_cgt"); vi("i_cgt", B, [1, 1, W, W])
    n("Or", ["i_rlt", "i_rgt"], "i_ro"); vi("i_ro", B, [1, 1, W, W])
    n("Or", ["i_clt", "i_cgt"], "i_co"); vi("i_co", B, [1, 1, W, W])
    n("Or", ["i_ro", "i_co"], "i_out"); vi("i_out", B, [1, 1, W, W])
    n("Not", ["i_out"], "interior"); vi("interior", B, [1, 1, W, W])

    # arms: rr-region ER <= r0-1
    n("Sub", ["r0", "one_f"], "r0m1"); vi("r0m1", F16, [1, 1, 1, 1])
    n("Less", ["r0m1", "ER"], "arm_r_gt"); vi("arm_r_gt", B, [1, 1, W, W])
    n("Not", ["arm_r_gt"], "arm_r"); vi("arm_r", B, [1, 1, W, W])
    # left arm: EC-ER == c0+2-r0
    n("Sub", ["c0p2", "r0"], "lk"); vi("lk", F16, [1, 1, 1, 1])
    n("Sub", ["EC", "ER"], "ecmer"); vi("ecmer", F16, [1, 1, W, W])
    n("Equal", ["ecmer", "lk"], "larm_d"); vi("larm_d", B, [1, 1, W, W])
    n("And", ["larm_d", "arm_r"], "larm"); vi("larm", B, [1, 1, W, W])
    # right arm: EC+ER == c1-2+r0
    n("Add", ["c1m2", "r0"], "rk"); vi("rk", F16, [1, 1, 1, 1])
    n("Add", ["EC", "ER"], "ecper"); vi("ecper", F16, [1, 1, W, W])
    n("Equal", ["ecper", "rk"], "rarm_d"); vi("rarm_d", B, [1, 1, W, W])
    n("And", ["rarm_d", "arm_r"], "rarm"); vi("rarm", B, [1, 1, W, W])

    # YELLOW = band OR interior OR larm OR rarm
    n("Or", ["band", "interior"], "y1"); vi("y1", B, [1, 1, W, W])
    n("Or", ["larm", "rarm"], "y2"); vi("y2", B, [1, 1, W, W])
    n("Or", ["y1", "y2"], "yellow"); vi("yellow", B, [1, 1, W, W])

    # FRAME = occ (input box cells) AND NOT yellow  -- input box minus yellow, no geometry
    n("Not", ["yellow"], "notyellow"); vi("notyellow", B, [1, 1, W, W])
    n("And", ["occ", "notyellow"], "frame"); vi("frame", B, [1, 1, W, W])

    # ----------------------------------------------------- route to output via L
    # Build a single value plane L[1,1,W,W]:
    #   off-grid -> sentinel 99 (no channel matches) ; yellow -> 4 ; frame -> colour ; else 0
    # then Equal(L_padded, arange[1,10,1,1]) -> FREE bool output.  Pad with sentinel.
    init("v4", np.array(4.0, np.float16), np.float16)
    init("v0", np.array(0.0, np.float16), np.float16)
    init("v99", np.array(99.0, np.float16), np.float16)
    # frame value = colour (= colmax scalar)
    n("Where", ["frame", "colmax", "v0"], "L_fb"); vi("L_fb", F16, [1, 1, W, W])
    n("Where", ["yellow", "v4", "L_fb"], "L_yfb"); vi("L_yfb", F16, [1, 1, W, W])
    # off-grid -> 99
    n("Greater", ["anyc", "half"], "ingrid"); vi("ingrid", B, [1, 1, W, W])
    n("Where", ["ingrid", "L_yfb", "v99"], "L"); vi("L", F16, [1, 1, W, W])
    # Pad to 30x30 with sentinel 99 (off-window cells are off-grid)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    n("Pad", ["L", "pads", "v99"], "Lp", mode="constant"); vi("Lp", F16, [1, 1, 30, 30])
    n("Equal", ["Lp", "chvec"], "output")

    out_vi = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    in_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])

    graph = helper.make_graph(nodes, "task268", [in_vi], [out_vi], inits, value_info=vis)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
