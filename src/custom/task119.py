"""task119 (ARC-AGI 508bd3b6) — "bounced diagonal ray" reconstruction.

Rule (from the ARC-GEN generator, verified fresh):
  Canonically (before flip / gravity): a solid RED(2) wall occupies the top
  `depth` rows.  A V-shaped diagonal "ray" lives below it:
      row(c) = depth + |mid - c|
  i.e. a 45-degree ray that comes down, bounces off the wall edge at the vertex
  (depth, mid), and goes back down.  The ray cells are CYAN(8) for the first
  `shown` columns (c < shown) and GREEN(3) elsewhere.
  The INPUT shows only the wall + the cyan stub (the green ray is erased to
  background); the OUTPUT redraws the full ray (green) keeping cyan + wall.
  Finally `flip` (horizontal mirror) and `gravity` (transpose / row-reverse)
  rotate the whole figure into one of 8 orientations, so the wall ends up a
  solid band on any of the 4 sides and the ray runs along the two diagonals.

  The full ray is exactly the union of two 45-degree diagonals through the
  vertex V=(vr,vc):   (r+c == vr+vc)  OR  (r-c == vr-vc),
  restricted to non-wall cells on the far side of V (away from the wall).

Recovery from the INPUT only (verified 0 / 40000 fresh instances, all 8
orientations):
  * cyan stub Y[1,1,12,12] (channel 8), wall R[1,1,12,12] (channel 2).
  * wall side = whichever full edge of the 12x12 grid is all-red (top/bot/
    left/right scalar flags from edge ReduceMin).  band = wall thickness.
  * define per-side NORMAL coord (perp to wall, grows away from wall) and
    PARALLEL coord.  The vertex sits on the wall inner edge (normal == band).
  * the cyan stub lies on one arm; its slope in (par vs normal) is +-1.
    Take the cyan cells NEAREST and FARTHEST from the wall (min/max normal),
    slope = (par_near - par_far)/(norm_near - norm_far) (exact +-1), then
    extrapolate vpar = par_near + slope*(band - norm_near).
  * a = vr+vc, b = vr-vc.  path = (r+c==a or r-c==b) & not-wall & far-side-of-V.
  * label L: red->2, cyan->8, path-green->3, else sentinel.  Equal(L,arange)
    into the FREE bool output.

Memory: all per-cell work on a 12x12 canvas (144 elems).  The only 30x30
intermediate is the uint8 label map padded just before the final Equal.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

W = 12  # active canvas side (grid is always 12x12 anchored top-left)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16   # all per-cell float work is fp16 (small ints, exact)
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    # ---- constants (fp16; coords 0..23, slope +-1 all fp16-exact) ----
    init("halfF", np.array(0.5, np.float32), np.float32)   # for fp32 CY/RE masks
    init("big", np.array(99.0, np.float16), np.float16)
    init("negbig", np.array(-99.0, np.float16), np.float16)
    init("Wm1", np.array(float(W - 1), np.float16), np.float16)   # 11
    # final Equal comparator + sentinel + pad
    init("chan_u8", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("sent_u8", np.array(99, np.uint8), np.uint8)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    # 1-D row / col ramps (12 elems each) -> nrm/par built separably
    init("RIv", np.arange(W, dtype=np.float16).reshape(1, 1, W, 1), np.float16)
    init("CIv", np.arange(W, dtype=np.float16).reshape(1, 1, 1, W), np.float16)

    # ---- crop cyan(ch8) and red(ch2) planes to 12x12 (top-left) ----
    # Slice the 30x30 one-hot down to channel + 12x12 spatial.
    init("s_cy_start", np.array([0, 8, 0, 0], np.int64), np.int64)
    init("s_cy_end", np.array([1, 9, W, W], np.int64), np.int64)
    init("s_re_start", np.array([0, 2, 0, 0], np.int64), np.int64)
    init("s_re_end", np.array([1, 3, W, W], np.int64), np.int64)
    init("axes4", np.array([0, 1, 2, 3], np.int64), np.int64)
    # keep these as fp32 (Greater/ReduceSum accept fp32); avoids an extra cast plane.
    n("Slice", ["input", "s_cy_start", "s_cy_end", "axes4"], "CY")   # [1,1,12,12] fp32
    n("Slice", ["input", "s_re_start", "s_re_end", "axes4"], "RE")   # [1,1,12,12] fp32

    # ---- wall side detection from the red plane's four edges ----
    # row sums / col sums (over the 12x12 plane).
    n("ReduceSum", ["RE"], "rowsum", axes=[3], keepdims=1)   # [1,1,12,1] cells per row
    n("ReduceSum", ["RE"], "colsum", axes=[2], keepdims=1)   # [1,1,1,12] cells per col
    # an edge is "wall" if that whole edge row/col is full (sum == 12).
    init("Wf", np.array(float(W), np.float32), np.float32)   # 12 (RE is fp32)
    init("Whalf", np.array(float(W) - 0.5, np.float32), np.float32)  # 11.5
    # extract scalar edge sums via Slice on the profile vectors
    init("r0s", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("r0e", np.array([1, 1, 1, 1], np.int64), np.int64)
    init("r11s", np.array([0, 0, W - 1, 0], np.int64), np.int64)
    init("r11e", np.array([1, 1, W, 1], np.int64), np.int64)
    init("c0s", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("c0e", np.array([1, 1, 1, 1], np.int64), np.int64)
    init("c11s", np.array([0, 0, 0, W - 1], np.int64), np.int64)
    init("c11e", np.array([1, 1, 1, W], np.int64), np.int64)
    n("Slice", ["rowsum", "r0s", "r0e", "axes4"], "top_e")     # scalar [1,1,1,1]
    n("Slice", ["rowsum", "r11s", "r11e", "axes4"], "bot_e")
    n("Slice", ["colsum", "c0s", "c0e", "axes4"], "left_e")
    n("Slice", ["colsum", "c11s", "c11e", "axes4"], "right_e")
    n("Greater", ["top_e", "Whalf"], "topB")
    n("Greater", ["bot_e", "Whalf"], "botB")
    n("Greater", ["left_e", "Whalf"], "leftB")
    n("Greater", ["right_e", "Whalf"], "rightB")
    for nm in ("topB", "botB", "leftB", "rightB"):
        n("Cast", [nm], nm + "f", to=H)
    # band = wall thickness. For a top/bottom wall, every column has `band` red
    # cells; for left/right, every row has `band`.  Total red = band*12, so
    # band = sum(RE)/12 in every orientation.
    n("ReduceSum", ["RE"], "redtot", axes=[2, 3], keepdims=1)   # [1,1,1,1] fp32
    n("Div", ["redtot", "Wf"], "band32")                        # scalar fp32
    n("Cast", ["band32"], "band", to=H)                         # fp16 (feeds fp16 math)

    # ---- normal / parallel coordinate fields (built SEPARABLY from 1-D) ----
    # top:   nrm = RI,        par = CI
    # bottom:nrm = 11 - RI,   par = CI
    # left:  nrm = CI,        par = RI
    # right: nrm = 11 - CI,   par = RI
    # All 1-D ops are 12-elem (~24 B); only nrm/par broadcast to one 12x12 plane.
    n("Sub", ["Wm1", "RIv"], "RIrev")   # 11 - RI  (1-D)
    n("Sub", ["Wm1", "CIv"], "CIrev")   # 11 - CI  (1-D)
    n("Add", ["topBf", "botBf"], "tb_sel")     # scalar
    n("Add", ["leftBf", "rightBf"], "lr_sel")  # scalar
    # nrm row component (1-D over rows): top->RI, bottom->RIrev, else 0
    n("Mul", ["topBf", "RIv"], "nrmR_t")
    n("Mul", ["botBf", "RIrev"], "nrmR_b")
    n("Add", ["nrmR_t", "nrmR_b"], "nrmRow")   # [1,1,12,1]
    # nrm col component (1-D over cols): left->CI, right->CIrev, else 0
    n("Mul", ["leftBf", "CIv"], "nrmC_l")
    n("Mul", ["rightBf", "CIrev"], "nrmC_r")
    n("Add", ["nrmC_l", "nrmC_r"], "nrmCol")   # [1,1,1,12]
    n("Add", ["nrmRow", "nrmCol"], "nrm")      # broadcast -> ONE [1,1,12,12]
    # par row / col components
    n("Mul", ["lr_sel", "RIv"], "parRow")      # [1,1,12,1] (rows used when left/right)
    n("Mul", ["tb_sel", "CIv"], "parCol")      # [1,1,1,12] (cols used when top/bot)
    n("Add", ["parRow", "parCol"], "par")      # broadcast -> ONE [1,1,12,12]

    # ---- cyan nearest / farthest from wall (min / max normal among cyan) ----
    n("Greater", ["CY", "halfF"], "cyB")             # bool cyan mask (CY fp32)
    # nrm masked: non-cyan -> +big for the min, -big for the max
    n("Where", ["cyB", "nrm", "big"], "nrm_for_min")
    n("ReduceMin", ["nrm_for_min"], "nmin", axes=[2, 3], keepdims=1)   # scalar
    n("Where", ["cyB", "nrm", "negbig"], "nrm_for_max")
    n("ReduceMax", ["nrm_for_max"], "nmax", axes=[2, 3], keepdims=1)   # scalar
    # par at the nearest cell: nmin is ALWAYS a single cyan cell (verified, no
    # ties) -> par_near is just the masked sum, no division.
    init("zeroH", np.array(0.0, np.float16), np.float16)
    n("Equal", ["nrm_for_min", "nmin"], "atmin")     # only cyan cells can match
    n("Where", ["atmin", "par", "zeroH"], "atmin_par")
    n("ReduceSum", ["atmin_par"], "par_near", axes=[2, 3], keepdims=1)
    # par at the farthest cell: may TIE (symmetric vertex-visible case) -> average.
    n("Equal", ["nrm_for_max", "nmax"], "atmax")
    n("Cast", ["atmax"], "atmax_f", to=H)
    n("Mul", ["atmax_f", "par"], "atmax_par")
    n("ReduceSum", ["atmax_par"], "atmax_par_s", axes=[2, 3], keepdims=1)
    n("ReduceSum", ["atmax_f"], "atmax_cnt", axes=[2, 3], keepdims=1)
    n("Div", ["atmax_par_s", "atmax_cnt"], "par_far")

    # ---- slope (exact +-1) and vertex parallel coordinate ----
    n("Sub", ["par_near", "par_far"], "dpar")
    n("Sub", ["nmin", "nmax"], "dnrm")            # nonzero, |.|>=1
    n("Div", ["dpar", "dnrm"], "slope")           # exact +-1
    n("Sub", ["band", "nmin"], "dnorm_v")         # band - nmin
    n("Mul", ["slope", "dnorm_v"], "slope_step")
    n("Add", ["par_near", "slope_step"], "vpar")  # vertex parallel coord (scalar)
    # vertex normal coord = band (on the wall inner edge)

    # ---- reconstruct vertex (vr,vc) -> a=vr+vc, b=vr-vc ----
    # top:   vr=band,    vc=vpar
    # bottom:vr=11-band, vc=vpar
    # left:  vc=band,    vr=vpar
    # right: vc=11-band, vr=vpar
    n("Sub", ["Wm1", "band"], "bandrev")          # 11 - band
    # vr = topB*band + botB*bandrev + (leftB+rightB)*vpar
    n("Mul", ["topBf", "band"], "vr_t")
    n("Mul", ["botBf", "bandrev"], "vr_b")
    n("Mul", ["lr_sel", "vpar"], "vr_lr")
    n("Add", ["vr_t", "vr_b"], "vr_tb")
    n("Add", ["vr_tb", "vr_lr"], "vr")
    # vc = leftB*band + rightB*bandrev + (topB+botB)*vpar
    n("Mul", ["leftBf", "band"], "vc_l")
    n("Mul", ["rightBf", "bandrev"], "vc_r")
    n("Mul", ["tb_sel", "vpar"], "vc_tb")
    n("Add", ["vc_l", "vc_r"], "vc_lr")
    n("Add", ["vc_lr", "vc_tb"], "vc")
    n("Add", ["vr", "vc"], "aconst")     # a = vr+vc
    n("Sub", ["vr", "vc"], "bconst")     # b = vr-vc

    # ---- diagonal predicate: (RI+CI==a) OR (RI-CI==b) (1-D ramps -> planes) ----
    # All values are exact small fp16 integers, so Equal is safe.
    n("Add", ["RIv", "CIv"], "RpC")    # broadcast -> [1,1,12,12]
    n("Sub", ["RIv", "CIv"], "RmC")
    n("Equal", ["RpC", "aconst"], "onA")
    n("Equal", ["RmC", "bconst"], "onB")
    n("Or", ["onA", "onB"], "ondiag")

    # ---- blocked cells = near side of vertex (normal < band) OR wall ----
    n("Less", ["nrm", "band"], "near_side")          # bool
    n("Greater", ["RE", "halfF"], "wallB")           # bool wall mask (RE fp32)
    n("Or", ["near_side", "wallB"], "blocked")
    n("Not", ["blocked"], "open_cell")
    n("And", ["ondiag", "open_cell"], "greenmask")

    # ---- assemble colour label in uint8 from THREE DISJOINT masks ----
    #   wall(2), cyan(8), green-not-cyan(3) never overlap, so a weighted sum is
    #   exact:  L = 2*wall + 8*cyan + 3*(green AND NOT cyan).  All uint8 (144 B).
    # priority red > cyan > green > bg, via 3 chained Where (uint8 throughout).
    init("z_u8", np.array(0, np.uint8), np.uint8)
    init("two_u8", np.array(2, np.uint8), np.uint8)
    init("eight_u8", np.array(8, np.uint8), np.uint8)
    init("three_u8", np.array(3, np.uint8), np.uint8)
    n("Where", ["greenmask", "three_u8", "z_u8"], "lab_g")   # green->3 (uint8)
    n("Where", ["cyB", "eight_u8", "lab_g"], "lab_c")        # cyan->8
    n("Where", ["wallB", "two_u8", "lab_c"], "L12")          # wall->2 (uint8)

    # ---- pad 12x12 -> 30x30 with sentinel (off-grid = all channels OFF) ----
    n("Pad", ["L12", "padpads", "sent_u8"], "L", mode="constant")  # [1,1,30,30]
    n("Equal", ["L", "chan_u8"], "output")             # [1,10,30,30] BOOL (free)

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task119", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
