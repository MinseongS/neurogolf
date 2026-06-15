"""task370 (ARC-AGI e8dc4411) — diagonal sprite-march replication.

Exact rule (reverse-engineered from the data; verified 262/262 arc-gen + 4/4
human train+test)
------------------------------------------------------------------------------
The input is a background-filled grid (bg in 1..9) containing:
  * a single black (colour 0) sprite occupying an L×L bounding box whose
    top-left corner is (r0,c0), L in {3,4,5}; the sprite is drawn identically
    in input and output (it is the "source");
  * exactly ONE coloured "hint" pixel (colour `color` in 1..9, color != bg) at
    offset (hofr,hofc) = (hr-r0, hc-c0) from the sprite corner.

The output keeps the source sprite and additionally stamps copies of it in
`color`, marching by a DIAGONAL step (sr,sc) with |sr|==|sc|==m:
  sgr = +1 if hofr > 0 else -1 ;  sgc = +1 if hofc > 0 else -1
  m   = the LARGEST m in 1..L for which sprite-cell (hofr - sgr*m, hofc - sgc*m)
        is black (i.e. the hint lies on copy #1, and we take the largest such
        diagonal step). sr = sgr*m, sc = sgc*m.
Copy k (k = 1,2,…) is the sprite translated by (k*sr, k*sc); copies may overlap
(when m < L) and continue until they leave the grid.

Per-cell characterisation that vectorises (verified 262/262 + 4/4):
  let u = r - r0, v = c - c0.  Cell (r,c) is coloured iff there exists a sprite
  row-offset a in 0..L-1 with
        k  = (u - a)/sr   an integer >= 1            (depends only on r)
        b  = v - k*sc     in 0..L-1  and  spr[a,b] black.
  We OR this over the (<=5) possible a values.  `k` is a 1-D row vector
  [1,1,30,1]; only the per-a `b`/`spr[a,b]` planes are 2-D.

output[r,c] = 0     where the source sprite is black (ch0==1)
            = color where coloured
            = bg    elsewhere in-grid
            = empty off-grid (all channels 0).

Encoding facts
--------------
Input is one-hot [1,10,30,30]; off-grid cells are ALL-zero.  Colour 0 is used
ONLY by the source sprite so channel-0 cleanly marks it.  The hint colour
channel has exactly one set pixel; bg fills the rest of the grid (count >> 1),
so the hint colour is the unique channel in 1..9 whose set-count == 1, and bg is
the channel in 1..9 with the largest count.

Tier / memory
-------------
Non-separable (the march couples rows & cols) and non-local → Tier-B label map:
emit a single uint8 label plane L[1,1,30,30] (output colour per cell, sentinel
255 off-grid) and  output = Equal(L, arange(10))  as BOOL.  The dominant
intermediates are a handful of [1,1,30,30] planes (ch0 fp32 + per-a coloured
planes + the label); per-axis quantities (u, k) stay 1-D.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

N = 30
LMAX = 5
# Padded sprite-vector layout. Each sprite row a occupies the slot
# [PAD + a*S, PAD + a*S + L).  The per-cell flat index is PAD + a*S + (v-kr*sc);
# the addend x = v-kr*sc is bounded by |x| <= 29 + 5*33 < 195 (v in [-29,29],
# |kr|<=33, |sc|<=5), so with S = 400 >> 195 the index can never spill from one
# row's slot into another's data, and with PAD = 200 it always stays in
# [0, SPRVEC_LEN).  => no per-cell range check AND no clip needed (both provably
# safe), removing the dominant per-a bool/fp16 planes.
PAD = 205
SPRVEC_S = 205
SPRVEC_LEN = PAD + LMAX * SPRVEC_S + PAD     # 1435 (S>194+L, PAD>194: safe)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL
    I64 = TensorProto.INT64

    # shared constants -------------------------------------------------
    init("half", np.array(0.5, np.float32), np.float32)
    init("zero", np.array(0.0, np.float32), np.float32)
    init("one", np.array(1.0, np.float32), np.float32)
    init("BIG", np.array(99.0, np.float32), np.float32)
    init("NEG", np.array(-1.0, np.float32), np.float32)
    ar_r = np.arange(N, dtype=np.float32).reshape(1, 1, N, 1)
    ar_c = np.arange(N, dtype=np.float32).reshape(1, 1, 1, N)
    init("ar_r", ar_r, np.float32)
    init("ar_c", ar_c, np.float32)
    chidx = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("chidx", chidx, np.float32)

    # ===== 0. channel-0 (source sprite) plane =========================
    W_ch0 = np.zeros([1, 10, 1, 1], np.float32)
    W_ch0[0, 0, 0, 0] = 1.0
    init("W_ch0", W_ch0, np.float32)
    n("Conv", ["input", "W_ch0"], "ch0")               # [1,1,30,30] f32

    n("ReduceMax", ["ch0"], "rh", axes=[3], keepdims=1)   # [1,1,30,1]
    n("ReduceMax", ["ch0"], "cw", axes=[2], keepdims=1)   # [1,1,1,30]
    n("Greater", ["rh", "half"], "rh_b")
    n("Greater", ["cw", "half"], "cw_b")

    # r0,c0,r1 via masked arange
    n("Where", ["rh_b", "ar_r", "BIG"], "r_lo_w")
    n("ReduceMin", ["r_lo_w"], "r0", axes=[2], keepdims=1)   # [1,1,1,1]
    n("Where", ["rh_b", "ar_r", "NEG"], "r_hi_w")
    n("ReduceMax", ["r_hi_w"], "r1", axes=[2], keepdims=1)
    n("Where", ["cw_b", "ar_c", "BIG"], "c_lo_w")
    n("ReduceMin", ["c_lo_w"], "c0", axes=[3], keepdims=1)
    n("Sub", ["r1", "r0"], "rspan")
    n("Add", ["rspan", "one"], "Lr")                        # L (square)
    L = "Lr"

    # ===== 1. hint colour + position ==================================
    n("ReduceSum", ["input"], "ch_cnt", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    mask19 = np.ones([1, 10, 1, 1], np.float32)
    mask19[0, 0, 0, 0] = 0.0
    init("mask19", mask19, np.float32)
    n("Mul", ["ch_cnt", "mask19"], "cnt19")
    n("Equal", ["cnt19", "one"], "iscolor_b")               # count==1 channel
    n("Cast", ["iscolor_b"], "iscolor", to=F)               # [1,10,1,1]
    n("Mul", ["iscolor", "chidx"], "cidx_m")
    n("ReduceSum", ["cidx_m"], "color", axes=[1], keepdims=1)  # [1,1,1,1]

    # hint plane via a 1x1 Conv selecting the colour channel, then 1-D reduces.
    n("Conv", ["input", "iscolor"], "hintp")                # [1,1,30,30] f32
    n("ReduceMax", ["hintp"], "hrh", axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["hintp"], "hcw", axes=[2], keepdims=1)  # [1,1,1,30]
    n("Mul", ["ar_r", "hrh"], "hr_m")
    n("ReduceSum", ["hr_m"], "hr", axes=[2], keepdims=1)    # [1,1,1,1]
    n("Mul", ["ar_c", "hcw"], "hc_m")
    n("ReduceSum", ["hc_m"], "hc", axes=[3], keepdims=1)    # [1,1,1,1]

    n("Sub", ["hr", "r0"], "hofr")                          # [1,1,1,1]
    n("Sub", ["hc", "c0"], "hofc")
    n("Greater", ["hofr", "zero"], "hofr_pos")
    n("Greater", ["hofc", "zero"], "hofc_pos")
    n("Where", ["hofr_pos", "one", "NEG"], "sgr")           # +1 / -1
    n("Where", ["hofc_pos", "one", "NEG"], "sgc")

    # ===== 2. sprite mask spr[LMAX,LMAX] (flat 25) ====================
    init("flat900", np.array([900], np.int64), np.int64)
    n("Cast", ["ch0"], "ch0u", to=U8)                       # [1,1,30,30] uint8
    n("Reshape", ["ch0u", "flat900"], "ch0f")               # [900] uint8
    aa = np.arange(LMAX, dtype=np.float32).reshape(1, 1, LMAX, 1)
    bb = np.arange(LMAX, dtype=np.float32).reshape(1, 1, 1, LMAX)
    init("aa", aa, np.float32)
    init("bb", bb, np.float32)
    n("Add", ["r0", "aa"], "absr")                          # [1,1,LMAX,1]
    n("Add", ["c0", "bb"], "absc")                          # [1,1,1,LMAX]
    init("thirty", np.array(30.0, np.float32), np.float32)
    n("Mul", ["absr", "thirty"], "absr30")
    n("Add", ["absr30", "absc"], "spr_idx_f")               # [1,1,LMAX,LMAX]
    # clamp the gather index into [0,899] (off-grid cells -> 0, value 0 anyway)
    init("z0", np.array(0.0, np.float32), np.float32)
    init("max899", np.array(899.0, np.float32), np.float32)
    n("Clip", ["spr_idx_f", "z0", "max899"], "spr_idx_c")
    n("Cast", ["spr_idx_c"], "spr_idx", to=I64)
    init("flat25", np.array([LMAX * LMAX], np.int64), np.int64)
    n("Reshape", ["spr_idx", "flat25"], "spr_idx25")
    n("Gather", ["ch0f", "spr_idx25"], "spr25")             # [25] uint8 sprite mask

    # --- padded sprite vector for range-check-free gathers ----------------
    # Layout: slot PAD + a*S + col holds spr[a,col]; everything else is 0.
    # With S > 2*30 and a leading/trailing PAD, any column index b in
    # [-30, 30) lands either on the correct cell or in a zero gap, so the
    # per-a gather needs NO explicit b-range check.
    # Append a 0 sentinel to spr25 (-> spr26) and scatter via a fixed map.
    init("z_sent", np.array([0], np.uint8), np.uint8)
    n("Concat", ["spr25", "z_sent"], "spr26", axis=0)       # [26] uint8
    scatter = np.full(SPRVEC_LEN, LMAX * LMAX, dtype=np.int32)  # default -> sentinel 0
    for a in range(LMAX):
        for col in range(LMAX):
            scatter[PAD + a * SPRVEC_S + col] = a * LMAX + col
    init("scatter", scatter, np.int32)                      # [SPRVEC_LEN] params
    n("Gather", ["spr26", "scatter"], "spr_vec")            # [SPRVEC_LEN] uint8

    # ===== 3. step magnitude m ========================================
    # for mm = 1..LMAX: cell (hofr - sgr*mm, hofc - sgc*mm); spr value there.
    # pick largest mm with spr value > 0.5.
    mm_v = np.arange(1, LMAX + 1, dtype=np.float32).reshape(1, 1, LMAX, 1)
    init("mm_v", mm_v, np.float32)                          # [1,1,LMAX,1]
    n("Mul", ["sgr", "mm_v"], "sgr_mm")                     # [1,1,LMAX,1]
    n("Sub", ["hofr", "sgr_mm"], "ma")                      # row offset per mm
    n("Mul", ["sgc", "mm_v"], "sgc_mm")
    n("Sub", ["hofc", "sgc_mm"], "mb")                      # col offset per mm
    # valid iff 0<=ma<LMAX and 0<=mb<LMAX (spr is 0 outside true L anyway)
    init("lmaxf", np.array(float(LMAX), np.float32), np.float32)
    n("GreaterOrEqual", ["ma", "zero"], "ma_ge0") if False else None
    # use Greater(>= via >-0.5) to stay opset-11 safe
    init("nhalf", np.array(-0.5, np.float32), np.float32)
    init("lmaxm05", np.array(float(LMAX) - 0.5, np.float32), np.float32)
    n("Greater", ["ma", "nhalf"], "ma_ge0")
    n("Less", ["ma", "lmaxm05"], "ma_lt")
    n("And", ["ma_ge0", "ma_lt"], "ma_ok")
    n("Greater", ["mb", "nhalf"], "mb_ge0")
    n("Less", ["mb", "lmaxm05"], "mb_lt")
    n("And", ["mb_ge0", "mb_lt"], "mb_ok")
    n("And", ["ma_ok", "mb_ok"], "ab_ok")                   # [1,1,LMAX,1]? mb is [1,1,LMAX,1]
    # ma,mb both [1,1,LMAX,1] (mm along axis2) -> ab_ok [1,1,LMAX,1]
    # gather spr value at (ma,mb): idx = ma*LMAX + mb, clamp to [0,24]
    n("Mul", ["ma", "lmaxf"], "ma_lm")
    n("Add", ["ma_lm", "mb"], "midx_f")                     # [1,1,LMAX,1]
    init("z0b", np.array(0.0, np.float32), np.float32)
    init("max24", np.array(float(LMAX * LMAX - 1), np.float32), np.float32)
    n("Clip", ["midx_f", "z0b", "max24"], "midx_c")
    n("Cast", ["midx_c"], "midx", to=I64)
    init("mflat", np.array([LMAX], np.int64), np.int64)
    n("Reshape", ["midx", "mflat"], "midx1")                # [LMAX]
    n("Gather", ["spr25", "midx1"], "msprv")                # [LMAX] uint8
    init("msh", np.array([1, 1, LMAX, 1], np.int64), np.int64)
    n("Reshape", ["msprv", "msh"], "msprv4")                # [1,1,LMAX,1] uint8
    n("Cast", ["msprv4"], "mspr_b", to=B)                   # 0/1 -> bool
    n("And", ["mspr_b", "ab_ok"], "m_hit")                  # [1,1,LMAX,1] bool
    # m = max mm over hits; where(hit, mm_v, 0) reduce-max
    n("Where", ["m_hit", "mm_v", "zero"], "m_cand")
    n("ReduceMax", ["m_cand"], "m", axes=[2], keepdims=1)   # [1,1,1,1]
    n("Mul", ["sgr", "m"], "sr")                            # [1,1,1,1]
    n("Mul", ["sgc", "m"], "sc")

    # ===== 4. coloured mask : OR over a in 0..LMAX-1 ==================
    # u = ar_r - r0 [1,1,30,1] ; v = ar_c - c0 [1,1,1,30]
    n("Sub", ["ar_r", "r0"], "u")
    n("Sub", ["ar_c", "c0"], "v")

    I32 = TensorProto.INT32
    n("Cast", ["v"], "v_i", to=I32)                         # [1,1,1,30] int32
    for a in range(LMAX):
        init(f"base_i_{a}", np.array(PAD + a * SPRVEC_S, np.int32), np.int32)
    colored_terms = []
    for a in range(LMAX):
        # --- all per-row (1-D [1,1,30,1]) work in fp32 ---------------
        init(f"a_{a}", np.array(float(a), np.float32), np.float32)
        n("Sub", ["u", f"a_{a}"], f"knum_{a}")              # [1,1,30,1]
        n("Div", [f"knum_{a}", "sr"], f"k_{a}")             # [1,1,30,1]
        n("Round", [f"k_{a}"], f"kr_{a}")
        n("Mul", [f"kr_{a}", "sr"], f"krsr_{a}")
        n("Equal", [f"krsr_{a}", f"knum_{a}"], f"kint_{a}")  # k integer
        n("Greater", [f"kr_{a}", "half"], f"kge1_{a}")      # k>=1
        n("And", [f"kint_{a}", f"kge1_{a}"], f"kok_{a}")    # [1,1,30,1] bool
        # padded-layout index: idx = (PAD + a*S + v) - kr*sc.  Out-of-range b
        # lands in a zero gap of spr_vec -> NO range check.  Build the index in
        # int32 directly: the only 2-D plane is the int32 result (no fp16 copy).
        n("Mul", [f"kr_{a}", "sc"], f"krsc_{a}")            # [1,1,30,1] fp32
        n("Cast", [f"krsc_{a}"], f"krsc_i_{a}", to=I32)     # [1,1,30,1] int32
        n("Add", ["v_i", f"base_i_{a}"], f"vbase_i_{a}")    # [1,1,1,30] int32
        n("Sub", [f"vbase_i_{a}", f"krsc_i_{a}"], f"sidx_{a}")  # [1,1,30,30] int32
        n("Gather", ["spr_vec", f"sidx_{a}"], f"sprab_{a}")  # [1,1,30,30] uint8
        n("Cast", [f"sprab_{a}"], f"sprab_b_{a}", to=B)
        # term(bool) = (spr>0) AND kok(row)
        n("And", [f"sprab_b_{a}", f"kok_{a}"], f"term_{a}")  # [1,1,30,30] bool
        colored_terms.append(f"term_{a}")

    # OR all bool terms
    cur = colored_terms[0]
    for j, t in enumerate(colored_terms[1:], 1):
        out = "colored" if j == len(colored_terms) - 1 else f"or_{j}"
        n("Or", [cur, t], out)
        cur = out
    colored = cur

    # ===== 5. label plane =============================================
    n("Greater", ["ch0", "half"], "ch0_b")                  # source sprite
    n("ReduceMax", ["input"], "ingrid_f", axes=[1], keepdims=1)
    n("Greater", ["ingrid_f", "half"], "ingrid_b")
    # bg colour = channel in 1..9 with max count
    n("ReduceMax", ["cnt19"], "bgcnt", axes=[1], keepdims=1)
    n("Equal", ["cnt19", "bgcnt"], "isbg_b")                # [1,10,1,1]
    n("Cast", ["isbg_b"], "isbg", to=F)
    n("Mul", ["isbg", "chidx"], "bgidx_m")
    n("ReduceMax", ["bgidx_m"], "bgcolor", axes=[1], keepdims=1)

    # cast scalar colours to uint8 so the [1,1,30,30] Where planes are uint8
    n("Cast", ["color"], "color_u", to=U8)                  # [1,1,1,1] uint8
    n("Cast", ["bgcolor"], "bgcolor_u", to=U8)
    init("v255u", np.array(255, np.uint8), np.uint8)
    init("v0u", np.array(0, np.uint8), np.uint8)
    n("Where", [colored, "color_u", "bgcolor_u"], "lab_cb")  # [1,1,30,30] uint8
    n("Where", ["ch0_b", "v0u", "lab_cb"], "lab_in")        # source -> 0
    n("Where", ["ingrid_b", "lab_in", "v255u"], "Lp")       # off-grid -> 255

    # ===== 6. output = Equal(L, arange) BOOL ==========================
    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["Lp", "chan"], "output")                    # [1,10,30,30] BOOL

    nodes_clean = [nd for nd in nodes if nd is not None]
    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes_clean, "task370", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
