"""task370 (ARC-AGI e8dc4411) — diagonal sprite-march replication.

Exact rule (from the generator)
-------------------------------
The input is a background-filled grid (bg in 1..9) containing:
  * a single black (colour 0) sprite occupying an L×L bounding box at top-left
    corner (r0,c0), L in {3,4}, drawn identically in input and output;
  * exactly ONE coloured "hint" pixel (colour `color` in 1..9, color != bg)
    placed at offset (dr_h, dc_h) from (r0,c0), where each of dr_h, dc_h is
    either +L  or  -1.

The output keeps the black sprite and additionally stamps copies of the sprite
in `color`, marching from the sprite by a step (sr, sc) with sr,sc in {+L,-L}:
  sr = +L if dr_h > 0 else -L   (dr_h in {+L, -1})
  sc = +L if dc_h > 0 else -L
Copy k (k = 1, 2, …) is the sprite translated by (k*sr, k*sc); copies tile the
diagonal band contiguously until they leave the grid.

Per-cell characterisation (vectorises; verified 300/300 vs the generator):
  let dr = r - r0,  dc = c - c0
      a  = dr mod L,           b  = dc mod L          (sprite-local offset)
      kr = (dr - a) / sr,      kc = (dc - b) / sc     (copy index per axis)
  cell (r,c) is coloured  iff  kr == kc  AND  kr >= 1  AND  spr[a,b]
  where spr[a,b] = black-sprite mask (1 if the (a,b) sprite cell is black).
  output[r,c] = 0     if the cell is in the original black sprite (ch0==1)
              = color if coloured (and in-grid)
              = bg    otherwise (in-grid)
              = empty (all channels 0) off-grid.

Encoding facts used
-------------------
Input is one-hot [1,10,30,30]; off-grid cells are ALL-zero (no channel set).
colour 0 is used ONLY by the sprite, so channel-0 cleanly marks the black
sprite.  The hint colour channel contains exactly one set pixel; bg fills the
rest of the grid (count >> 1), so the hint colour is the unique channel in
1..9 whose set-count == 1.

Tier / memory
-------------
Non-separable (the kr==kc diagonal couples rows and cols) and non-local, so
Tier S / A are impossible — this is the Tier-B label-map form: build a single
uint8 label plane L[1,1,30,30] (output colour per cell, sentinel 255 off-grid /
bg-via-nomatch) and emit  output = Equal(L, arange(10))  as BOOL.  a, b, kr, kc
are computed as 1-D vectors ([1,1,30,1] / [1,1,1,30]); only a few [1,1,30,30]
planes are materialised (ch0 fp32 + the coupling/label planes).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

N = 30
LMAX = 4


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

    # ===================================================================
    # 0.  channel-0 plane (black sprite) as float ----------------------
    # ===================================================================
    W_ch0 = np.zeros([1, 10, 1, 1], np.float32)
    W_ch0[0, 0, 0, 0] = 1.0
    init("W_ch0", W_ch0, np.float32)
    n("Conv", ["input", "W_ch0"], "ch0")          # [1,1,30,30] f32

    # row-has-black / col-has-black (1-D)
    n("ReduceMax", ["ch0"], "rh", axes=[3], keepdims=1)   # [1,1,30,1]
    n("ReduceMax", ["ch0"], "cw", axes=[2], keepdims=1)   # [1,1,1,30]

    # ---- r0, r1, c0, c1 via masked arange ----------------------------
    ar_r = np.arange(N, dtype=np.float32).reshape(1, 1, N, 1)
    ar_c = np.arange(N, dtype=np.float32).reshape(1, 1, 1, N)
    init("ar_r", ar_r, np.float32)
    init("ar_c", ar_c, np.float32)
    init("BIG", np.array(99.0, np.float32), np.float32)
    init("NEG", np.array(-1.0, np.float32), np.float32)

    # r0 = min row with black via boolean masks
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["rh", "half"], "rh_b")           # [1,1,30,1] bool
    n("Greater", ["cw", "half"], "cw_b")           # [1,1,1,30] bool

    n("Where", ["rh_b", "ar_r", "BIG"], "r_lo_w")  # [1,1,30,1]
    n("ReduceMin", ["r_lo_w"], "r0", axes=[2], keepdims=1)   # [1,1,1,1]
    n("Where", ["rh_b", "ar_r", "NEG"], "r_hi_w")
    n("ReduceMax", ["r_hi_w"], "r1", axes=[2], keepdims=1)   # [1,1,1,1]
    n("Where", ["cw_b", "ar_c", "BIG"], "c_lo_w")
    n("ReduceMin", ["c_lo_w"], "c0", axes=[3], keepdims=1)
    n("Where", ["cw_b", "ar_c", "NEG"], "c_hi_w")
    n("ReduceMax", ["c_hi_w"], "c1", axes=[3], keepdims=1)

    init("one", np.array(1.0, np.float32), np.float32)
    n("Sub", ["r1", "r0"], "rspan")
    n("Add", ["rspan", "one"], "Lr")               # [1,1,1,1] = L (rows)
    # L is square; use Lr as L.
    L = "Lr"

    # ===================================================================
    # 1.  hint colour + position --------------------------------------
    # ===================================================================
    # per-channel set count over (h,w): [1,10,1,1]
    n("ReduceSum", ["input"], "ch_cnt", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    # colour channel = the channel whose count == 1.  channel-0 may also be a
    # small count but is colour 0 (never the hint colour, which is in 1..9 and
    # has count exactly 1, whereas bg count >> 1).  We must avoid matching ch0
    # if the sprite happened to have count 1 (impossible: smallest sprite has
    # >=? cells) — but to be safe restrict to channels 1..9 by zeroing ch0.
    mask19 = np.ones([1, 10, 1, 1], np.float32)
    mask19[0, 0, 0, 0] = 0.0
    init("mask19", mask19, np.float32)
    n("Mul", ["ch_cnt", "mask19"], "cnt19")        # [1,10,1,1]
    # is_color = (cnt19 == 1)
    init("one_b", np.array(1.0, np.float32), np.float32)
    n("Equal", ["cnt19", "one_b"], "iscolor_b")    # [1,10,1,1] bool
    n("Cast", ["iscolor_b"], "iscolor", to=F)      # [1,10,1,1]
    # color index = sum(channel_idx * iscolor)
    chidx = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("chidx", chidx, np.float32)
    n("Mul", ["iscolor", "chidx"], "cidx_m")
    n("ReduceSum", ["cidx_m"], "color", axes=[1], keepdims=1)  # [1,1,1,1]

    # hint plane = the colour channel of the input = Conv(input, iscolor as 1x1)
    # iscolor is [1,10,1,1]; reshape to weight [1,10,1,1] and Conv.
    n("Conv", ["input", "iscolor"], "hintp")       # [1,1,30,30] f32 (single 1)
    n("ReduceMax", ["hintp"], "hrh", axes=[3], keepdims=1)   # [1,1,30,1]
    n("ReduceMax", ["hintp"], "hcw", axes=[2], keepdims=1)   # [1,1,1,30]
    # hr = sum(ar_r * hrh) ; hc = sum(ar_c * hcw)
    n("Mul", ["ar_r", "hrh"], "hr_m")
    n("ReduceSum", ["hr_m"], "hr", axes=[2], keepdims=1)     # [1,1,1,1]
    n("Mul", ["ar_c", "hcw"], "hc_m")
    n("ReduceSum", ["hc_m"], "hc", axes=[3], keepdims=1)     # [1,1,1,1]

    # dr_h = hr - r0 ; sr = +L if dr_h>0 else -L
    n("Sub", ["hr", "r0"], "drh")
    n("Sub", ["hc", "c0"], "dch")
    init("zero", np.array(0.0, np.float32), np.float32)
    n("Greater", ["drh", "zero"], "drh_pos")       # bool
    n("Greater", ["dch", "zero"], "dch_pos")
    # sr = where(pos, L, -L)
    n("Neg", [L], "negL")
    n("Where", ["drh_pos", L, "negL"], "sr")       # [1,1,1,1]
    n("Where", ["dch_pos", L, "negL"], "sc")

    # ===================================================================
    # 2.  spr[a,b] sprite mask (cropped to L×L at (r0,c0)) -------------
    # ===================================================================
    # We need spr as a flat 16-vector spr16[a*4+b], a,b in 0..3.
    # spr[a,b] = ch0[r0+a, c0+b].  Gather a fixed 4×4 block by computing
    # absolute indices then Gather from flattened ch0.
    # flat index = (r0+a)*30 + (c0+b), a,b in 0..3 (LMAX).
    init("flat900", np.array([900], np.int64), np.int64)
    n("Reshape", ["ch0", "flat900"], "ch0f")       # [900] f32
    # build a/b grids of LMAX×LMAX
    aa = np.arange(LMAX, dtype=np.float32).reshape(1, 1, LMAX, 1)
    bb = np.arange(LMAX, dtype=np.float32).reshape(1, 1, 1, LMAX)
    init("aa", aa, np.float32)
    init("bb", bb, np.float32)
    # abs row = r0 + aa  (broadcast [1,1,1,1]+[1,1,4,1] -> [1,1,4,1])
    n("Add", ["r0", "aa"], "absr")                 # [1,1,4,1]
    n("Add", ["c0", "bb"], "absc")                 # [1,1,1,4]
    init("thirty", np.array(30.0, np.float32), np.float32)
    n("Mul", ["absr", "thirty"], "absr30")         # [1,1,4,1]
    n("Add", ["absr30", "absc"], "spr_idx_f")      # [1,1,4,4] flat idx
    n("Cast", ["spr_idx_f"], "spr_idx", to=I64)    # [1,1,4,4]
    init("flat16", np.array([16], np.int64), np.int64)
    n("Reshape", ["spr_idx", "flat16"], "spr_idx16")  # [16]
    n("Gather", ["ch0f", "spr_idx16"], "spr16")    # [16] f32 (sprite mask)

    # ===================================================================
    # 3.  per-axis a, b, kr, kc  (1-D vectors) -------------------------
    # ===================================================================
    # dr = ar_r - r0   [1,1,30,1]   ;  dc = ar_c - c0  [1,1,1,30]
    n("Sub", ["ar_r", "r0"], "dr")
    n("Sub", ["ar_c", "c0"], "dc")
    # a = dr mod L  (>=0 since L>0 and python/onnx Mod with fmod=0 gives
    # result with sign of divisor when ints; use float Mod fmod=0 -> sign of
    # divisor (L>0) -> non-negative).  ONNX Mod requires fmod for floats.
    n("Mod", ["dr", L], "a", fmod=1)               # [1,1,30,1] but fmod sign=dividend
    # fmod gives sign of dividend; we need true modulo (0..L-1). Fix below.
    n("Mod", ["dc", L], "b", fmod=1)

    # true non-negative modulo: tmod = fmod(dr,L); a = tmod + (tmod<0 ? L : 0)
    n("Less", ["a", "zero"], "a_neg")
    n("Where", ["a_neg", L, "zero"], "a_fix")
    n("Add", ["a", "a_fix"], "amod")               # [1,1,30,1] in 0..L-1
    n("Less", ["b", "zero"], "b_neg")
    n("Where", ["b_neg", L, "zero"], "b_fix")
    n("Add", ["b", "b_fix"], "bmod")               # [1,1,1,30]

    # kr = (dr - amod) / sr ; kc = (dc - bmod)/sc
    n("Sub", ["dr", "amod"], "dr_a")
    n("Div", ["dr_a", "sr"], "kr")                 # [1,1,30,1]
    n("Sub", ["dc", "bmod"], "dc_b")
    n("Div", ["dc_b", "sc"], "kc")                 # [1,1,1,30]

    # ===================================================================
    # 4.  coupling: kr == kc  AND  kr >= 1  AND  spr[amod,bmod] ---------
    # ===================================================================
    # kr==kc : broadcast [1,1,30,1] vs [1,1,1,30] -> [1,1,30,30] bool
    n("Equal", ["kr", "kc"], "keq")                # [1,1,30,30] bool (900B)
    # kr>=1 -> Greater(kr,0.5) [1,1,30,1]
    init("half2", np.array(0.5, np.float32), np.float32)
    n("Greater", ["kr", "half2"], "kr_ge1")        # [1,1,30,1] bool
    n("And", ["keq", "kr_ge1"], "k_ok")            # [1,1,30,30] bool

    # spr[amod,bmod] : flat index = amod*4 + bmod (broadcast -> [1,1,30,30])
    init("four", np.array(4.0, np.float32), np.float32)
    n("Mul", ["amod", "four"], "amod4")            # [1,1,30,1]
    n("Add", ["amod4", "bmod"], "sab_idx_f")       # [1,1,30,30]
    n("Cast", ["sab_idx_f"], "sab_idx", to=I64)    # [1,1,30,30]
    init("flat900b", np.array([900], np.int64), np.int64)
    n("Reshape", ["sab_idx", "flat900b"], "sab_idx_flat")  # [900]
    n("Gather", ["spr16", "sab_idx_flat"], "sprab_flat")   # [900] f32
    init("shp30", np.array([1, 1, 30, 30], np.int64), np.int64)
    n("Reshape", ["sprab_flat", "shp30"], "sprab")         # [1,1,30,30] f32
    n("Greater", ["sprab", "half"], "sprab_b")             # bool

    n("And", ["k_ok", "sprab_b"], "colored")       # [1,1,30,30] bool (coloured)

    # ===================================================================
    # 5.  assemble label plane L[1,1,30,30] uint8 ----------------------
    # ===================================================================
    # ch0 mask (sprite) -> label 0
    n("Greater", ["ch0", "half"], "ch0_b")         # [1,1,30,30] bool
    # in-grid mask = any channel set = ReduceMax(input over channels)
    n("ReduceMax", ["input"], "ingrid_f", axes=[1], keepdims=1)  # [1,1,30,30]
    n("Greater", ["ingrid_f", "half"], "ingrid_b")

    # color as scalar plane (float). Start label = bg sentinel; we encode via
    # value choices then Equal(arange).  Build label value:
    #   off-grid    -> 255 (sentinel, matches no channel)
    #   sprite      -> 0
    #   coloured    -> color
    #   else in-grid-> bg  (we need bg colour index)
    # bg index = the in-grid background colour. bg = argmax channel of a known
    # bg cell, e.g. corner (0,0) is bg ONLY if in-grid; safer: bg is the
    # channel (1..9) with the LARGEST count.
    n("ReduceMax", ["ch_cnt"], "bgcnt", axes=[1], keepdims=1)    # [1,1,1,1] max count
    n("Equal", ["ch_cnt", "bgcnt"], "isbg_b")      # [1,10,1,1] bool (argmax)
    n("Cast", ["isbg_b"], "isbg", to=F)
    n("Mul", ["isbg", "chidx"], "bgidx_m")
    n("ReduceMax", ["bgidx_m"], "bgcolor", axes=[1], keepdims=1)  # [1,1,1,1]

    # Build label by nested Where (all broadcasts to [1,1,30,30]):
    #  lab = where(sprite, 0, where(colored, color, bg))   [in-grid region]
    #  lab = where(ingrid, lab, 255)
    init("zero255", np.array(255.0, np.float32), np.float32)
    init("zero_lab", np.array(0.0, np.float32), np.float32)
    n("Where", ["colored", "color", "bgcolor"], "lab_cb")  # [1,1,30,30]
    n("Where", ["ch0_b", "zero_lab", "lab_cb"], "lab_in")  # sprite -> 0
    n("Where", ["ingrid_b", "lab_in", "zero255"], "lab_f") # off-grid -> 255
    n("Cast", ["lab_f"], "L", to=U8)               # [1,1,30,30] uint8 (900B)

    # ===================================================================
    # 6.  output = Equal(L, arange) BOOL -------------------------------
    # ===================================================================
    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")            # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task370", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
