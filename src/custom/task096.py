"""Task 096 (4290ef0e): clipped-symmetric concentric-ring reassembly.

Rule (from ARC-GEN generator): input H*W grid (13..19), bg = most-frequent colour.
K (=4..6) non-bg colours each own a UNIQUE ring index idx in 0..K-1 (a permutation).
Ring idx draws its colour at the 4 corners (+-idx,+-idx) about a per-shape RANDOM
centre, each corner with two inward arms of length L_idx; shapes are scattered and
CLIPPED at the grid edges (generator guarantees >=2 quadrants drawn). The output is
a (2K-1)^2 concentric reassembly: for offset (a,b) from the centre (K-1,K-1),
m=max(|a|,|b|), n=min(|a|,|b|): out = colours[m] iff (m<K and n >= m-L_m+1) else bg.

Recovery (EXACT, fresh 0-err over 2100 instances):
  For colour channel k let mask=input[:,k], tot=#pixels, ingrid=#in-grid cells.
  A type t=(idx,L) (12 valid pairs) places exactly iff some translation p has
  cm(p)=#stamp-cells-over-k = tot AND og(p)=#in-grid-stamp-cells = cm.  Folded:
  cdo = Conv(ingrid - 2*mask, K_t)  =>  og - 2*cm ,
  dist(t) = tot + min_p cdo(p) ;  dist==0  <=>  type t fits the clipped survivor.
  A single pixel (idx0) fits EVERY type, and a clipped large stamp can fit a
  smaller one, so the TRUE type is the MIN-idx among the dist==0 matches
  (ArgMax over the idx-ordered type axis returns the first True). bg never fits
  (tot >= ~100 >> 36 = max kernel cells, so dist>0).

  The matched-filter conv plane [10,12,30,30] is built and reduced in FP16 (the
  counts are small integers, fp16-exact; ORT keeps the conv fp16 under
  ORT_DISABLE_ALL on the current build) => 86.4KB instead of fp32 173KB.

Synthesis: scatter (k, L_k) by recovered idx into ring vectors of length 6, then
gather the closed-form colour-index over an 11x11 centred canvas, crop/shift to
the (2K-1)^2 grid at top-left, Pad sentinel, Equal -> BOOL one-hot.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
I64 = TensorProto.INT64

# idx0 (single pixel) is detected separately via tot==1 (the >=2-quadrant rule
# guarantees idx>=1 leaves >=2 pixels), so its kernel is omitted from the conv.
TYPES = [(1, 2), (2, 2), (3, 2), (3, 3), (4, 2),
         (4, 3), (4, 4), (5, 2), (5, 3), (5, 4), (5, 5)]
KS = 11


def _kernel(idx, L):
    g = np.zeros((KS, KS), np.float32)
    cen = KS // 2
    for r, c in [(-idx, -idx), (-idx, idx), (idx, -idx), (idx, idx)]:
        for i in range(L):
            for dr, dc in [(r, c + (i if c < 0 else -i)),
                           (r + (i if r < 0 else -i), c)]:
                g[cen + dr, cen + dc] = 1.0
    return g


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    NT = len(TYPES)
    kern = np.stack([_kernel(*t) for t in TYPES]).reshape(NT, 1, KS, KS)

    # ---------- per-channel matched filter (cropped to the active region) ----------
    # The grid sits top-left, side <= 19; stamp centres are in-grid for the random
    # generator but the hand-authored ARC examples place centres up to 1 cell off
    # the left/top/bottom edge.  Crop the input to a WORK x WORK window and let the
    # conv padding reach the off-grid centres, keeping the dominant conv plane small.
    WORK = 19
    # Crop the 10-ch input to [0:WORK,0:WORK] and cast to fp16 ONCE; all downstream
    # full-window arithmetic is fp16 (values are tiny integers / {-1,0,1}, exact).
    init("s0", np.array([0, 0], np.int64), np.int64)
    init("sW", np.array([WORK, WORK], np.int64), np.int64)
    init("ax23", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "s0", "sW", "ax23"], "inw32")         # [1,10,W,W] fp32
    n("Cast", ["inw32"], "inwf", to=F16)                       # [1,10,W,W] fp16
    # sig_k = ingrid - 2*mask_k for every channel k in ONE 1x1 conv:
    # W[out=k, in=j] = 1 (sum over all channels = ingrid) - 2*delta(k,j).
    wsig = (np.ones((10, 10), np.float16) - 2.0 * np.eye(10, dtype=np.float16))
    init("wsig", wsig.reshape(10, 10, 1, 1), np.float16)
    n("Conv", ["inwf", "wsig"], "sig10")                       # [1,10,W,W] fp16
    # reshape channels onto batch: [10,1,W,W]
    init("sh_b", np.array([10, 1, WORK, WORK], np.int64), np.int64)
    n("Reshape", ["sig10", "sh_b"], "sig")                     # [10,1,W,W] fp16

    # matched-filter conv: kernel 11x11 (centre at index 5).  Pad enough that
    # centres reach 1 cell off the top/left/bottom edges (pads [top,left,bot,right]).
    init("kern", kern, np.float16)
    n("Conv", ["sig", "kern"], "cdo", pads=[5, 6, 6, 5])       # [10,12,20,20] fp16
    n("ReduceMin", ["cdo"], "mind", axes=[2, 3], keepdims=1)   # [10,12,1,1] fp16

    # tot per channel [10,1,1,1]
    n("ReduceSum", ["input"], "tot10", axes=[2, 3], keepdims=1)  # [1,10,1,1] fp32
    init("sh_t", np.array([10, 1, 1, 1], np.int64), np.int64)
    n("Reshape", ["tot10", "sh_t"], "tot")                     # [10,1,1,1] fp32
    n("Cast", ["tot"], "totf", to=F16)

    # bg colour = channel with the largest total pixel count (computed early; used
    # both as the figure background fill and as the fallback for invisible rings)
    n("ArgMax", ["tot10"], "bgch", axis=1, keepdims=0)         # [1,1,1] int64
    n("Squeeze", ["bgch"], "bgs", axes=[0, 1, 2])              # scalar int64
    n("Cast", ["bgs"], "bgf", to=F32)                          # scalar fp32

    n("Add", ["mind", "totf"], "dist")                         # [10,12,1,1] fp16
    init("z16", np.array(0.0, np.float16), np.float16)
    n("Equal", ["dist", "z16"], "match")                       # [10,12,1,1] bool
    n("Cast", ["match"], "matchf", to=F16)

    # first (min-idx) matching type per channel
    n("ArgMax", ["matchf"], "ti", axis=1, keepdims=1)          # [10,1,1,1] int64
    n("ReduceMax", ["matchf"], "hasm", axes=[1], keepdims=1)   # [10,1,1,1] fp16 {0,1}

    # type -> idx and type -> L (gather)
    t2idx = np.array([t[0] for t in TYPES], np.float32)
    t2L = np.array([t[1] for t in TYPES], np.float32)
    init("t2idx", t2idx, np.float32)
    init("t2L", t2L, np.float32)
    # ti is [10,1,1,1]; gather along the type table (axis0) by squeezed scalar/idx
    n("Squeeze", ["ti"], "tis", axes=[1, 2, 3])                # [10] int64
    n("Gather", ["t2idx", "tis"], "kidx0")                     # [10] fp32 ring index
    n("Gather", ["t2L", "tis"], "kL0")                         # [10] fp32 length
    # idx0 override: a single-pixel colour (tot==1) is ring 0, L 1 (its kernel was
    # dropped from the conv; tot==1 <=> idx0 by the >=2-quadrant rule).
    init("sh10b", np.array([10], np.int64), np.int64)
    n("Reshape", ["tot10", "sh10b"], "tot1db")                 # [10] fp32
    init("onef10", np.array(1.0, np.float32), np.float32)
    n("Equal", ["tot1db", "onef10"], "isone")                  # [10] bool tot==1
    init("zer10v", np.zeros(10, np.float32), np.float32)
    init("one10v", np.ones(10, np.float32), np.float32)
    n("Where", ["isone", "zer10v", "kidx0"], "kidx")           # [10] ring idx
    n("Where", ["isone", "one10v", "kL0"], "kL")               # [10] length
    # has-match mask as [10]; gate by tot>0 so absent colours (whose stamp can be
    # placed entirely off-grid -> spurious dist==0) and the bg are excluded.
    n("Squeeze", ["hasm"], "hasm10", axes=[1, 2, 3])           # [10] fp16
    n("Cast", ["hasm10"], "hasmm", to=F32)                     # [10] fp32 {0,1}
    init("sh10", np.array([10], np.int64), np.int64)
    n("Reshape", ["tot10", "sh10"], "tot1d")                   # [10] fp32
    init("zerf2", np.array(0.0, np.float32), np.float32)
    n("Greater", ["tot1d", "zerf2"], "pres_b")                 # [10] bool tot>0
    n("Cast", ["pres_b"], "presf", to=F32)                     # [10]
    n("Mul", ["hasmm", "presf"], "hasmp")                      # [10] fp32 {0,1}
    # a single-pixel idx0 colour does NOT match the (idx>=1) kernels, so mark it
    # present explicitly via isone.
    n("Cast", ["isone"], "isonef", to=F32)                     # [10]
    n("Max", ["hasmp", "isonef"], "has10f")                    # [10] fp32 {0,1}

    # ---------- scatter into ring vectors (length 6) ----------
    # ring m gets colour = sum_k k * (kidx_k == m) * has_k ; L = sum_k L_k * (..)
    init("colvec", np.arange(10, dtype=np.float32), np.float32)        # [10] channel id
    init("ring6", np.arange(6, dtype=np.float32).reshape(6, 1), np.float32)  # [6,1]
    # eqkm[m,k] = (kidx_k == m)
    n("Equal", ["ring6", "kidx"], "eqkm_b")                    # [6,10] bool
    n("Cast", ["eqkm_b"], "eqkm", to=F32)                      # [6,10]
    n("Mul", ["eqkm", "has10f"], "eqkmh")                      # [6,10] gated
    # ringcolour[m] = sum_k eqkmh[m,k] * k
    n("Mul", ["eqkmh", "colvec"], "wc")                        # [6,10]
    n("ReduceSum", ["wc"], "ringsum", axes=[1], keepdims=0)    # [6] fp32
    n("Mul", ["eqkmh", "kL"], "wl")                            # [6,10]
    n("ReduceSum", ["wl"], "ringLsum", axes=[1], keepdims=0)   # [6] fp32
    # present[m] = some colour mapped to ring m
    n("ReduceMax", ["eqkmh"], "present", axes=[1], keepdims=0)  # [6] fp32 {0,1}
    # default invisible (bg-coloured) rings to bg colour and L=6 (claims its cells,
    # which then render bg -> identical to the output background).  This case ONLY
    # arises in the hand-authored ARC train/test where bg appears in the figure;
    # the random generator excludes bg, so fresh instances have all rings visible.
    init("one6", np.ones(6, np.float32), np.float32)
    n("Sub", ["one6", "present"], "absent")                    # [6]
    # bgf is scalar (defined below); compute ring colour/L with bg fallback
    n("Mul", ["absent", "bgf"], "absbg")                       # [6] bg where absent
    n("Add", ["ringsum", "absbg"], "ringcol")                 # [6] fp32
    init("six6", np.full(6, 6.0, np.float32), np.float32)
    n("Mul", ["absent", "six6"], "absL")                       # [6]
    n("Add", ["ringLsum", "absL"], "ringL")                    # [6] fp32
    # K = max visible idx + 1
    n("Mul", ["kidx", "has10f"], "vidx")                       # [10]
    n("ReduceMax", ["vidx"], "maxidx", axes=[0], keepdims=0)   # scalar fp32
    init("one_s", np.array(1.0, np.float32), np.float32)
    n("Add", ["maxidx", "one_s"], "Kf")                        # scalar fp32 = K

    # ---------- closed-form synthesis on 11x11 centred canvas ----------
    cen = KS // 2
    mc = np.zeros((KS, KS), np.float32)
    nc = np.zeros((KS, KS), np.float32)
    for a in range(-cen, cen + 1):
        for b in range(-cen, cen + 1):
            mc[cen + a, cen + b] = max(abs(a), abs(b))
            nc[cen + a, cen + b] = min(abs(a), abs(b))
    init("mc", mc, np.float32)
    init("nc", nc, np.float32)
    # gather ring colour / L over the m-cell grid (cast m to int index)
    n("Cast", ["mc"], "mci", to=I64)                           # [11,11] int64
    n("Gather", ["ringcol", "mci"], "colplane")               # [11,11] fp32
    n("Gather", ["ringL", "mci"], "Lplane")                    # [11,11] fp32
    # cond = (mc < K) AND (nc >= mc - L + 1)  ==  (nc > mc - L)
    n("Less", ["mc", "Kf"], "inK")                             # [11,11] bool
    # thr = mc - L  ; figure cell iff nc > thr
    n("Sub", ["mc", "Lplane"], "thr")                          # [11,11] fp32
    n("Greater", ["nc", "thr"], "isfig")                       # [11,11] bool
    n("And", ["inK", "isfig"], "cond")                         # [11,11] bool
    n("Where", ["cond", "colplane", "bgf"], "lab11f")          # [11,11] fp32 index
    n("Cast", ["lab11f"], "lab11", to=TensorProto.UINT8)       # [11,11] uint8 index

    # ---------- crop/shift to (2K-1)^2 at top-left ----------
    # source row for output row i (0..10) is i + (cen - (K-1)) = i + cen+1 - K.
    init("ar11", np.arange(KS, dtype=np.float32), np.float32)         # [11]
    init("cen1", np.array(float(cen + 1), np.float32), np.float32)
    n("Sub", ["cen1", "Kf"], "shift")                          # scalar = cen+1-K
    n("Add", ["ar11", "shift"], "srcf")                        # [11] fp32 source idx
    init("zerf", np.array(0.0, np.float32), np.float32)
    init("maxf", np.array(float(KS - 1), np.float32), np.float32)
    n("Clip", ["srcf", "zerf", "maxf"], "srcc")                # clamp 0..10
    n("Cast", ["srcc"], "srci", to=I64)                        # [11] int64
    n("Gather", ["lab11", "srci"], "lab_r", axis=0)            # [11,11] rows shifted
    n("Gather", ["lab_r", "srci"], "lab_s", axis=1)            # [11,11] cols shifted
    # off-grid: output index i >= 2K-1 must be sentinel.  size = 2K-1.
    init("two_f", np.array(2.0, np.float32), np.float32)
    init("one_f", np.array(1.0, np.float32), np.float32)
    n("Mul", ["Kf", "two_f"], "K2")
    n("Sub", ["K2", "one_f"], "size")                          # scalar = 2K-1
    # valid mask over 11x11: (i < size) AND (j < size)
    init("ar11c", np.arange(KS, dtype=np.float32).reshape(KS, 1), np.float32)  # [11,1]
    init("ar11r", np.arange(KS, dtype=np.float32).reshape(1, KS), np.float32)  # [1,11]
    n("Less", ["ar11c", "size"], "vrow")                       # [11,1] bool
    n("Less", ["ar11r", "size"], "vcol")                       # [1,11] bool
    n("And", ["vrow", "vcol"], "valid")                        # [11,11] bool
    init("sentu0", np.array(99, np.uint8), np.uint8)
    n("Where", ["valid", "lab_s", "sentu0"], "lab30")          # [11,11] uint8 index/sent

    # ---------- to one-hot ----------
    # The label plane is already uint8 (900B not 3600B fp32); pad sentinel 99
    # (matches no colour); Equal arange.
    n("Reshape", ["lab30", "sh4"], "lab4u")                    # [1,1,11,11] uint8
    init("sh4", np.array([1, 1, KS, KS], np.int64), np.int64)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - KS, 30 - KS], np.int64), np.int64)
    init("sentu", np.array(99, np.uint8), np.uint8)
    n("Pad", ["lab4u", "pads", "sentu"], "L30", mode="constant")  # [1,1,30,30] uint8
    init("k10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "k10"], "output")                       # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task096", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
