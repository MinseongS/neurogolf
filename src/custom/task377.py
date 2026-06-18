"""Task 377 (ARC-AGI eb5a1d5d) — nested rectangles -> concentric square rings.

Rule (from the generator): the input grid is fully tiled by N strictly-NESTED
axis-aligned rectangles painted in order; rect 0 fills the whole grid, each later
rect sits strictly inside the previous (top-left corner moves SE, bottom-right
moves NW) in colours[0..N-1] (adjacent rings differ, but NON-adjacent rings may
repeat a colour).  The OUTPUT is a (2N-1)x(2N-1) grid of concentric SQUARE rings:
ring idx (Chebyshev distance from the border) = colours[idx], placed at top-left,
and the rest of the 30x30 canvas is all-zero (background channel never set).

Because the WHOLE output is determined by the scalar N and the length-N colour
SEQUENCE cv[0..N-1], the figure is built on a TINY fixed W x W canvas (W = 2K-1 =
15 for capacity K = 8) and Pad-ed into the FREE 30x30 output as the final op.  No
full-grid ring / index / value plane is ever materialised — only the input-scan
planes needed to RECOVER the depth profile remain at 30-width.

Recovery (closed-form, separable, no flood-fill / argmax op):
  * colf = colour-index plane (1x1 conv sum_k k*input_k); every in-grid cell >=1,
    off-grid pad cells = 0.  Cast to fp16 so the one input-scan plane counts at half.
  * Per-row horizontal colour TRANSITIONS: a row that passes through depth d shows
    the nested palindrome of segments => #transitions = 2*(rowdepth-1) (+1 spurious
    at the grid right edge, absorbed by floor) so rowdepth = floor((W-1-#same)/2)+1
    (a 1-D [30] profile).  This is the ONLY 30-wide scan (the vertical scan is gone).
  * N = max rowdepth (scalar): every rect has row- AND col-extent >=3, so the deepest
    row passes through ALL N rects (verified) -> no separate vertical depth scan.
  * colour SEQUENCE cv[0..N-1] is read off the SINGLE deepest row: rowselN = 1-hot of
    the first row reaching depth N; deeprow = rowselN @ colf [1,1,1,30] reads the
    nested palindrome c0 c1..c_{N-1}..c1 c0.  A tiny 1-D transition->prefix-sum gives a
    per-column segment index; cv[d] = deeprow at the first column of segment d.  Every
    tensor after colf is <= [1,1,K,30] -- no second full-grid scan plane.

Output ring plane (TINY 15x15):
  ring(i,j) = min(i, j, m-i, m-j), m = 2N-2 (Chebyshev distance from border) over a
  fixed 15x15 grid; out-of-region cells get ring<0, clamped to -1 so Gather wraps to
  a cv sentinel of -1.  L = cv[ring] is [1,1,15,15]; Cast to int32, Pad to [1,1,30,30]
  with pad value -1 (opset-11 Pad accepts int32), then free BOOL output =
  Equal(Lpad, arange[0..9]).  -1 (off-figure / off-canvas) and channel-0 stay all-off.

Memory (~15.6KB, pts ~15.28): the dominant planes are the fp32 colour conv entry
(3600B, irreducible 10->1 reduction off the fp32 input) + fp16 colf (1800B) + the
single horizontal transition scan (hdiff/h_eq/h_eqf ~4350B).  The ring/index/value
planes are all 15x15 (225-900B) and the colour-sequence machinery is <= [1,1,8,30].
The one big remaining cost is the un-removable 30-wide horizontal depth scan.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
B = TensorProto.BOOL

K = 8   # nesting-depth capacity (observed max over 230k+ samples = 7).
W = 2 * K - 1  # = 15: tiny output canvas, fits every figure (2N-1 <= 15 for N<=K).


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F16 = TensorProto.FLOAT16

    # ---- colour-index plane via a 1x1 conv on the FREE input ----
    kw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("kw", kw, np.float32)
    n("Conv", ["input", "kw"], "colf32", kernel_shape=[1, 1])  # [1,1,30,30] fp32
    n("Cast", ["colf32"], "colf", to=F16)                  # [1,1,30,30] fp16 (1800B)

    init("zero", np.array(0.0, np.float16), np.float16)
    init("half", np.array(0.5, np.float16), np.float16)

    # ---- horizontal transitions -> rowdepth [1,1,30,1] ----
    khdiff = np.array([1.0, -1.0], np.float16).reshape(1, 1, 1, 2)
    init("khdiff", khdiff, np.float16)
    n("Conv", ["colf", "khdiff"], "hdiff", kernel_shape=[1, 2])  # [1,1,30,29] fp16
    n("Equal", ["hdiff", "zero"], "h_eq")                  # bool: diff==0 (same)
    n("Cast", ["h_eq"], "h_eqf", to=F16)                   # [1,1,30,29] fp16
    n("ReduceSum", ["h_eqf"], "hsame", axes=[3], keepdims=1)  # [1,1,30,1]
    # rowdepth = floor((29 - hsame)/2) + 1 = floor((31 - hsame)/2)
    init("c31", np.array(31.0, np.float16), np.float16)
    n("Sub", ["c31", "hsame"], "htr_a")
    n("Mul", ["htr_a", "half"], "htr_h")
    n("Floor", ["htr_h"], "rowdepth")                      # [1,1,30,1] fp16

    # ---- N = max rowdepth (scalar) ----
    # Every rect has both row- and col-extent >=3, so the deepest row passes through
    # ALL N rects => max rowdepth == N (verified 4000/4000).  No vertical scan needed.
    n("ReduceMax", ["rowdepth"], "Nf", axes=[2, 3], keepdims=1)  # [1,1,1,1] = N fp16

    # ---- rowselN [1,1,1,30]: indicator of the FIRST row reaching depth N ----
    # rowdepth rises by +1 per level, so first-row-of-depth-N = (rd==N) AND (rd_prev<N).
    init("rs1d30", np.array([1, 1, 1, 30], np.int64), np.int64)
    n("Reshape", ["rowdepth", "rs1d30"], "rd1")            # [1,1,1,30] fp16
    init("pad_l1", np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), np.int64)
    init("s_a0b", np.array([0], np.int64), np.int64)
    init("s_30", np.array([30], np.int64), np.int64)
    init("ax3b", np.array([3], np.int64), np.int64)
    n("Pad", ["rd1", "pad_l1"], "rd_p")                    # [1,1,1,31] fp16
    n("Slice", ["rd_p", "s_a0b", "s_30", "ax3b"], "rd_prev")  # [1,1,1,30]
    n("Equal", ["rd1", "Nf"], "rd_eqN")                    # [1,1,1,30] bool
    n("Less", ["rd_prev", "Nf"], "rd_ltN")                 # [1,1,1,30] bool
    n("And", ["rd_eqN", "rd_ltN"], "rsN_b")               # [1,1,1,30] bool
    n("Cast", ["rsN_b"], "rowselN", to=F16)                # [1,1,1,30] fp16 (1-hot row)

    # ---- deeprow = colour profile of the deepest row (reads c0..c_{N-1}..c1 c0) ----
    # MatMul(rowselN[1,1,1,30], colf[1,1,30,30]) -> [1,1,1,30]: selects that one row.
    n("MatMul", ["rowselN", "colf"], "deeprow")            # [1,1,1,30] fp16

    # ---- segment index along deeprow via a triangular MatMul (no CumSum: fp16-rejected) ----
    # trans[c] = (deeprow[c] != deeprow[c-1]) for c>=1, trans[0]=0 ; seg[c] = sum_{k<=c} trans[k].
    khdiff1 = np.array([1.0, -1.0], np.float16).reshape(1, 1, 1, 2)
    init("khdiff1", khdiff1, np.float16)
    n("Conv", ["deeprow", "khdiff1"], "drdiff", kernel_shape=[1, 2])  # [1,1,1,29] fp16
    n("Equal", ["drdiff", "zero"], "dr_eq")                # bool: same (diff==0)
    n("Not", ["dr_eq"], "dr_ne")                           # bool: transition
    n("Cast", ["dr_ne"], "dr_nef", to=F16)                 # [1,1,1,29] fp16
    # prepend a 0 (trans[0]=0) -> length 30, then inclusive prefix-sum via upper-tri.
    init("pad_l1c", np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), np.int64)
    n("Pad", ["dr_nef", "pad_l1c"], "trans30")             # [1,1,1,30] fp16 (left-pad 0)
    ut = np.triu(np.ones((30, 30), np.float16)).reshape(1, 1, 30, 30)  # ut[k,c]=1 if k<=c
    init("ut", ut, np.float16)
    n("MatMul", ["trans30", "ut"], "seg")                  # [1,1,1,30] fp16: seg[c]

    # ---- colsel1d [1,1,K,30]: first column of segment d (d=0..K-1) ----
    darr0 = np.arange(0, K, dtype=np.float16).reshape(1, 1, K, 1)  # segment ids 0..K-1
    init("darr0", darr0, np.float16)
    # left-pad seg with a -1 sentinel so segment 0's first column qualifies (prev<0).
    init("neg1c", np.array(-1.0, np.float16), np.float16)
    n("Pad", ["seg", "pad_l1", "neg1c"], "seg_p")          # [1,1,1,31] fp16 (left = -1)
    n("Slice", ["seg_p", "s_a0b", "s_30", "ax3b"], "seg_prev")  # [1,1,1,30]
    n("Equal", ["seg", "darr0"], "seg_eqd")                # [1,1,K,30] bool: seg==d
    n("Less", ["seg_prev", "darr0"], "seg_ltp")            # [1,1,K,30] bool: prev<d
    n("And", ["seg_eqd", "seg_ltp"], "colsel_b")           # [1,1,K,30] bool
    n("Cast", ["colsel_b"], "colsel", to=F16)              # [1,1,K,30] fp16

    # ---- cv[d] = sum_c deeprow[c] * colsel1d[d,c] ----
    n("Mul", ["deeprow", "colsel"], "cvprod")              # [1,1,K,30] fp16
    n("ReduceSum", ["cvprod"], "cv0", axes=[3], keepdims=0)  # [1,1,K] fp16
    init("rsk1", np.array([K], np.int64), np.int64)
    n("Reshape", ["cv0", "rsk1"], "cvr")                   # [K] fp16
    init("cvpad", np.array([0, 1], np.int64), np.int64)    # pad 1 on the right
    # sentinel 250: a uint8-safe "no colour" value (figure ring colours are 1..9, so
    # 250 lets the padded value plane be uint8 below, halving it vs an fp16 -1 plane).
    init("sent", np.array(250.0, np.float16), np.float16)
    n("Pad", ["cvr", "cvpad", "sent"], "cv")               # [K+1] fp16; cv[K] = 250

    # ---- TINY ring-colour plane L(i,j) = cv[ring(i,j)] on a W x W canvas ----
    # ring(i,j) = min(i, j, m-i, m-j), m = 2N-2 (Chebyshev distance from border).
    ii = np.arange(W, dtype=np.float16).reshape(1, 1, W, 1)
    jj = np.arange(W, dtype=np.float16).reshape(1, 1, 1, W)
    init("ii", ii, np.float16)
    init("jj", jj, np.float16)
    init("two", np.array(2.0, np.float16), np.float16)
    init("neg1b", np.array(-1.0, np.float16), np.float16)
    n("Mul", ["Nf", "two"], "twoN")                        # 2N
    n("Sub", ["twoN", "two"], "mval")                      # m = 2N-2 [1,1,1,1]
    n("Sub", ["mval", "ii"], "m_i")                        # [1,1,W,1]
    n("Sub", ["mval", "jj"], "m_j")                        # [1,1,1,W]
    n("Min", ["ii", "m_i"], "ri0")                         # [1,1,W,1]
    n("Max", ["ri0", "neg1b"], "ri")                       # clamp >= -1
    n("Min", ["jj", "m_j"], "rj0")                         # [1,1,1,W]
    n("Max", ["rj0", "neg1b"], "rj")
    n("Min", ["ri", "rj"], "ring")                         # [1,1,W,W] fp16 in [-1,N-1]
    n("Cast", ["ring"], "idx_i", to=TensorProto.INT32)     # [1,1,W,W] int32
    n("Gather", ["cv", "idx_i"], "L", axis=0)              # [1,1,W,W] fp16; off-fig=250

    # ---- Cast L -> uint8, Pad into 30x30, then Equal -> FREE output ----
    # L (fp16, values {1..9} inside the figure, 250 outside) -> uint8 (225B); Pad
    # W x W -> 30x30 with value 250 (opset-11 Pad accepts uint8) so the ONLY 30x30
    # output-side tensor is uint8 (900B, half the fp16 1800B value plane).  Then
    # Equal(Lpad, arange[0..9]) (ORT Equal runs on uint8) emits the 10-ch one-hot BOOL
    # into the FREE graph output; channel-0 and the 250 sentinel (off-figure / trailing
    # padding) match no channel 0..9 and stay all-off = the all-zero target there.
    n("Cast", ["L"], "Lu", to=TensorProto.UINT8)           # [1,1,W,W] uint8
    padW = 30 - W
    init("pad_br", np.array([0, 0, 0, 0, 0, 0, padW, padW], np.int64), np.int64)
    init("sent_u", np.array(250, np.uint8), np.uint8)
    n("Pad", ["Lu", "pad_br", "sent_u"], "Lpad")           # [1,1,30,30] uint8 (900B)
    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["Lpad", "chan"], "output")                 # [1,10,30,30] BOOL (free)

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task377", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
