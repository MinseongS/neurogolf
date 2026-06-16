"""Task 377 (ARC-AGI eb5a1d5d) — nested rectangles -> concentric square rings.

Rule (from the generator): the input grid is fully tiled by N strictly-NESTED
axis-aligned rectangles painted in order; rect 0 fills the whole grid, each later
rect sits strictly inside the previous (top-left corner moves SE, bottom-right
moves NW) in colours[0..N-1] (adjacent rings differ, but NON-adjacent rings may
repeat a colour).  The OUTPUT is a (2N-1)x(2N-1) grid of concentric SQUARE rings:
ring idx (Chebyshev distance from the border) = colours[idx], placed at top-left.

Recovery (closed-form, separable, no flood-fill / argmax op):
  * colf = colour-index plane (1x1 conv sum_k k*input_k); every in-grid cell >=1,
    off-grid pad cells = 0.
  * Per-row horizontal colour TRANSITIONS t_r ; a row that passes through depth d
    shows the nested palindrome of segments => t_r = 2*(rowdepth-1) (+1 spurious
    at the grid right edge, absorbed by floor) so rowdepth = floor(t_r/2)+1.
    Symmetric coldepth from vertical transitions.  Both are 1-D [30] profiles.
  * depth(r,c) = min(rowdepth[r], coldepth[c])  (true because of strict nesting);
    N = max depth = min(max rowdepth, max coldepth)  -> NO 30x30 depth plane.
  * colour-by-depth vector cv[d] = colf at (first row with rowdepth>=d, first col
    with coldepth>=d).  rowsel[d,r] = (rd[r]==d AND rd[r-1]<d) (rowdepth rises by
    +1 per level, so this is the first row reaching depth d); colsel[d,c] likewise.
    cv = ReduceSum((rowsel @ colf) * colsel) -> [K], all tiny tensors.
  * output ring-colour plane L(i,j) = cv[ring(i,j)] with ring(i,j)=min(i,j,m-i,m-j),
    m=2N-2 (Chebyshev distance from border).  Out-of-region cells have ring<0,
    clamped to -1 so Gather wraps to a cv sentinel of -1 (matches no channel 0..9).
    Free BOOL output = Equal(L, arange[0..9]); off-region & channel-0 stay all-zero
    (colours are 1..9, never 0).

Memory: the colour plane is computed in fp32 (free input forces it) then cast to
fp16; ALL full-canvas working planes (colf16, ring, L, the two transition diffs)
then count at ~1740-1800B each.  The two irreducible 3600B planes are the fp32
colour conv (input-reduction entry point) and the int32 gather-index (ORT rejects
narrower index dtypes).  No depth plane, no per-depth masked planes; ~25.9KB total.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
B = TensorProto.BOOL

K = 8  # nesting-depth capacity.  Observed max over 230k+ samples = 7 (n=7 hit
# only ~6/200k).  The geometry allows up to ~14 nests but each step must leave room
# for a >=1 offset, so deeper nests are vanishingly unlikely; K=8 gives a margin
# past every sample seen.  The recovery RULE is exact for any n<=K (the cap only
# bounds the colour-by-depth vector length).


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
    # Conv in fp32 (input is free), then cast the single [1,1,30,30] colour plane to
    # fp16 (colours 0..9 are fp16-exact) so all downstream working planes count at
    # ~1800B instead of ~3600B.  (Casting the whole 10-ch input to fp16 would cost
    # 18000B, so the cast is applied AFTER the 10->1 channel collapse.)
    kw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("kw", kw, np.float32)
    n("Conv", ["input", "kw"], "colf32", kernel_shape=[1, 1])  # [1,1,30,30] fp32
    n("Cast", ["colf32"], "colf", to=F16)                  # [1,1,30,30] fp16 (1800B)

    init("half", np.array(0.5, np.float16), np.float16)
    init("zero", np.array(0.0, np.float16), np.float16)

    # ---- horizontal transitions -> rowdepth [1,1,30,1] ----
    # diff = colf[...,c] - colf[...,c-1] via a [1,-1] conv (one [1,1,30,29] plane);
    # transition <=> diff != 0.  rowdepth = floor(#trans / 2) + 1.  All fp16.
    # transitions = (#adjacent pairs) - (#equal pairs); only the EQUAL bool plane is
    # built (one bool + one Cast), and rowdepth = floor((W-1 - #same)/2) + 1 folds the
    # pair-count and the +1 into one constant.
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

    # ---- vertical transitions -> coldepth [1,1,1,30] ----
    kvdiff = np.array([1.0, -1.0], np.float16).reshape(1, 1, 2, 1)
    init("kvdiff", kvdiff, np.float16)
    n("Conv", ["colf", "kvdiff"], "vdiff", kernel_shape=[2, 1])  # [1,1,29,30] fp16
    n("Equal", ["vdiff", "zero"], "v_eq")
    n("Cast", ["v_eq"], "v_eqf", to=F16)                   # [1,1,29,30] fp16
    n("ReduceSum", ["v_eqf"], "vsame", axes=[2], keepdims=1)  # [1,1,1,30]
    n("Sub", ["c31", "vsame"], "vtr_a")
    n("Mul", ["vtr_a", "half"], "vtr_h")
    n("Floor", ["vtr_h"], "coldepth")                      # [1,1,1,30] fp16

    # ---- N = min(max rowdepth, max coldepth) (scalar) ----
    n("ReduceMax", ["rowdepth"], "maxrd", axes=[2, 3], keepdims=1)  # [1,1,1,1]
    n("ReduceMax", ["coldepth"], "maxcd", axes=[2, 3], keepdims=1)  # [1,1,1,1]
    n("Min", ["maxrd", "maxcd"], "Nf")                     # [1,1,1,1] = N (fp32)

    # ---- rowsel/colsel [1,1,K,30] (first-qualifying-row/col indicators) ----
    # keep everything 4D ([1,1,K,30]) so the cv MatMul needs no reshapes.  rd is
    # [1,1,1,30]; darr is [1,1,K,1]; comparisons broadcast to [1,1,K,30].
    init("rs1d30", np.array([1, 1, 1, 30], np.int64), np.int64)
    n("Reshape", ["rowdepth", "rs1d30"], "rd1")            # [1,1,1,30] fp16
    n("Reshape", ["coldepth", "rs1d30"], "cd1")            # [1,1,1,30] fp16
    darr = np.arange(1, K + 1, dtype=np.float16).reshape(1, 1, K, 1)
    init("darr", darr, np.float16)                         # [1,1,K,1] fp16
    # rowdepth rises by exactly +1 at each new level, so the FIRST row of depth d is
    # where rowdepth==d AND rowdepth[r-1]<d.  rowsel[d,r]=Eq(rd,d) AND Lt(rd_prev,d).
    # All comparisons (Equal/Less ok on fp16); avoids fp16 Min/Max (ORT crashes).
    init("pad_l1", np.array([0, 0, 0, 1, 0, 0, 0, 0], np.int64), np.int64)  # axis3 begin
    init("s_a0b", np.array([0], np.int64), np.int64)
    init("s_30", np.array([30], np.int64), np.int64)
    init("ax3b", np.array([3], np.int64), np.int64)

    def rise(d1, tag):
        n("Pad", [d1, "pad_l1"], tag + "_p")               # [1,1,1,31] fp16
        n("Slice", [tag + "_p", "s_a0b", "s_30", "ax3b"], tag + "_prev")  # [1,1,1,30]
        n("Equal", [d1, "darr"], tag + "_eqd")             # [1,1,K,30] bool: rd==d
        n("Less", [tag + "_prev", "darr"], tag + "_ltp")   # [1,1,K,30] bool: prev<d
        n("And", [tag + "_eqd", tag + "_ltp"], tag + "_b")  # [1,1,K,30] bool
        n("Cast", [tag + "_b"], tag + "sel", to=F16)       # [1,1,K,30] fp16
    rise("rd1", "row")   # -> rowsel  [1,1,K,30]
    rise("cd1", "col")   # -> colsel  [1,1,K,30]

    # ---- cv[d] = sum_c (rowsel[d] @ colf)[c] * colsel[d,c] ----
    # batched MatMul rowsel[1,1,K,30] @ colf[1,1,30,30] -> [1,1,K,30]; no reshapes.
    n("MatMul", ["rowsel", "colf"], "rowcolf")             # [1,1,K,30]
    n("Mul", ["rowcolf", "colsel"], "cvprod")              # [1,1,K,30] fp16
    n("ReduceSum", ["cvprod"], "cv0", axes=[3], keepdims=0)  # [1,1,K] fp16
    init("rsk1", np.array([K], np.int64), np.int64)
    n("Reshape", ["cv0", "rsk1"], "cvr")                   # [K] fp16
    init("cvpad", np.array([0, 1], np.int64), np.int64)    # pad 1 on the right
    init("neg1", np.array(-1.0, np.float16), np.float16)
    n("Pad", ["cvr", "cvpad", "neg1"], "cv")               # [K+1] fp16; cv[K] = -1

    # ---- output ring-colour plane L(i,j) = cv[ring(i,j)] ----
    # ring(i,j) = min(i, j, m-i, m-j), m = 2N-2 (Chebyshev distance from border).
    # ri = min(i, m-i) [1,1,30,1], rj = min(j, m-j) [1,1,1,30] are tiny; ring =
    # min(ri, rj) is the ONLY full plane.  Out-of-region cells have ring<0; clamp
    # to -1 so Gather wraps to cv[K] = -1 (matches no channel 0..9 -> all-off).
    ii = np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1)
    jj = np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30)
    init("ii", ii, np.float16)
    init("jj", jj, np.float16)
    init("two", np.array(2.0, np.float16), np.float16)
    # ring geometry in fp16 (values are small ints, exact); the only full plane
    # (ring) counts at 1800B.
    n("Mul", ["Nf", "two"], "twoN")                        # 2N (fp16)
    n("Sub", ["twoN", "two"], "mval")                      # m = 2N-2 [1,1,1,1]
    n("Sub", ["mval", "ii"], "m_i")                        # [1,1,30,1]
    n("Sub", ["mval", "jj"], "m_j")                        # [1,1,1,30]
    # clamp the negative tails in the TINY ri/rj vectors (>= -1) so the only full
    # ring plane is already in [-1, N-1]; -1 wraps Gather to cv[K] = -1 sentinel.
    init("neg1b", np.array(-1.0, np.float16), np.float16)
    n("Min", ["ii", "m_i"], "ri0")                         # [1,1,30,1]
    n("Max", ["ri0", "neg1b"], "ri")                       # clamp >= -1
    n("Min", ["jj", "m_j"], "rj0")                         # [1,1,1,30]
    n("Max", ["rj0", "neg1b"], "rj")
    n("Min", ["ri", "rj"], "ring")                         # [1,1,30,30] fp16 in [-1,N-1]
    n("Cast", ["ring"], "idx_i", to=TensorProto.INT32)     # [1,1,30,30] int32
    n("Gather", ["cv", "idx_i"], "L", axis=0)              # [1,1,30,30] fp16 (1800B)
    chan = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("chan", chan, np.float16)
    n("Equal", ["L", "chan"], "output")                    # [1,10,30,30] BOOL (free)

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task377", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
