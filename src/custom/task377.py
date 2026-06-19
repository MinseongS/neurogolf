"""Task 377 (ARC-AGI eb5a1d5d) — nested rectangles -> concentric square rings.

Rule (from the generator): the input grid is fully tiled by N strictly-NESTED
axis-aligned rectangles painted in order; rect 0 fills the whole grid, each later
rect sits strictly INSIDE the previous (its top-left corner moves strictly SE) in
colours[0..N-1] (adjacent rings differ, but NON-adjacent rings may repeat a colour).
The OUTPUT is a (2N-1)x(2N-1) grid of concentric SQUARE rings: ring idx d (Chebyshev
distance from the border) = colours[d], placed at the top-left of the 30x30 canvas,
all-zero elsewhere.

The WHOLE output is determined by the scalar N and the length-N colour SEQUENCE
cv[0..N-1].  Both are recovered closed-form from the TOP-LEFT staircase of the input:
each rect i introduces a unique CORNER cell at cumulative offset (R_i, C_i) whose
colour is colours[i].  A corner = a cell that differs from BOTH its up- and its
left-neighbour (and is in-grid).  Corners march strictly SE, so there is at most ONE
corner per row; reading the per-row corner colours top-to-bottom yields colours[0..N-1].

Recovery (kojimar7113 backbone, re-used): colour-index grid via a dilated 1x1 Conv
cropped to 27x27, then per-row vertical-delta ArgMax finds each row's first
transition column, GatherElements reads that colour and rejects rows whose candidate
equals its left neighbour (non-corner) -> the inner colours; TopK + concat with the
outer-cell colour gives cv[0..N-1] and depth N.

Output ring tail (re-golfed vs kojimar, the win): kojimar materialised the ring index
at the FULL 30x30 (int32, 3600B) then Gathered the colour table straight into output.
Instead build the ring index on a TINY W x W canvas (W = 13 covers 2N-1 for N<=7, the
max depth observed over 500k samples; N=7 fills 13x13 exactly), Gather the cv VALUE
plane there, Pad into 30x30 (uint8) and Equal->BOOL output -- the only 30x30 output-side
tensor is uint8 (900B) and the ring/value planes are <=WxW.  This removes the 3600B
int32 full-canvas ring plane (kojimar 10887 -> 8581, +0.24).

Floor: grid_f is the irreducible fp32 27x27 colour-index entry (2916B, 10->1 channel
reduction off the fp32 input -- kojimar pays this too).  The recovery scan needs three
26x27 uint8 planes (curr/prev/row_delta = 2106B) and the value tail one 30x30 uint8
plane (Lpad 900B); none can be removed without a fundamentally different recovery, so
the architecture floors near ~8.4KB (the +0.3 target ~8.3KB is just out of reach).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
I32 = TensorProto.INT32
I64 = TensorProto.INT64
B = TensorProto.BOOL

K = 8           # depth capacity (observed max N over 500k samples = 7)
W = 13          # tiny output canvas, fits every figure (2N-1 <= 15 for N<=8)
GW = 27         # cropped grid for recovery (max corner row/col = 26)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour-index grid (cropped GWxGW) via a 1x1 dilated Conv on the FREE input ----
    # weight[k,0,0,0]=k (only the top-left tap of a 2x2 dilation-(GW-1) kernel is nonzero)
    # so grid_f[r][c] = sum_k k*input[k][r][c] for the top-left GWxGW window.
    cw = np.zeros((1, 10, 2, 2), np.float32)
    for k in range(10):
        cw[0, k, 0, 0] = k
    init("cw", cw, np.float32)
    # kernel is 2x2 but only the [0,0] tap is nonzero; dilation (30-GW) sets how much
    # bottom/right border is cropped, so output = grid_f[r][c] = colf for the GWxGW window.
    n("Conv", ["input", "cw"], "grid_f", dilations=[30 - GW, 30 - GW])  # [1,1,27,27] fp32
    n("Cast", ["grid_f"], "grid", to=U8)                              # [1,1,27,27] uint8

    # ---- per-row first VERTICAL transition -> candidate corner column ----
    # curr = grid rows 1..26, prev = grid rows 0..25; row_delta = curr - prev (uint8 wrap,
    # ==0 iff equal to the row above).  curr serves double duty: subtraction operand AND
    # the colour-read source (GatherElements), so no extra plane is needed for reads.
    init("ax_hw", np.array([2, 3], np.int64), np.int64)
    init("s_curr_s", np.array([1, 0], np.int64), np.int64)
    init("s_curr_e", np.array([GW, GW], np.int64), np.int64)
    init("s_prev_s", np.array([0, 0], np.int64), np.int64)
    init("s_prev_e", np.array([GW - 1, GW], np.int64), np.int64)
    n("Slice", ["grid", "s_curr_s", "s_curr_e", "ax_hw"], "curr")   # [1,1,26,27] uint8
    n("Slice", ["grid", "s_prev_s", "s_prev_e", "ax_hw"], "prev")   # [1,1,26,27] uint8
    n("Sub", ["curr", "prev"], "row_delta")                         # [1,1,26,27] uint8
    n("ArgMax", ["row_delta"], "row_idx", axis=3, keepdims=1)       # [1,1,26,1] int64

    # candidate colour at that column, and the colour just to its left
    n("GatherElements", ["curr", "row_idx"], "cand", axis=3)        # [1,1,26,1] uint8
    init("one_i64", np.array([1], np.int64), np.int64)
    n("Sub", ["row_idx", "one_i64"], "left_idx")
    n("GatherElements", ["curr", "left_idx"], "cand_left", axis=3)  # [1,1,26,1] uint8

    # a real inner corner row: candidate column>0 AND candidate != its left neighbour
    init("zero_i64", np.array([0], np.int64), np.int64)
    n("Equal", ["cand", "cand_left"], "cand_eq_l")
    n("Not", ["cand_eq_l"], "cand_diff_l")
    n("Greater", ["row_idx", "zero_i64"], "row_any")
    n("And", ["row_any", "cand_diff_l"], "is_inner")               # [1,1,26,1] bool

    # rank the inner-corner rows TOP-first (smaller row index = larger weight)
    rw = np.arange(GW - 1, 0, -1, np.float16).reshape(1, 1, GW - 1, 1)  # 26..1
    init("rw", rw, np.float16)
    init("zero_f16", np.array([0.0], np.float16), np.float16)
    n("Where", ["is_inner", "rw", "zero_f16"], "row_scores")       # [1,1,26,1] fp16
    init("topk_k", np.array([K - 1], np.int64), np.int64)          # at most K-1 inner rects
    nodes.append(helper.make_node("TopK", ["row_scores", "topk_k"],
                                  ["top_values", "top_idx"],
                                  axis=2, largest=1, sorted=1))
    n("GatherElements", ["cand", "top_idx"], "inner_colors", axis=2)  # [1,1,K-1,1] uint8

    # depth N = 1 (outer rect) + #active inner rows
    n("Greater", ["top_values", "zero_f16"], "active")
    n("Cast", ["active"], "active_f", to=F16)
    n("ReduceSum", ["active_f"], "ninner_f", keepdims=0)           # scalar fp16
    init("one_f16", np.array([1.0], np.float16), np.float16)
    n("Add", ["ninner_f", "one_f16"], "Nf")                       # N as fp16 scalar

    # ---- cv[0..K-1] colour sequence: outer cell colour, then inner colours ----
    init("s_outer_e", np.array([1, 1], np.int64), np.int64)
    n("Slice", ["grid", "s_prev_s", "s_outer_e", "ax_hw"], "outer_cell")  # [1,1,1,1] uint8
    n("Concat", ["outer_cell", "inner_colors"], "colorsKpre", axis=2)     # [1,1,K,1] uint8
    init("rs_k1", np.array([1, K], np.int64), np.int64)
    n("Reshape", ["colorsKpre", "rs_k1"], "colorsK")               # [1,K] uint8
    # append a sentinel 250 so an off-figure ring index (-1 -> wraps to K) reads 250.
    init("sent_row", np.full((1, 1), 250, np.uint8), np.uint8)
    n("Concat", ["colorsK", "sent_row"], "cv2d", axis=1)           # [1,K+1] uint8
    init("rs_kp1", np.array([K + 1], np.int64), np.int64)
    n("Reshape", ["cv2d", "rs_kp1"], "cv")                         # [K+1] uint8

    # ---- TINY ring index L(i,j) = cv[ ring(i,j) ] on a W x W canvas, all in int32 ----
    # ring(i,j) = min(i, j, m-i, m-j), m = 2N-2 (Chebyshev distance from the border).
    n("Cast", ["Nf"], "Ni", to=I32)                               # N as int32 scalar
    init("two_i", np.array(2, np.int32), np.int32)
    n("Mul", ["Ni", "two_i"], "twoN")
    n("Sub", ["twoN", "two_i"], "mval")                           # m = 2N-2, int32 scalar
    ii = np.arange(W, dtype=np.int32).reshape(1, 1, W, 1)
    jj = np.arange(W, dtype=np.int32).reshape(1, 1, 1, W)
    init("ii", ii, np.int32)
    init("jj", jj, np.int32)
    init("neg1", np.array(-1, np.int32), np.int32)
    n("Sub", ["mval", "ii"], "m_i")                               # [1,1,W,1] int32
    n("Sub", ["mval", "jj"], "m_j")                               # [1,1,1,W] int32
    n("Min", ["ii", "m_i"], "ri0")
    n("Max", ["ri0", "neg1"], "ri")                               # clamp >= -1
    n("Min", ["jj", "m_j"], "rj0")
    n("Max", ["rj0", "neg1"], "rj")
    n("Min", ["ri", "rj"], "ring_i")                              # [1,1,W,W] int32, in [-1,N-1]
    n("Gather", ["cv", "ring_i"], "L", axis=0)                    # [1,1,W,W] uint8 (-1 wraps to cv[K]=250)

    # ---- Pad W x W into 30x30 (uint8) then Equal -> FREE BOOL output ----
    padW = 30 - W
    init("pad_br", np.array([0, 0, 0, 0, 0, 0, padW, padW], np.int64), np.int64)
    init("sent_u", np.array(250, np.uint8), np.uint8)
    n("Pad", ["L", "pad_br", "sent_u"], "Lpad")                  # [1,1,30,30] uint8 (900B)
    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["Lpad", "chan"], "output")                       # [1,10,30,30] BOOL (free)

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task377", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 18)])
