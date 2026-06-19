"""Task 264 (a8c38be5): 9 fixed-glyph sprites recolored, scattered to a 9x9 chart.

Rule (from ARC-GEN generator): the input is an H x W grid (H,W in 14..16) on a
black(0) background holding 9 non-overlapping 3x3 sprites.  Each sprite is a solid
gray(5) 3x3 block whose cells at a FIXED per-index glyph shape are overwritten by
that sprite's color (colors are random non-gray, MAY repeat across sprites).  The
output is a fixed 9x9 grid (gray background) laid out as a 3x3 arrangement of 3x3
glyphs: cell (idx//3, idx%3) holds glyph[idx] painted in sprite idx's color; gray
elsewhere.  idx==4 (center) has an EMPTY glyph -> a solid gray center block.

So only thing that varies per instance is the COLOR of each of the 9 glyphs; the
glyph SHAPES and their output positions are fixed constants.  The task is therefore
recover-8-color-scalars + stamp-a-fixed-template.

Recovery (the load-bearing trick).  Each glyph shape is unique, and its complement
within the 3x3 block (the GRAY cells of that sprite) is therefore also unique.  Gray
is a single fixed channel (no color-collision ambiguity, unlike the color cells which
can repeat).  So for index k we build a 3x3 matched filter on the gray channel
    F_k = comp_k - 9 * glyph_k        (comp_k = 1 - glyph_k = gray cells of sprite k)
and convolve (valid, no pad).  The response equals count(comp_k) EXACTLY iff every
comp_k cell is gray AND no glyph_k cell is gray within that window -- i.e. only at the
top-left of sprite k (verified over thousands of fresh instances: exactly one hit per
k, no false positives, correct color every time).  `Equal(resp, count_comp_k)` is the
binary detector D_k.

Color readout.  colorval = sum_c c*input[c] (a 1x1x10 channel-collapse conv).
colorsum_k = colorval (x) glyph_k (valid conv) = sum of color over the glyph cells at
each top-left = count(glyph_k) * color[k] at the detected top-left.  Multiply by D_k,
ReduceSum over space -> count(glyph_k)*color[k], divide by count(glyph_k) -> color[k].
All 8 indices (skip k=4, empty) done as one 8-output Conv each.

Output build.  We have an 8-vector of colors.  Place each into the fixed 9x9 template:
Tcol[1,1,9,9] = sum_k color[k]*placedglyph_k (fixed 0/1 templates), Mg[1,1,9,9] = union
of placed glyphs (1 where a glyph cell, else 0).  Label L = where(Mg, Tcol, 5) (gray),
Pad to 30x30 with sentinel 12, final Equal(L, arange[1,10,1,1]) -> free BOOL output.

Memory: the two H-2 x W-2 (<=14x14) 8-channel conv planes dominate; everything after
collapses to <=9x9.  fp16 throughout (values are small integers, exact).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def _glyph(idx):
    r, c = [], []
    r, c = (r if idx != 0 else [0, 0, 1]), (c if idx != 0 else [0, 1, 0])
    r, c = (r if idx != 1 else [0, 0, 0, 1]), (c if idx != 1 else [0, 1, 2, 1])
    r, c = (r if idx != 2 else [0, 0, 1]), (c if idx != 2 else [1, 2, 2])
    r, c = (r if idx != 3 else [0, 1, 1, 2]), (c if idx != 3 else [0, 0, 1, 0])
    r, c = (r if idx != 5 else [0, 1, 1, 2]), (c if idx != 5 else [2, 1, 2, 2])
    r, c = (r if idx != 6 else [1, 2, 2]), (c if idx != 6 else [0, 0, 1])
    r, c = (r if idx != 7 else [1, 2, 2, 2]), (c if idx != 7 else [1, 0, 1, 2])
    r, c = (r if idx != 8 else [1, 2, 2]), (c if idx != 8 else [2, 1, 2])
    m = np.zeros((3, 3), np.float32)
    for dr, dc in zip(r, c):
        m[dr][dc] = 1.0
    return m


# indices that carry a non-empty glyph (idx 4 is empty -> center stays gray)
KIDX = [0, 1, 2, 3, 5, 6, 7, 8]


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    glyphs = [_glyph(k) for k in range(9)]
    comps = [1.0 - glyphs[k] for k in range(9)]
    cnt_comp = np.array([comps[k].sum() for k in KIDX], np.float32)   # gray-cell counts
    cnt_gly = np.array([glyphs[k].sum() for k in KIDX], np.float32)   # color-cell counts
    NK = len(KIDX)  # 8

    # Sprites live within the top-left WORK x WORK corner (grid <= 16x16, sprite
    # top-lefts <= H-3,W-3 <= 13).  Slice everything to WORK=16 so the matched-
    # filter conv planes are [1,8,14,14] (3136B) not [1,8,28,28] (12544B).
    WORK = 16

    # ---- gray channel of the 16x16 corner -> fp16 ----
    init("g_st", np.array([0, 5, 0, 0], np.int64), np.int64)
    init("g_en", np.array([1, 6, WORK, WORK], np.int64), np.int64)
    init("g_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "g_st", "g_en", "g_ax"], "gray32")   # [1,1,16,16] f32
    n("Cast", ["gray32"], "gray", to=TensorProto.FLOAT16)     # fp16

    # ---- colorval = sum_c c*input[c]: 1x1 channel-collapse conv on the FREE
    # 10-ch input (fp32, exact small ints), then Slice to the 16x16 corner and
    # cast fp16.  (Collapsing 10ch->1ch BEFORE slicing avoids a 10240B fp32
    # 10-channel corner slice.) ----
    cw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("colw", cw, np.float32)
    n("Conv", ["input", "colw"], "colorval30", pads=[0, 0, 0, 0])   # [1,1,30,30] f32
    init("c_st", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("c_en", np.array([1, 1, WORK, WORK], np.int64), np.int64)
    init("c_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["colorval30", "c_st", "c_en", "c_ax"], "colorval32")  # [1,1,16,16] f32
    n("Cast", ["colorval32"], "colorval", to=TensorProto.FLOAT16)     # fp16

    # ---- FUSED single-conv combined-max recovery ----
    # Stack [colorval; gray] -> [1,2,16,16] and run ONE 8-output Conv whose
    # per-channel kernel is (glyph_k on the colorval plane, BIG*F_k on the gray
    # plane).  The conv yields  colorsum_k + BIG*resp_k  at every window.  The
    # gray matched filter F_k = comp_k - 9*glyph_k satisfies resp_k <= cnt_comp_k
    # with EQUALITY ONLY at sprite k's true top-left, so subtracting BIG*cnt_comp
    # leaves combined_k = cnt_gly_k*color[k] at that one cell and a large negative
    # value everywhere else.  Then ReduceMax over space recovers cnt_gly_k*color[k]
    # with no detector/Where/picked plane (kills the bool det + picked planes and
    # the separate resp/colorsum planes -> ONE [1,8,14,14] plane total).
    # BIG must exceed (max non-hit colorsum 36) - (min hit value 3) = 33; pick 40
    # so every packed value |BIG*resp| <= 40*36 = 1440 stays fp16 integer-exact.
    BIG = 40.0
    n("Concat", ["colorval", "gray"], "stk", axis=1)               # [1,2,16,16] f16
    Kw = np.zeros((NK, 2, 3, 3), np.float32)
    for j, k in enumerate(KIDX):
        Kw[j, 0] = glyphs[k]                       # colorsum part
        Kw[j, 1] = BIG * (comps[k] - 9.0 * glyphs[k])  # BIG * F_k
    init("Kw", Kw.astype(np.float16), np.float16)
    # fold the -BIG*cnt_comp offset into the Conv BIAS -> no separate Sub plane
    init("Kb", (-BIG * cnt_comp).astype(np.float16), np.float16)
    n("Conv", ["stk", "Kw", "Kb"], "combined", pads=[0, 0, 0, 0])  # [1,8,14,14] f16
    n("ReduceMax", ["combined"], "csum", axes=[2, 3], keepdims=1)  # [1,8,1,1]
    init("cntgly", cnt_gly.reshape(1, NK, 1, 1).astype(np.float16), np.float16)
    n("Div", ["csum", "cntgly"], "color")                         # [1,8,1,1] f16 colors

    # ---- build 9x9 template: Tcol = sum_k color_k * placed_k ----
    # placed[k] is the glyph k stamped into block (k//3,k%3) of a 9x9 grid.
    placed = np.zeros((NK, 9, 9), np.float16)
    union = np.zeros((9, 9), np.float32)
    for j, k in enumerate(KIDX):
        br, bc = (k // 3) * 3, (k % 3) * 3
        placed[j, br:br + 3, bc:bc + 3] = glyphs[k]
        union[br:br + 3, bc:bc + 3] = np.maximum(union[br:br + 3, bc:bc + 3], glyphs[k])
    # Tcol = color contracted with placed over the 8 glyph channels via a tiny
    # MatMul (color[1,8] @ placed[8,81] -> [1,81]) -> reshape [1,1,9,9].  No
    # [1,8,9,9] tparts plane materialises.
    init("placedmat", placed.reshape(NK, 81), np.float16)         # [8,81]
    init("cshape", np.array([1, NK], np.int64), np.int64)
    n("Reshape", ["color", "cshape"], "colrow")                   # [1,8] f16
    n("MatMul", ["colrow", "placedmat"], "tflat")                 # [1,81] f16
    init("t9shape", np.array([1, 1, 9, 9], np.int64), np.int64)
    n("Reshape", ["tflat", "t9shape"], "Tcol")                    # [1,1,9,9] f16
    # mask of glyph cells (constant): bool initializer, no runtime op
    init("Mg", (union > 0.5).reshape(1, 1, 9, 9), np.bool_)        # [1,1,9,9] bool
    # L = where(Mg, round(Tcol) as uint8, 5)
    n("Cast", ["Tcol"], "Tu", to=TensorProto.UINT8)               # color value
    init("gray5", np.array(5, np.uint8), np.uint8)
    n("Where", ["Mg", "Tu", "gray5"], "L9")                       # [1,1,9,9] uint8

    # ---- pad L9 (9x9) -> 30x30 with sentinel 12 (never matches ch 0..9) ----
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - 9, 30 - 9], np.int64), np.int64)
    init("padval", np.array(12, np.uint8), np.uint8)
    n("Pad", ["L9", "padpads", "padval"], "L", mode="constant")   # [1,1,30,30] uint8

    # ---- final Equal -> free BOOL output ----
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                           # [1,10,30,30] bool

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
