"""Task 308 (c8cbb738): reconstruct the symmetric (2h+1)x(2h+1) square.

Rule (from ARC-GEN generator): the input is a large bg grid in which several
"broken border" elements are scattered.  Each element draws a small set of
non-bg pixels in 4-fold (corner elements) or 8-fold (the on-axis element)
symmetry about its OWN centre.  Each element uses a DISTINCT colour
(`random_colors`, exclude=bg) and one offset on the perimeter of a Chebyshev
square of radius h (= halfsize).  The OUTPUT is the single (2h+1)x(2h+1) square
that all these fragments reconstruct: bg everywhere except colour c at the
symmetric positions of its offset relative to the centre (h,h).

Because each colour belongs to exactly ONE element/centre, the global bounding
box of colour c directly yields its offset magnitudes:
    ar = (rmax-rmin)/2 ,  ac = (cmax-cmin)/2 ,  h = max over c of max(ar,ac).
Pattern type per colour:
    * corner element  -> coloured at (h +- ar, h +- ac)            (4 corners)
    * on-axis element -> coloured at (h, 0),(h,2h),(0,h),(2h,h)    (plus shape)
The two are distinguished by how many pixels of colour c sit in its top bbox
row: a corner element puts 2 there, the on-axis element puts 1.

Construction: recover per-colour (rmin,rmax,cmin,cmax) as [1,10,1,1] scalars
from per-row / per-col presence reductions, derive (ar,ac,h), build a per-cell
[1,10,7,7] pattern mask, fold to a [1,1,7,7] colour-index plane, pad to 30x30
and write into the FREE bool output via Equal against colour ids.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

K = 7  # max output side = 2*hmax+1 = 7 (hmax = 3)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    f32 = TensorProto.FLOAT
    f16 = TensorProto.FLOAT16

    # ---- constants ----
    init("ZEROh", np.array(0.0, np.float16), np.float16)
    init("HALFh", np.array(0.5, np.float16), np.float16)
    init("ONE5h", np.array(1.5, np.float16), np.float16)
    init("BIG", np.array(99.0, np.float16), np.float16)
    init("NEG", np.array(-99.0, np.float16), np.float16)
    init("ROW30", np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1), np.float16)
    init("COL30", np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30), np.float16)
    init("ROW7", np.arange(K, dtype=np.float16).reshape(1, 1, K, 1), np.float16)
    init("COL7", np.arange(K, dtype=np.float16).reshape(1, 1, 1, K), np.float16)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - K, 30 - K], np.int64), np.int64)

    # ---- background channel = most-frequent colour (argmax of channel count) ----
    # A coloured pixel CAN land at the grid corner, so bg != corner cell; the bg
    # fill always dominates the pixel count, so argmax of per-channel total is bg.
    n("ReduceSum", ["input"], "chan_cnt", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    n("ArgMax", ["chan_cnt"], "bg_i64", axis=1, keepdims=1)         # [1,1,1,1] i64
    n("Equal", ["colk_i", "bg_i64"], "bg_ch_b")                     # [1,10,1,1]
    init("colk_i", np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1), np.int64)
    n("Not", ["bg_ch_b"], "not_bg_b")                               # [1,10,1,1]

    # ---- per-channel row counts (fp16) -> presence + rmin/rmax + top-row count ----
    # ONE fp32 entry reduction, then cast to fp16 and keep all full-canvas
    # work in fp16 (half the bytes; ORT ReduceMin/Max accept fp16).
    n("ReduceSum", ["input"], "row_cnt_f32", axes=[3], keepdims=1)  # [1,10,30,1] f32
    n("Cast", ["row_cnt_f32"], "row_cnt_h", to=f16)                 # [1,10,30,1] f16
    n("Greater", ["row_cnt_h", "ZEROh"], "row_has")                 # [1,10,30,1] bool
    n("Where", ["row_has", "ROW30", "BIG"], "row_min_src")
    n("ReduceMin", ["row_min_src"], "rmin", axes=[2], keepdims=1)   # [1,10,1,1] f16
    n("Where", ["row_has", "ROW30", "NEG"], "row_max_src")
    n("ReduceMax", ["row_max_src"], "rmax", axes=[2], keepdims=1)

    # ---- per-channel col presence (fp16) -> cmin/cmax ----
    n("ReduceMax", ["input"], "cpres_f32", axes=[2], keepdims=1)    # [1,10,1,30] f32
    n("Cast", ["cpres_f32"], "cpres_h", to=f16)
    n("Greater", ["cpres_h", "ZEROh"], "col_has")
    n("Where", ["col_has", "COL30", "BIG"], "col_min_src")
    n("ReduceMin", ["col_min_src"], "cmin", axes=[3], keepdims=1)
    n("Where", ["col_has", "COL30", "NEG"], "col_max_src")
    n("ReduceMax", ["col_max_src"], "cmax", axes=[3], keepdims=1)

    # ---- offsets & half size (all fp16 scalars); exclude bg channel HERE ----
    # bg channel fills the whole grid -> spurious huge bbox; mask it at scalar level.
    n("Sub", ["rmax", "rmin"], "rh0")     # 2*ar (incl bg) [1,10,1,1]
    n("Sub", ["cmax", "cmin"], "cw0")     # 2*ac (incl bg)
    n("Where", ["not_bg_b", "rh0", "NEGh"], "rh")  # bg -> -1 (never a valid extent)
    init("NEGh", np.array(-1.0, np.float16), np.float16)
    n("Where", ["not_bg_b", "cw0", "NEGh"], "cw")
    n("Mul", ["rh", "HALFh"], "ar")
    n("Mul", ["cw", "HALFh"], "ac")
    n("Max", ["rh", "cw"], "extent")      # 2*h for that channel
    n("Greater", ["extent", "ZEROh"], "present_b")  # colour used (bg -> false)
    n("ReduceMax", ["extent"], "ext_max", axes=[1, 2, 3], keepdims=1)  # scalar 2*h
    n("Mul", ["ext_max", "HALFh"], "Rh")  # h scalar [1,1,1,1]

    # ---- rect vs diamond: count of colour in its top bbox row ----
    n("Equal", ["ROW30", "rmin"], "is_top_b")          # [1,10,30,1]
    n("Where", ["is_top_b", "row_cnt_h", "ZEROh"], "top_cnt_img")  # f16
    n("ReduceSum", ["top_cnt_img"], "top_cnt", axes=[2], keepdims=1)  # [1,10,1,1]
    n("Greater", ["top_cnt", "ONE5h"], "rect_b")      # >1 pixel in top row -> corner
    n("Not", ["rect_b"], "diamond_b")

    # ---- build per-cell pattern in the 7x7 ----
    n("Sub", ["ROW7", "Rh"], "dr")     # [1,1,7,1] - h    (broadcast scalar)
    n("Sub", ["COL7", "Rh"], "dc")     # [1,1,1,7]
    n("Abs", ["dr"], "adr")
    n("Abs", ["dc"], "adc")
    n("Equal", ["adr", "ar"], "adr_eq")   # [1,10,7,1]
    n("Equal", ["adc", "ac"], "adc_eq")   # [1,10,1,7]
    n("Equal", ["adr", "ZEROh"], "adr0")   # [1,1,7,1]
    n("Equal", ["adc", "ZEROh"], "adc0")
    # corner: |dr|==ar AND |dc|==ac
    n("And", ["adr_eq", "adc_eq"], "rect_pat")          # [1,10,7,7]
    # diamond: (|dr|==ar AND dc==0) OR (dr==0 AND |dc|==ac)
    n("And", ["adr_eq", "adc0"], "dia_v")
    n("And", ["adr0", "adc_eq"], "dia_h")
    n("Or", ["dia_v", "dia_h"], "dia_pat")
    # per-channel select corner-vs-diamond: pat = (rect AND rect_pat) OR (~rect AND dia_pat)
    n("And", ["rect_b", "rect_pat"], "rect_mark")
    n("And", ["diamond_b", "dia_pat"], "dia_mark")
    n("Or", ["rect_mark", "dia_mark"], "pat")            # [1,10,7,7] bool
    n("And", ["present_b", "pat"], "mark")               # [1,10,7,7] bool

    # ---- fold to a colour-index plane and write the FREE output ----
    # Pack colour+1 (so colour 0 stays distinguishable from "no mark") into ONE
    # fp16 plane; exactly one channel marks each cell, so a channel-sum recovers it.
    n("Where", ["mark", "colp1h", "ZEROh"], "mark_col")  # [1,10,7,7] fp16 (colour+1)
    init("colp1h", (np.arange(10) + 1).astype(np.float16).reshape(1, 10, 1, 1), np.float16)
    n("ReduceSum", ["mark_col"], "s7", axes=[1], keepdims=1)  # [1,1,7,7] = colour+1 or 0
    n("Sub", ["s7", "ONEh"], "L7")                       # colour index where marked
    init("ONEh", np.array(1.0, np.float16), np.float16)
    n("Greater", ["s7", "ZEROh"], "anymark_b")           # [1,1,7,7] marked?
    n("Cast", ["bg_i64"], "bgcol", to=f16)               # [1,1,1,1] bg colour id
    n("Where", ["anymark_b", "L7", "bgcol"], "Lcell")    # [1,1,7,7] fp16 colour idx
    # restrict to the actual (2h+1)x(2h+1) region: cells outside -> sentinel -1
    n("Add", ["ext_max", "ONEh"], "side")                # 2h+1 (fp16)
    n("Less", ["ROW7", "side"], "row_in")                # [1,1,7,1]
    n("Less", ["COL7", "side"], "col_in")                # [1,1,1,7]
    n("And", ["row_in", "col_in"], "in_b")               # [1,1,7,7]
    # in-region cells keep colour index; outside -> sentinel -1 (fp16 Where)
    n("Where", ["in_b", "Lcell", "padvalh"], "Lin")      # fp16 [1,1,7,7]
    init("padvalh", np.array(-1.0, np.float16), np.float16)
    # cast to uint8 then pad (uint8 pad = 900B vs fp16 1800B); -1 -> 255 sentinel
    n("Cast", ["Lin"], "Lu", to=TensorProto.UINT8)       # [1,1,7,7] uint8 (255 outside)
    n("Pad", ["Lu", "padpads", "padval2"], "Lpad", mode="constant")  # [1,1,30,30] u8
    init("padval2", np.array(254, np.uint8), np.uint8)
    n("Equal", ["Lpad", "colk_u"], "output")             # -> BOOL output (FREE)
    init("colk_u", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)

    x = helper.make_tensor_value_info("input", f32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task308", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
