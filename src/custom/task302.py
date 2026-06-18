"""task302 (ARC-AGI c0f76784) — fill each hollow gray box's interior, colour by size.

Rule (generator task_c0f76784.py, size=12 fixed): 3 (sometimes 2) non-overlapping
square gray (=5) box outlines of total side length L in {3,4,5} sit on a 12x12 grid,
each hollow (its (L-2)x(L-2) interior is background black=0).  The OUTPUT keeps every
gray frame and FILLS each interior solid with colour 5+(L-2) = 3+L:
  L=3 (1x1 hole) -> 6 ; L=4 (2x2 hole) -> 7 ; L=5 (3x3 hole) -> 8.
Equivalently every interior cell's fill = 5 + s where s = hole side (=L-2).

Closed-form per-cell rule (verified 0/3000 fresh, variant L; final net 500/500 fresh):
  Only the gray plane G (input ch5, sliced to 12x12) is needed.  Because boxes are
  generated >=1 cell apart, ALONG ONE ROW within +-2 cells there is at most ONE gray
  (a box frame), so the nearest-gray distance is a LINEAR weighted conv (no argmin):
    valsum = 2*G[c-1] + G[c-2] + 2*G[c+1] + G[c+2]            (= 5 - s for hole cells)
    fill   = 10 - valsum  in {6,7,8}                          (= 5 + s)
  Enclosure (a black cell is a hole interior iff a gray frame lies within 3 cells in ALL
  FOUR directions):  gl,gr,gu,gd = within-3 directional gray sums; enclosed <=> their
  channel-product > 0.  interior = enclosed AND non-gray (in-grid only has {black,gray},
  so non-gray == black -> no separate black-channel slice).  No flood-fill, no run-length
  products: the hole SIZE is read purely from distances to the two flanking gray frames.

Encoding (all on the 12x12 active region, fp16 12x12 plane = 288B):
  - G = Cast(Slice(input,ch5,12x12)) fp16.
  - valsum: one [1,1,1,7] Conv on G; fill = 10 - valsum.
  - enclosure: ONE [4,1,7,7] Conv packs gl/gr/gu/gd; ReduceProd over the channel axis
    -> encv; encG = encv > 0; interior = (G==0) AND encG.
  - L = Where(interior, fill, 5*G)  -> colour index in {0,5,6,7,8}; Cast uint8.
  - Pad L to 30x30 with sentinel 99 (off-grid matches no colour -> all-zero one-hot =
    the off-canvas target; in-grid black bg keeps L=0 -> ch0=1, so NO ingrid mask needed).
  - output = Equal(L_u8, arange[0..9]_u8) -> BOOL [1,10,30,30] routed into the FREE output.
  pts 16.45, mem 4932, params 234, 266/266 stored, 500/500 fresh.  Dominant intermediate =
  the [1,4,12,12] fp16 enclosure conv stack (1152B) + the [1,1,30,30] uint8 padded L (900B).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

N = 12  # active grid is always 12x12


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- slice gray (ch5) and black (ch0) to the 12x12 active region --------
    init("g_s", np.array([5, 0, 0], np.int64), np.int64)
    init("g_e", np.array([6, N, N], np.int64), np.int64)
    init("g_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "g_s", "g_e", "g_ax"], "G_f32")    # [1,1,12,12] gray
    n("Cast", ["G_f32"], "G", to=F16)                       # gray fp16 {0,1}

    # ---- value & enclosure from gray convs -----------------------------------
    # Because boxes are >=1 cell apart (generator margin), within any width-7 window
    # along a row there is AT MOST ONE gray (a box frame), so the nearest-gray
    # distance is read by a LINEAR weighted conv (no argmin):
    #   valsum = 2*G[c-1]+G[c-2] + 2*G[c+1]+G[c+2]   (= 5 - s for interior cells)
    #   fill   = 10 - valsum  in {6,7,8}.
    # Enclosure: a black cell is a hole interior iff a gray frame lies within 3 cells
    # in ALL FOUR directions.  gl/gr/gu/gd = within-3 gray sums; interior <=> their
    # product > 0 AND the cell is black.
    # valsum (one 1x7 channel) -> fill = 10 - valsum
    Vw = np.zeros((1, 1, 1, 7), np.float16)
    Vw[0, 0, 0, 3 - 1] = 2.0; Vw[0, 0, 0, 3 - 2] = 1.0
    Vw[0, 0, 0, 3 + 1] = 2.0; Vw[0, 0, 0, 3 + 2] = 1.0
    init("Vw", Vw, np.float16)
    n("Conv", ["G", "Vw"], "valsum", pads=[0, 3, 0, 3])    # [1,1,12,12] fp16
    init("TEN", np.array(10.0, np.float16), np.float16)
    n("Sub", ["TEN", "valsum"], "fillv")        # 6..8 fp16

    # enclosure: four within-3 directional gray sums packed in ONE 4-channel conv,
    # then their channel-product > 0 (gray present in all 4 directions).
    Ek = np.zeros((4, 1, 7, 7), np.float16)
    Ek[0, 0, 3, [3 - 1, 3 - 2, 3 - 3]] = 1.0               # gl (left)
    Ek[1, 0, 3, [3 + 1, 3 + 2, 3 + 3]] = 1.0               # gr (right)
    Ek[2, 0, [3 - 1, 3 - 2, 3 - 3], 3] = 1.0               # gu (up)
    Ek[3, 0, [3 + 1, 3 + 2, 3 + 3], 3] = 1.0               # gd (down)
    init("Ek", Ek, np.float16)
    n("Conv", ["G", "Ek"], "encstack", pads=[3, 3, 3, 3])  # [1,4,12,12] fp16
    n("ReduceProd", ["encstack"], "encv", axes=[1], keepdims=1)  # [1,1,12,12]
    init("Z", np.array(0.0, np.float16), np.float16)
    n("Greater", ["encv", "Z"], "encG")         # bool [1,1,12,12]
    # a hole-interior cell is enclosed AND non-gray (in-grid only 2 colours, so
    # non-gray == black); avoids slicing the black channel separately.
    n("Equal", ["G", "Z"], "nGray")             # bool: cell is not gray
    n("And", ["nGray", "encG"], "interior")     # bool [1,1,12,12]

    # ---- colour-index plane L = where(interior, fill, 5*G) -------------------
    init("FIVE", np.array(5.0, np.float16), np.float16)
    n("Mul", ["G", "FIVE"], "grayv")           # 0 or 5 (non-interior keeps gray/black)
    n("Where", ["interior", "fillv", "grayv"], "L12f")   # 0/5/6/7/8 fp16
    n("Cast", ["L12f"], "L12", to=U8)          # uint8 (Pad accepts uint8, halves bytes)

    # ---- pad to 30x30 with sentinel 99 (off-grid matches no colour -> all-zero
    #      one-hot = off-canvas target; in-grid black bg keeps L=0 -> ch0=1) -----
    init("Lpads", np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64), np.int64)
    init("SENT", np.array(99, np.uint8), np.uint8)
    n("Pad", ["L12", "Lpads", "SENT"], "L30", mode="constant")  # [1,1,30,30] u8

    # ---- output = Equal(L, arange[0..9]) -> BOOL [1,10,30,30] (FREE output) ----
    init("arange", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "arange"], "output")     # [1,10,30,30] bool

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task302", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
