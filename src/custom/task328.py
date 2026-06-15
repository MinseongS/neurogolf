"""task328 (ARC-AGI d22278a0) — corner-anchor nearest-colour with even-Chebyshev gate.

Rule (from ARC-GEN generator): the (square, side H = W in 6..18) input grid holds
2-4 coloured anchor pixels, each at a DISTINCT CORNER of the grid
((0,0), (H-1,0), (0,W-1), (H-1,W-1)).  For each in-grid cell (r,c) we find the
anchor with the minimal Manhattan distance |r-ar|+|c-ac|; if that minimiser is
UNIQUE and its Chebyshev distance max(|r-ar|,|c-ac|) is EVEN, the cell is painted
with that anchor's colour, otherwise it stays background.

Floor-break (label map + final Equal, 18x18 fp16 working canvas):
  Anchors live only at the 4 corners, so everything reduces to four per-corner
  distance planes built from 1-D row/col index ramps.  H <= 18, so the whole grid
  fits in the top-left WORK=18 corner; the colour-index plane and in-grid mask are
  Sliced to 18x18 up front and every per-cell plane is fp16 (648 B).  Per corner
  we form the Manhattan plane (BIG sentinel when the corner is empty), the
  Chebyshev-even mask, and the corner colour (one-hot row/col select + reduce).
  Min over the 4 Manhattan planes + minimiser count gives uniqueness; the winning
  colour and even-flag are picked by summing col_k * ismin_k.  A single uint8
  label L (18x18) is Padded to 30x30 (sentinel 10) and fed to the final
  Equal(L, arange[1,10,1,1]) -> free BOOL output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 18
BIG = 999.0


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
    H16 = TensorProto.FLOAT16

    # ---- constants ----
    init("half", np.array(0.5, np.float32), np.float32)
    init("one", np.array(1.0, np.float32), np.float32)
    init("half16", np.array(0.5, np.float16), np.float16)
    init("two16", np.array(2.0, np.float16), np.float16)
    init("big16", np.array(BIG, np.float16), np.float16)
    init("c1half16", np.array(1.5, np.float16), np.float16)
    init("zero16", np.array(0.0, np.float16), np.float16)
    init("ar_row", np.arange(WORK, dtype=np.float16).reshape(1, 1, WORK, 1), np.float16)
    init("ar_col", np.arange(WORK, dtype=np.float16).reshape(1, 1, 1, WORK), np.float16)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("Wcol", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("v0u", np.array(0, np.uint8), np.uint8)
    init("v10u", np.array(10, np.uint8), np.uint8)
    init("st", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("en", np.array([1, 1, WORK, WORK], np.int64), np.int64)
    init("ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # ---- in-grid mask, colour-index plane, recover H (all sliced to 18x18) ----
    n("ReduceMax", ["input"], "ig_raw", axes=[1], keepdims=1)        # [1,1,30,30] f32
    n("Conv", ["input", "Wcol"], "cidx_full")                       # [1,1,30,30] colour
    n("Slice", ["ig_raw", "st", "en", "ax"], "ig_raw18")            # [1,1,18,18]
    n("Slice", ["cidx_full", "st", "en", "ax"], "cidx_f18")        # [1,1,18,18] f32
    n("Greater", ["ig_raw18", "half"], "ig_b")                     # bool in-grid
    n("Cast", ["cidx_f18"], "cidx", to=H16)                        # fp16 colour plane

    # recover H (square => W = H) from in-grid occupancy
    n("Cast", ["ig_b"], "igf", to=F)                               # 0/1 f32 (18x18)
    n("ReduceMax", ["igf"], "rowocc", axes=[3], keepdims=1)        # [1,1,18,1]
    n("ReduceSum", ["rowocc"], "Hf", keepdims=0)                   # scalar H
    n("Sub", ["Hf", "one"], "Hm1f")
    n("Cast", ["Hm1f"], "Hm1", to=H16)                            # scalar H-1 fp16
    init("zerosc", np.array(0.0, np.float16), np.float16)

    corners = [("zerosc", "zerosc"), ("Hm1", "zerosc"),
               ("zerosc", "Hm1"), ("Hm1", "Hm1")]

    dm_names, even_names, col_names = [], [], []
    for k, (kr, kc) in enumerate(corners):
        # 1-D abs distances ([1,1,18,1] / [1,1,1,18] fp16)
        n("Sub", ["ar_row", kr], f"dr{k}")
        n("Abs", [f"dr{k}"], f"adr{k}")
        n("Sub", ["ar_col", kc], f"dc{k}")
        n("Abs", [f"dc{k}"], f"adc{k}")
        # Manhattan + Chebyshev planes (18x18 fp16)
        n("Add", [f"adr{k}", f"adc{k}"], f"man{k}")
        n("Max", [f"adr{k}", f"adc{k}"], f"che{k}")
        n("Mod", [f"che{k}", "two16"], f"chm{k}", fmod=1)
        n("Less", [f"chm{k}", "half16"], f"even{k}")              # bool even
        even_names.append(f"even{k}")

        # corner colour: one-hot row/col select on the 18x18 colour plane + reduce
        n("Equal", ["ar_row", kr], f"rsel{k}")                    # [1,1,18,1] bool
        n("Equal", ["ar_col", kc], f"csel{k}")                    # [1,1,1,18] bool
        n("Cast", [f"rsel{k}"], f"rself{k}", to=H16)
        n("Cast", [f"csel{k}"], f"cself{k}", to=H16)
        n("Mul", ["cidx", f"rself{k}"], f"cm{k}")                 # [1,1,18,18] fp16
        n("ReduceSum", [f"cm{k}"], f"rowk{k}", axes=[2], keepdims=1)   # [1,1,1,18]
        n("Mul", [f"rowk{k}", f"cself{k}"], f"rck{k}")
        n("ReduceSum", [f"rck{k}"], f"colk{k}", axes=[3], keepdims=1)  # [1,1,1,1]
        col_names.append(f"colk{k}")
        n("Greater", [f"colk{k}", "half16"], f"pres{k}")          # bool present
        n("Where", [f"pres{k}", f"man{k}", "big16"], f"dm{k}")    # [1,1,18,18]
        dm_names.append(f"dm{k}")

    # ---- min Manhattan over corners + uniqueness ----
    n("Min", dm_names, "mind")
    ismin_f = []
    for k in range(4):
        n("Equal", [dm_names[k], "mind"], f"ism{k}")
        n("Cast", [f"ism{k}"], f"ismf{k}", to=H16)
        ismin_f.append(f"ismf{k}")
    n("Sum", ismin_f, "cnt")
    n("Less", ["cnt", "c1half16"], "unique")                     # cnt == 1

    # ---- winning colour & even-flag = sum_k col_k * ismin_k ----
    colour_terms, even_terms = [], []
    for k in range(4):
        n("Mul", [col_names[k], ismin_f[k]], f"ct{k}")
        colour_terms.append(f"ct{k}")
        n("Cast", [even_names[k]], f"evf{k}", to=H16)
        n("Mul", [f"evf{k}", ismin_f[k]], f"et{k}")
        even_terms.append(f"et{k}")
    n("Sum", colour_terms, "colour_f")
    n("Sum", even_terms, "even_f")
    n("Greater", ["even_f", "half16"], "even_win")
    n("Cast", ["colour_f"], "colour_u", to=TensorProto.UINT8)

    # ---- paint mask & label map ----
    n("And", ["unique", "even_win"], "pe")
    n("And", ["pe", "ig_b"], "paint")
    n("Where", ["ig_b", "v0u", "v10u"], "Lbg")
    n("Where", ["paint", "colour_u", "Lbg"], "L18")

    # ---- pad to 30x30 + final Equal -> free BOOL output ----
    n("Pad", ["L18", "padpads", "padval"], "L", mode="constant")
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task328", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
