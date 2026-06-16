"""task328 (ARC-AGI d22278a0) — corner-anchor nearest-colour with even-Chebyshev gate.

Rule (from ARC-GEN generator): the (square, side H = W in 6..18) input grid holds
2-4 coloured anchor pixels, each at a DISTINCT CORNER of the grid
((0,0), (H-1,0), (0,H-1), (H-1,H-1)).  For each in-grid cell (r,c) we find the
anchor with the minimal Manhattan distance |r-ar|+|c-ac|; if that minimiser is
UNIQUE and its Chebyshev distance max(|r-ar|,|c-ac|) is EVEN, the cell is painted
with that anchor's colour, otherwise it stays background.

Floor-break (label map + final Equal, 18x18 fp16 working canvas).  Every lever
that removed a full-grid plane:
  * H from per-row occupancy = ReduceMax(input, axes=[1,3]) -> [1,1,30,1] (120 B),
    NOT a 30x30 occupancy plane (-3480 B).
  * in-grid mask is SEPARABLE: square + fully bg-filled => ig = (r<H) AND (c<H),
    two tiny 1-D Less + one And -> bool (no 18x18 f32 occupancy slice).
  * 4 corner colour SCALARS via Gather(rows {0,H-1}) -> Gather(cols {0,H-1}) of the
    raw input (no full 30x30 colour Conv).
  * per corner only ONE 18x18 fp16 plane (dm) materialises: the ABSENT sentinel is
    folded into the 1-D row vector (dm = (adr+presbig) + adc), so no separate
    Manhattan plane.
  * Chebyshev parity with NO max/mod plane: parity(max(a,b)) = row parity when
    a>=b else col parity, via a 1-D parity multiplexer (even = adr_ev XOR (a<b AND
    (adr_ev XOR adc_ev))) -> 4 bool planes/corner, no fp16 che/chm planes.
  * uniqueness + winning colour + even-gate fold into ONE accumulation: each
    minimiser corner emits pk = 100 (+colour when Chebyshev EVEN); acc = Sum(pk) =
    100*cnt + even-colour.  acc==100 -> odd (bg); acc in [101,109] -> paint colour
    acc-100; acc>=200 -> tie (bg).  Kills the ismf/cnt/ct/colour_f plane army.
A single uint8 label L (18x18) is Padded to 30x30 (sentinel 10) and fed to the
final Equal(L, arange[1,10,1,1]) -> free BOOL output.

  mem 23296, params 113 -> pts 14.939 (stored 14.28, +0.66).
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
    init("one", np.array(1.0, np.float32), np.float32)
    init("half16", np.array(0.5, np.float16), np.float16)
    init("two16", np.array(2.0, np.float16), np.float16)
    init("big16", np.array(BIG, np.float16), np.float16)
    init("ar_row", np.arange(WORK, dtype=np.float16).reshape(1, 1, WORK, 1), np.float16)
    init("ar_col", np.arange(WORK, dtype=np.float16).reshape(1, 1, 1, WORK), np.float16)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("Wk", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)
    init("v0u", np.array(0, np.uint8), np.uint8)
    init("v10u", np.array(10, np.uint8), np.uint8)
    init("ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    init("zero16", np.array(0.0, np.float16), np.float16)
    init("c100", np.array(100.0, np.float16), np.float16)
    init("c100h", np.array(100.5, np.float16), np.float16)
    init("c150", np.array(150.0, np.float16), np.float16)
    init("idx0_1", np.array([0], np.int64), np.int64)
    init("shp1", np.array([1], np.int64), np.int64)

    # ---- H recovery + separable in-grid mask (no 30x30 occupancy plane) ----
    # per-row occupancy directly: max over channel AND col axes -> [1,1,30,1] (120B)
    n("ReduceMax", ["input"], "rowocc", axes=[1, 3], keepdims=1)     # [1,1,30,1] f32
    n("ReduceSum", ["rowocc"], "Hf", keepdims=0)                   # scalar H (f32)
    n("Cast", ["Hf"], "Hf16", to=H16)                              # scalar H fp16
    n("Sub", ["Hf", "one"], "Hm1f")                                # scalar H-1 (f32)
    # grid is square & fully bg-filled in-grid -> in-grid = (r<H) AND (c<H)
    n("Less", ["ar_row", "Hf16"], "rowin")                         # [1,1,18,1] bool
    n("Less", ["ar_col", "Hf16"], "colin")                         # [1,1,1,18] bool
    n("And", ["rowin", "colin"], "ig_b")                           # [1,1,18,18] bool
    n("Cast", ["Hm1f"], "Hm1", to=H16)                             # scalar H-1 fp16
    n("Cast", ["Hm1f"], "Hm1_i", to=TensorProto.INT64)             # scalar H-1 int64

    # ---- recover the 4 corner colour scalars via Gather of the 4 corner cells ----
    # gather rows {0, H-1} then cols {0, H-1}: [1,10,2,2] of channel-onehots
    n("Reshape", ["Hm1_i", "shp1"], "Hm1_1")                       # [1]
    n("Concat", ["idx0_1", "Hm1_1"], "rcidx", axis=0)             # [2] = [0, H-1]
    n("Gather", ["input", "rcidx"], "grows", axis=2)              # [1,10,2,30]
    n("Gather", ["grows", "rcidx"], "gcells", axis=3)            # [1,10,2,2]
    n("Cast", ["gcells"], "gcells16", to=H16)
    n("Mul", ["gcells16", "Wk"], "gcw")                          # [1,10,2,2]
    n("ReduceSum", ["gcw"], "ccol", axes=[1], keepdims=1)        # [1,1,2,2] colour idx
    # ccol[0, 0, i, j] = colour of corner (row=i*(H-1), col=j*(H-1))
    # corners order: (0,0),(H-1,0),(0,H-1),(H-1,H-1) -> (i,j)=(0,0),(1,0),(0,1),(1,1)
    cell_idx = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}

    # ---- 1-D distance vectors (only 2 distinct rows, 2 distinct cols) ----
    # row dist to top = r ; to bottom = (H-1)-r.  col dist to left = c ; to right = (H-1)-c.
    # adr[s], adc[s] for s in {0:near-side(0), 1:far-side(H-1)}.  Parities are tiny 1-D.
    adr = {}; adc = {}; adr_ev = {}; adc_ev = {}
    for s, kk in ((0, "zero16"), (1, "Hm1")):
        n("Sub", ["ar_row", kk], f"dr{s}"); n("Abs", [f"dr{s}"], f"adr{s}")   # [1,1,18,1]
        n("Sub", ["ar_col", kk], f"dc{s}"); n("Abs", [f"dc{s}"], f"adc{s}")   # [1,1,1,18]
        adr[s], adc[s] = f"adr{s}", f"adc{s}"
        n("Mod", [f"adr{s}", "two16"], f"adrm{s}", fmod=1)
        n("Less", [f"adrm{s}", "half16"], f"adre{s}")   # bool even (row), [1,1,18,1]
        n("Mod", [f"adc{s}", "two16"], f"adcm{s}", fmod=1)
        n("Less", [f"adcm{s}", "half16"], f"adce{s}")   # bool even (col), [1,1,1,18]
        adr_ev[s], adc_ev[s] = f"adre{s}", f"adce{s}"

    # corner -> (row-side, col-side): (0,0),(H-1,0),(0,H-1),(H-1,H-1)
    sides = [(0, 0), (1, 0), (0, 1), (1, 1)]

    dm_names, even_names, col_names = [], [], []
    for k, (rs, cs) in enumerate(sides):
        ar, ac = adr[rs], adc[cs]
        # corner colour scalar: slice ccol at this corner's (i,j)
        ii, jj = cell_idx[k]
        init(f"cst{k}", np.array([0, 0, ii, jj], np.int64), np.int64)
        init(f"cen{k}", np.array([1, 1, ii + 1, jj + 1], np.int64), np.int64)
        n("Slice", ["ccol", f"cst{k}", f"cen{k}", "ax"], f"colk{k}")  # [1,1,1,1]
        col_names.append(f"colk{k}")
        n("Add", [f"colk{k}", "c100"], f"colk100_{k}")            # scalar 100+colour

        # Manhattan dist + ABSENT-sentinel folded into the 1-D row vector (no man
        # plane): dm = (adr + presbig) + adc, where presbig = BIG when absent else 0.
        n("Greater", [f"colk{k}", "half16"], f"pres{k}")          # scalar bool present
        n("Where", [f"pres{k}", "zero16", "big16"], f"pbig{k}")  # scalar 0 or BIG
        n("Add", [ar, f"pbig{k}"], f"arp{k}")                     # [1,1,18,1] tiny
        n("Add", [f"arp{k}", ac], f"dm{k}")                       # [1,1,18,18]
        dm_names.append(f"dm{k}")

        # Chebyshev parity WITHOUT a max/mod plane: parity(max(a,b)) is the row-side
        # parity when a>=b, else the col-side parity.  Multiplexer in 2 bool ops:
        #   even = adr_ev XOR ( (a<b) AND (adr_ev XOR adc_ev) )
        # dxor is built from the two 1-D parity vectors (broadcast), shared structure.
        n("Xor", [adr_ev[rs], adc_ev[cs]], f"dxor{k}")           # bool [1,1,18,18]
        n("Less", [ar, ac], f"rlt{k}")                            # bool a<b
        n("And", [f"rlt{k}", f"dxor{k}"], f"sw{k}")             # flip when col side & differ
        n("Xor", [adr_ev[rs], f"sw{k}"], f"even{k}")            # bool even
        even_names.append(f"even{k}")

    # ---- min Manhattan + uniqueness + winning colour in ONE accumulation ----
    # Each minimiser corner emits a packed value: 100 (always, for the tie count)
    # plus its colour when its Chebyshev is EVEN.  acc = 100*cnt + colour_of_even_min.
    #   acc == 100          -> unique min, ODD  -> background
    #   acc in [101,109]    -> unique min, EVEN -> paint colour (acc-100)
    #   acc >= 200          -> tie               -> background
    n("Min", dm_names, "mind")
    pack_terms = []
    for k in range(4):
        n("Equal", [dm_names[k], "mind"], f"ism{k}")              # bool min-here
        # eo = (100 + colour) if even else 100  (colour added only on even cells)
        n("Where", [even_names[k], f"colk100_{k}", "c100"], f"eo{k}")
        n("Where", [f"ism{k}", f"eo{k}", "zero16"], f"pk{k}")     # 0 if not min
        pack_terms.append(f"pk{k}")
    n("Sum", pack_terms, "acc")                                  # 100*cnt + even-colour
    n("Less", ["acc", "c150"], "unique")                         # acc < 150  (cnt==1)
    n("Greater", ["acc", "c100h"], "haspaint")                   # acc > 100.5 (even win)
    n("Sub", ["acc", "c100"], "colour_f")                       # colour (valid when paint)
    n("Cast", ["colour_f"], "colour_u", to=TensorProto.UINT8)

    # ---- paint mask & label map ----
    n("And", ["unique", "haspaint"], "pe")
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
