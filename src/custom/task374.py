"""task374 (ARC-AGI ea32f347) — recolor 3 gray lines by length rank.

Rule (from the generator):
  A 10x10 grid holds 3 gray (color 5) line segments, each horizontal (up=0) or
  vertical (up=1), of DISTINCT lengths (2..9).  Lines never touch (parallel lines
  keep gap >= 2; the mixed-orientation case is also kept separated) so each gray
  cell belongs to exactly one line and no two lines are adjacent.  The output
  recolors each line by its LENGTH RANK among the three:
      shortest -> 2, middle -> 4, longest -> 1.

  Because lines are isolated:
    - A gray cell belonging to a HORIZONTAL line has a horizontal gray neighbour
      (and no vertical gray neighbour); a VERTICAL-line cell is the opposite.
    - A horizontal line occupies a single row, so its length =
        sum over that row of (gray AND has-horizontal-gray-neighbour).
      (Vertical-line cells crossing the same row contribute 0 to that sum.)
    - Symmetrically a vertical line's length = the per-column sum of vertical
      cells.
  So each gray cell's line length L is a broadcast of a tiny per-row / per-col
  1-D length vector — no full-canvas run-length product chains needed.

  The three lengths are distinct, so over gray cells:
      L == global min  -> shortest -> 2
      L == global max  -> longest  -> 1
      else             -> middle   -> 4

Encoding (10x10 active canvas, 10-ch expansion routed into the FREE bool output):
  gray g[1,1,10,10] (channel 5).  hcell/vcell via two col/row shifts + Or + And.
  Hlen[1,1,10,1]=ReduceSum(hcell,col); Vlen[1,1,1,10]=ReduceSum(vcell,row).
  Lcell = hcell*Hlen + vcell*Vlen.  Lmin/Lmax = global reductions.
  Label map L[1,1,10,10] uint8 -> Pad 30x30 (sentinel 99) -> Equal(L, arange).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

W = 10  # active canvas (grid is always 10x10 for this task)


def build(task):
    inits, nodes = [], []
    counter = [0]

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out=None, **attrs):
        if out is None:
            counter[0] += 1
            out = f"t{counter[0]}"
        nodes.append(helper.make_node(op, list(ins), [out], **attrs))
        return out

    # ---- gray mask g[1,1,10,10] (channel 5) --------------------------------
    init("g_s", np.array([5, 0, 0], np.int64), np.int64)
    init("g_e", np.array([6, W, W], np.int64), np.int64)
    init("g_ax", np.array([1, 2, 3], np.int64), np.int64)
    g_f32 = n("Slice", ["input", "g_s", "g_e", "g_ax"])  # [1,1,10,10] f32
    g = n("Cast", [g_f32], to=F16)                        # [1,1,10,10] f16 {0,1}
    init("ZERO16", np.array(0.0, np.float16), np.float16)

    def shift(src, axis, amt):
        # out[.., i] = src[.., i+amt]; amt>0 looks forward, amt<0 looks back.
        if amt > 0:
            s = init(f"s_{counter[0]}", np.array([amt], np.int64), np.int64)
            e = init(f"e_{counter[0]}", np.array([W], np.int64), np.int64)
            a = init(f"a_{counter[0]}", np.array([axis], np.int64), np.int64)
            sl = n("Slice", [src, s, e, a])
            pads = [0] * 8
            pads[axis + 4] = amt
            p = init(f"p_{counter[0]}", np.array(pads, np.int64), np.int64)
            return n("Pad", [sl, p, "ZERO16"], mode="constant")
        else:
            amt = -amt
            s = init(f"s_{counter[0]}", np.array([0], np.int64), np.int64)
            e = init(f"e_{counter[0]}", np.array([W - amt], np.int64), np.int64)
            a = init(f"a_{counter[0]}", np.array([axis], np.int64), np.int64)
            sl = n("Slice", [src, s, e, a])
            pads = [0] * 8
            pads[axis] = amt
            p = init(f"p_{counter[0]}", np.array(pads, np.int64), np.int64)
            return n("Pad", [sl, p, "ZERO16"], mode="constant")

    # ---- classify gray cells: horizontal-line vs vertical-line -------------
    gL = shift(g, 3, 1)   # gray to the right (col+1)
    gR = shift(g, 3, -1)  # gray to the left  (col-1)
    hnb = n("Add", [gL, gR])            # >0 if a horizontal gray neighbour exists
    hcell = n("Mul", [g, hnb])          # g * hnb ; >0 only on H-line cells
    init("HALF", np.array(0.5, np.float16), np.float16)
    hcell = n("Greater", [hcell, "HALF"])      # bool H-cell
    hcell = n("Cast", [hcell], to=F16)         # {0,1} f16

    gU = shift(g, 2, 1)   # gray below (row+1)
    gD = shift(g, 2, -1)  # gray above (row-1)
    vnb = n("Add", [gU, gD])
    vcell = n("Mul", [g, vnb])
    vcell = n("Greater", [vcell, "HALF"])
    vcell = n("Cast", [vcell], to=F16)

    # ---- per-row horizontal length, per-col vertical length ----------------
    Hlen = n("ReduceSum", [hcell], axes=[3], keepdims=1)  # [1,1,10,1] f16
    Vlen = n("ReduceSum", [vcell], axes=[2], keepdims=1)  # [1,1,1,10] f16

    # per-cell line length = hcell*Hlen(row) + vcell*Vlen(col)
    Lh = n("Mul", [hcell, Hlen])  # broadcast [1,1,10,1] over cols
    Lv = n("Mul", [vcell, Vlen])  # broadcast [1,1,1,10] over rows
    Lcell = n("Add", [Lh, Lv])    # [1,1,10,10] f16, line length on gray, 0 bg

    # ---- global Lmin / Lmax over GRAY cells --------------------------------
    BIG = 100.0
    init("BIG16", np.array(BIG, np.float16), np.float16)
    gb = n("Greater", [g, "ZERO16"])  # bool gray mask
    lc_for_min = n("Where", [gb, Lcell, "BIG16"])
    Lmin = n("ReduceMin", [lc_for_min], axes=[2, 3], keepdims=1)  # [1,1,1,1]
    Lmax = n("ReduceMax", [Lcell], axes=[2, 3], keepdims=1)        # [1,1,1,1]

    # ---- per-cell rank -> color label --------------------------------------
    is_min = n("Equal", [Lcell, Lmin])
    is_max = n("Equal", [Lcell, Lmax])
    is_min_g = n("And", [is_min, gb])
    is_max_g = n("And", [is_max, gb])

    init("L1", np.array(1, np.uint8), np.uint8)
    init("L2", np.array(2, np.uint8), np.uint8)
    init("L4", np.array(4, np.uint8), np.uint8)
    init("L0", np.array(0, np.uint8), np.uint8)
    # base: gray -> 4 (middle), bg -> 0 ; then override min/max gray cells.
    lab = n("Where", [gb, "L4", "L0"])
    lab = n("Where", [is_min_g, "L2", lab])
    lab = n("Where", [is_max_g, "L1", lab])  # [1,1,10,10] uint8

    # ---- pad to 30x30 with bg sentinel -------------------------------------
    init("pads30", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("SENT", np.array(99, np.uint8), np.uint8)
    L30 = n("Pad", [lab, "pads30", "SENT"], mode="constant")  # [1,1,30,30] u8

    arange = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("arange", arange, np.uint8)
    n("Equal", [L30, "arange"], "output")  # [1,10,30,30] BOOL

    graph = helper.make_graph(
        nodes, "task374",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
