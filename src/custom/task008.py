"""Task 008 (ARC-AGI 05f2a901): slide the red block to touch the cyan anchor.

A red(2) rectangular block (with black nibble-holes inside its bbox) and a 2x2
cyan(8) anchor sit in the grid, separated along exactly one axis (rows OR cols)
with a clear gap.  The red block translates rigidly along that axis toward the
cyan block until its facing edge is adjacent to cyan.  Cyan stays put.

Per-cell label map.  Let the block shift be (dr, dc) (only one nonzero):
  src cell of (r,c) is (r-dr, c-dc); the cell is part of the moved block iff that
  source lies in the red bbox.  Then
    L[r,c] = colourindex(input[r-dr, c-dc])   if source in red bbox
           = 8                                 if (r,c) is a cyan cell
           = 0                                 otherwise (background)
  The shifted block and cyan never overlap, so this is unambiguous.

  output = Equal(L, arange) into the free BOOL output (opset 11).

Geometry (all scalars from 1-D occupancy reductions):
  red rows disjoint from cyan rows -> vertical move, else horizontal move.
  vertical: red above cyan -> dr = (cyan_rmin-1) - red_rmax ;
            red below cyan -> dr = (cyan_rmax+1) - red_rmin .
  horizontal analogous on cols.  The "other" axis shift is 0.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

H = W = 30


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
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    init("BIG", np.array(1e6, np.float32), np.float32)
    init("nBIG", np.array(-1.0, np.float32), np.float32)
    init("zero", np.array(0.0, np.float32), np.float32)
    init("half", np.array(0.5, np.float32), np.float32)
    init("one", np.array(1.0, np.float32), np.float32)

    # ---- per-row / per-col occupancy of red(2) and cyan(8) via cheap Convs --
    # A [1,10,1,30] conv summing channel-2 across each row -> [1,1,30,1] (120B).
    Wrr = np.zeros((1, 10, 1, W), np.float32); Wrr[0, 2, 0, :] = 1.0
    init("Wrr", Wrr, np.float32); n("Conv", ["input", "Wrr"], "red_row")  # [1,1,30,1]
    Wrc = np.zeros((1, 10, H, 1), np.float32); Wrc[0, 2, :, 0] = 1.0
    init("Wrc", Wrc, np.float32); n("Conv", ["input", "Wrc"], "red_col")  # [1,1,1,30]
    Wcr = np.zeros((1, 10, 1, W), np.float32); Wcr[0, 8, 0, :] = 1.0
    init("Wcr", Wcr, np.float32); n("Conv", ["input", "Wcr"], "cy_row")
    Wcc = np.zeros((1, 10, H, 1), np.float32); Wcc[0, 8, :, 0] = 1.0
    init("Wcc", Wcc, np.float32); n("Conv", ["input", "Wcc"], "cy_col")

    # index ramps
    init("rampR", np.arange(H, dtype=np.float32).reshape(1, 1, H, 1), np.float32)
    init("rampC", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), np.float32)

    # helper to get min / max present index along an axis -> scalar
    def minmax(occ, ramp, axes, tag):
        # occ>0.5 ? ramp : BIG  -> ReduceMin ; occ>0.5 ? ramp : -1 -> ReduceMax
        n("Greater", [occ, "half"], tag + "_b")
        n("Where", [tag + "_b", ramp, "BIG"], tag + "_lo")
        n("ReduceMin", [tag + "_lo"], tag + "_min", axes=axes, keepdims=0)  # []
        n("Where", [tag + "_b", ramp, "nBIG"], tag + "_hi")
        n("ReduceMax", [tag + "_hi"], tag + "_max", axes=axes, keepdims=0)
        return tag + "_min", tag + "_max"

    rr_min, rr_max = minmax("red_row", "rampR", [0, 1, 2, 3], "redR")  # red rmin/rmax
    rc_min, rc_max = minmax("red_col", "rampC", [0, 1, 2, 3], "redC")
    cyr_min, cyr_max = minmax("cy_row", "rampR", [0, 1, 2, 3], "cyR")
    cyc_min, cyc_max = minmax("cy_col", "rampC", [0, 1, 2, 3], "cyC")

    # ---- decide axis: rows disjoint  <=>  vertical move --------------------
    # rows disjoint if red_rmax < cy_rmin  OR  cy_rmax < red_rmin
    n("Less", [rr_max, cyr_min], "redAbove")     # red above cyan (vertical)
    n("Less", [cyr_max, rr_min], "redBelow")     # red below cyan (vertical)
    n("Or", ["redAbove", "redBelow"], "vert")    # vertical move axis

    # dr (vertical):
    #   redAbove -> (cy_rmin-1) - red_rmax ;  redBelow -> (cy_rmax+1) - red_rmin
    n("Sub", [cyr_min, "one"], "vA1"); n("Sub", ["vA1", rr_max], "drA")
    n("Add", [cyr_max, "one"], "vB1"); n("Sub", ["vB1", rr_min], "drB")
    n("Where", ["redAbove", "drA", "drB"], "dr_v")   # vertical dr (if vert)
    # dc (horizontal):
    n("Less", [rc_max, cyc_min], "redLeft")
    n("Sub", [cyc_min, "one"], "hA1"); n("Sub", ["hA1", rc_max], "dcA")
    n("Add", [cyc_max, "one"], "hB1"); n("Sub", ["hB1", rc_min], "dcB")
    n("Where", ["redLeft", "dcA", "dcB"], "dc_h")    # horizontal dc (if horiz)

    n("Where", ["vert", "dr_v", "zero"], "dr")       # final dr scalar
    n("Where", ["vert", "zero", "dc_h"], "dc")       # final dc scalar

    # ---- colour index of input (one-hot -> 0..9) via 1x1 Conv (mem 0) -------
    kw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("kw", kw, np.float32)
    n("Conv", ["input", "kw"], "cidxf")              # [1,1,30,30] f32 colour idx
    n("Cast", ["cidxf"], "cidx", to=U8)              # uint8 900B

    # ---- shift block source: gather rows by (r - dr), cols by (c - dc) ------
    # source row index sr[r] = r - dr ; clamp to [0,29] (out-of-range filled bg
    # via the bbox mask below, value there is irrelevant).
    n("Sub", ["rampR", "dr"], "srf")                 # [1,1,30,1] f32 = r-dr
    n("Sub", ["rampC", "dc"], "scf")                 # [1,1,1,30] f32 = c-dc
    init("c0", np.array(0.0, np.float32), np.float32)
    init("c29", np.array(float(H - 1), np.float32), np.float32)
    n("Clip", ["srf", "c0", "c29"], "src_c")
    n("Clip", ["scf", "c0", "c29"], "scc")
    n("Squeeze", ["src_c"], "sr2", axes=[0, 1, 3])   # [30]
    n("Squeeze", ["scc"], "sc2", axes=[0, 1, 2])     # [30]
    n("Cast", ["sr2"], "sri", to=TensorProto.INT32)
    n("Cast", ["sc2"], "sci", to=TensorProto.INT32)
    n("Gather", ["cidx", "sri"], "g1", axis=2)       # shift rows
    n("Gather", ["g1", "sci"], "blkidx", axis=3)     # [1,1,30,30] shifted colour idx

    # ---- inBlock mask: source (r-dr, c-dc) lies in red bbox -----------------
    # red bbox = rr_min..rr_max  x  rc_min..rc_max ; test on (srf, scf).
    n("Less", ["srf", rr_min], "lt_r"); n("Not", ["lt_r"], "ge_r")
    n("Greater", ["srf", rr_max], "gt_r"); n("Not", ["gt_r"], "le_r")
    n("And", ["ge_r", "le_r"], "in_r")               # [1,1,30,1]
    n("Less", ["scf", rc_min], "lt_c"); n("Not", ["lt_c"], "ge_c")
    n("Greater", ["scf", rc_max], "gt_c"); n("Not", ["gt_c"], "le_c")
    n("And", ["ge_c", "le_c"], "in_c")               # [1,1,1,30]
    n("And", ["in_r", "in_c"], "inBlock")            # [1,1,30,30] bool

    # ---- cyan cell mask: cidx == 8 (uint8) ----------------------------------
    init("u8_8", np.array(8, np.uint8).reshape(1, 1, 1, 1), np.uint8)
    n("Equal", ["cidx", "u8_8"], "cymask")           # [1,1,30,30] bool

    # ---- assemble label map -------------------------------------------------
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v8", np.array(8, np.uint8), np.uint8)
    n("Where", ["cymask", "v8", "v0"], "Lbase")      # cyan over bg
    n("Where", ["inBlock", "blkidx", "Lbase"], "Lin")  # moved block on top

    # off-grid (all-zero input) -> sentinel 10 so output stays all-zero there
    n("ReduceMax", ["input"], "occ", axes=[1], keepdims=1)   # [1,1,30,30]
    n("Greater", ["occ", "half"], "ingrid")
    init("v10", np.array(10, np.uint8), np.uint8)
    n("Where", ["ingrid", "Lin", "v10"], "L")

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task008", [x], [y], inits)
    return helper.make_model(
        g, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
