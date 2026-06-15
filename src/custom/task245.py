"""task245 (ARC-AGI a1570a43) — "shift the red sprite back into the green box".

Rule (from the generator):
  A 7x7 green "box" is marked only by its four green(3) CORNERS at
  (brow,bcol), (brow,bcol+6), (brow+6,bcol), (brow+6,bcol+6).  A red(2) conway
  sprite lives inside the box (interior rows brow+1..brow+5, cols bcol+1..bcol+5)
  in the OUTPUT.  In the INPUT the same red sprite is translated OUT of the box by
  a uniform offset: either UP by k in 1..4 (roff=-k, coff=0) OR LEFT by k in 1..4
  (roff=0, coff=-k).  The green corners are IDENTICAL in input and output.
  Transform = shift the red pixels by (dr,dc) (DOWN/RIGHT, exactly one nonzero)
  to bring them back inside the box; green stays fixed.  Only colours {0,2,3}.
  Grid is at most 10x10 (width,height in 7..10).

Detection (verified byte-exact, 5000 fresh):
  brow = min row containing green, bcol = min col containing green
  rmin = min row containing red,   cmin = min col containing red
  dr = max(0, brow+1 - rmin),  dc = max(0, bcol+1 - cmin)   (one is 0)
  output_red = red shifted DOWN by dr and RIGHT by dc.

Encoding (Tier B — variable-offset gather + label-map + final Equal):
  Work on a 10x10 top-left canvas (the true active region).  The translation
  offset is input-derived, so this is a Tier-B gather (not S, not separable A).
  Shift red via Gather(axis=2, arange-dr) then Gather(axis=3, arange-dc), masked
  by validity.  Build uint8 label L[1,1,10,10] = 2*red_out + 3*green, sentinel 10
  off-grid; Pad to 30x30; output = Equal(L, arange[0..9]) -> BOOL.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
B = TensorProto.BOOL
I64 = TensorProto.INT64

W = 10  # working canvas (grid is at most 10x10)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- channel slices on the 10x10 canvas --------------------------------
    # red = channel 2, green = channel 3, bg = channel 0.
    def chan_slice(ch, name):
        init(f"{name}_s", np.array([ch, 0, 0], np.int64), np.int64)
        init(f"{name}_e", np.array([ch + 1, W, W], np.int64), np.int64)
        init(f"{name}_ax", np.array([1, 2, 3], np.int64), np.int64)
        n("Slice", ["input", f"{name}_s", f"{name}_e", f"{name}_ax"], name)
        return name  # [1,1,W,W] f32

    chan_slice(0, "bg")     # [1,1,W,W] f32
    chan_slice(2, "red")    # [1,1,W,W] f32
    chan_slice(3, "green")  # [1,1,W,W] f32

    init("ZEROF", np.array(0.0, np.float32), np.float32)
    init("BIG", np.array(1e6, np.float32), np.float32)
    init("ONEF", np.array(1.0, np.float32), np.float32)
    arange = init("arange", np.arange(W, dtype=np.float32).reshape(1, 1, W, 1),
                  np.float32)              # [1,1,W,1] row index ramp

    # ---- min row / min col of green and of red (scalars) -------------------
    # presence-per-row = ReduceMax over col axis; presence-per-col = over row.
    def min_index(plane, axis_keep, tag):
        # axis_keep=2 -> reduce over axis 3 (cols) giving per-row presence [1,1,W,1]
        # axis_keep=3 -> reduce over axis 2 (rows) giving per-col presence [1,1,1,W]
        red_axes = [3] if axis_keep == 2 else [2]
        n("ReduceMax", [plane], f"pres_{tag}", axes=red_axes, keepdims=1)
        n("Greater", [f"pres_{tag}", "ZEROF"], f"presb_{tag}")  # bool
        # ramp along the kept axis
        ramp = (np.arange(W, dtype=np.float32).reshape(1, 1, W, 1) if axis_keep == 2
                else np.arange(W, dtype=np.float32).reshape(1, 1, 1, W))
        init(f"ramp_{tag}", ramp, np.float32)
        n("Where", [f"presb_{tag}", f"ramp_{tag}", "BIG"], f"idx_{tag}")
        n("ReduceMin", [f"idx_{tag}"], f"min_{tag}", axes=[2, 3], keepdims=1)
        return f"min_{tag}"  # [1,1,1,1] f32 scalar

    min_index("green", 2, "grow")   # brow
    min_index("green", 3, "gcol")   # bcol
    min_index("red", 2, "rrow")     # rmin
    min_index("red", 3, "rcol")     # cmin

    # dr = max(0, brow+1 - rmin) ; dc = max(0, bcol+1 - cmin)
    n("Add", ["min_grow", "ONEF"], "brow1")
    n("Sub", ["brow1", "min_rrow"], "dr_raw")
    n("Max", ["dr_raw", "ZEROF"], "dr")          # [1,1,1,1] f32
    n("Add", ["min_gcol", "ONEF"], "bcol1")
    n("Sub", ["bcol1", "min_rcol"], "dc_raw")
    n("Max", ["dc_raw", "ZEROF"], "dc")          # [1,1,1,1] f32

    # ---- shift red DOWN by dr (axis=2 gather) and RIGHT by dc (axis=3) ------
    # gathered[r] = red[r-dr]; valid only where r-dr in [0,W-1].
    # row source index = clip(arange_row - dr, 0, W-1)
    arange_row = init("arange_row", np.arange(W, dtype=np.float32).reshape(1, 1, W, 1),
                      np.float32)
    arange_col = init("arange_col", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W),
                      np.float32)
    init("WM1", np.array(W - 1, np.float32), np.float32)

    # cast red to uint8 (1 byte) so the two gathers are 100B planes, not 400B.
    init("shapeW", np.array([W], np.int64), np.int64)
    n("Greater", ["red", "ZEROF"], "redorigb")             # [1,1,W,W] bool
    n("Cast", ["redorigb"], "red_u8", to=U8)               # [1,1,W,W] uint8

    n("Sub", ["arange_row", "dr"], "rsrc_raw")   # [1,1,W,1]
    n("Clip", ["rsrc_raw", "ZEROF", "WM1"], "rsrc_c")
    n("Cast", ["rsrc_c"], "rsrc_i", to=I64)
    n("Reshape", ["rsrc_i", "shapeW"], "rsrc_flat")  # [W]
    n("Gather", ["red_u8", "rsrc_flat"], "red_r", axis=2)   # [1,1,W,W] uint8

    n("Sub", ["arange_col", "dc"], "csrc_raw")   # [1,1,1,W]
    n("Clip", ["csrc_raw", "ZEROF", "WM1"], "csrc_c")
    n("Cast", ["csrc_c"], "csrc_i", to=I64)
    n("Reshape", ["csrc_i", "shapeW"], "csrc_flat")  # [W]
    n("Gather", ["red_r", "csrc_flat"], "red_rc", axis=3)  # [1,1,W,W] uint8

    # validity: row r valid iff arange_row - dr >= 0 ; col c valid iff >=0.
    # (opset 11 has no GreaterOrEqual -> use Not(Less(.,0)))
    n("Less", ["rsrc_raw", "ZEROF"], "rinvalid")
    n("Not", ["rinvalid"], "rvalid")                       # [1,1,W,1] bool
    n("Less", ["csrc_raw", "ZEROF"], "cinvalid")
    n("Not", ["cinvalid"], "cvalid")                       # [1,1,1,W] bool
    # combine in BOOL: redb = gathered>0 & rvalid & cvalid.
    n("Cast", ["red_rc"], "red_rc_b", to=B)                 # [1,1,W,W] bool
    n("And", ["red_rc_b", "rvalid"], "red_rb")             # broadcast rows
    n("And", ["red_rb", "cvalid"], "redb")                 # [1,1,W,W] bool

    # ---- in-grid mask: bg OR red OR green (only colours {0,2,3}) ------------
    n("Greater", ["bg", "ZEROF"], "bgb")
    n("Greater", ["green", "ZEROF"], "greenb")
    n("Greater", ["red", "ZEROF"], "redorigb")
    n("Or", ["bgb", "greenb"], "ing1")
    n("Or", ["ing1", "redorigb"], "ingrid_b")              # [1,1,W,W] bool
    init("V0", np.array(0, np.uint8), np.uint8)
    init("V2", np.array(2, np.uint8), np.uint8)
    init("V3", np.array(3, np.uint8), np.uint8)
    init("V10", np.array(10, np.uint8), np.uint8)
    n("Where", ["ingrid_b", "V0", "V10"], "Lbase")     # 0 in-grid, 10 off-grid
    n("Where", ["greenb", "V3", "Lbase"], "L1")        # paint green
    n("Where", ["redb", "V2", "L1"], "Lw")             # paint red (disjoint)

    # pad 10x10 -> 30x30 with sentinel 10
    init("padval", np.array(10, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    n("Pad", ["Lw", "pads", "padval"], "L", mode="constant")  # [1,1,30,30] uint8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task245", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
