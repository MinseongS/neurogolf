"""task242 (ARC-AGI 9ecd008a) — recover a blacked-out 3x3 hole from D2 symmetry.

Rule (from the generator, verified fresh):
  The input is a 2*size x 2*size grid (size=8 -> 16x16) built with full D2
  symmetry: a symmetric `bitmap` is replicated into all four quadrants via
  horizontal AND vertical reflections, so
      grid[r][c] == grid[2*size-1-r][c]      (vertical  mirror)
                 == grid[r][2*size-1-c]       (horizontal mirror).
  Every cell is a colour in 1..9 (random_color never returns 0).  A contiguous
  minisize x minisize (=3x3) block at (row, col) is then BLACKED OUT (set to 0).
  The output is the ORIGINAL 3x3 values that were blacked out, placed at the
  top-left corner of the output grid.

  Recovery: the hole is the (only) all-zero 3x3 block inside the 16x16 grid.
  Its values equal a mirror of the same positions.  The vertical mirror block
  may OVERLAP the hole when the hole straddles the row centre (between rows 7,8);
  likewise the horizontal mirror when it straddles the col centre.  A hole never
  straddles both centres such that BOTH mirrors read black at the same cell, so
      output[dr,dc] = max( vflip(grid)[row+dr,col+dc],
                            hflip(grid)[row+dr,col+dc] )
  recovers every cell (verified exact 0/3000 fresh).  vflip maps r->15-r,
  hflip maps c->15-c within the 16x16 grid.

Pipeline (ONNX, opset 11):
  1. colf = 1x1 Conv sum_k k*input_k  -> [1,1,30,30] fp32 value plane.
  2. Slice to the 16x16 grid (g16); vflip rows (negative-step Slice) -> vf16.
  3. holemask = (g16 == 0) on the 16x16 grid (the only zeros are the hole).
  4. r0 = min hole row, c0 = min hole col via ramp ReduceMin over a Where.
  5. Gather a 3x3 window of vf16 at (r0, c0): row Gather(axis=2) then col
     Gather(axis=3) -> [1,1,3,3] recovered colours.
  6. Label map L (uint8) on the 3x3, Pad to 30x30 with sentinel 10, then
     Equal(L, arange[0..9]) into the free BOOL output (sentinel never matches
     0..9 so cells outside the 3x3 are all-channels-off, as required).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL

GRID = 16   # 2*size
MINI = 3    # minisize


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    BIG = 1000.0

    # ---- 1. colour-index plane (the one allowed fp32 entry) ------------------
    kw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("kw", kw, np.float32)                                # [1,10,1,1]
    n("Conv", ["input", "kw"], "colf")                        # [1,1,30,30] fp32

    # cast to fp16 for cheap downstream full-grid ops
    n("Cast", ["colf"], "colf16", to=F16)                     # [1,1,30,30] fp16

    # ---- 2. crop to the 16x16 grid, and vertical flip ------------------------
    init("s0", np.array([0], np.int64), np.int64)
    init("sG", np.array([GRID], np.int64), np.int64)
    init("ax2", np.array([2], np.int64), np.int64)
    init("ax3", np.array([3], np.int64), np.int64)
    n("Slice", ["colf16", "s0", "sG", "ax2"], "g16r")         # rows 0..15
    n("Slice", ["g16r", "s0", "sG", "ax3"], "g16")            # [1,1,16,16] fp16

    # vflip rows / hflip cols: start=15, end=before-0, step=-1
    init("vstart", np.array([GRID - 1], np.int64), np.int64)
    init("vend", np.array([-GRID - 1], np.int64), np.int64)
    init("vstep", np.array([-1], np.int64), np.int64)
    n("Slice", ["g16", "vstart", "vend", "ax2", "vstep"], "vf16")  # vflip rows
    n("Slice", ["g16", "vstart", "vend", "ax3", "vstep"], "hf16")  # hflip cols

    # ---- 3. hole mask = (g16 == 0) -------------------------------------------
    init("z16", np.array(0.0, np.float16), np.float16)
    n("Equal", ["g16", "z16"], "holeb")                       # bool [1,1,16,16]
    n("Cast", ["holeb"], "holef", to=F16)                     # fp16 0/1

    # row-has / col-has hole
    n("ReduceMax", ["holef"], "rowhas", axes=[3], keepdims=1)  # [1,1,16,1] fp16
    n("ReduceMax", ["holef"], "colhas", axes=[2], keepdims=1)  # [1,1,1,16] fp16

    init("half", np.array(0.5, np.float16), np.float16)
    n("Greater", ["rowhas", "half"], "rowb")                  # bool [1,1,16,1]
    n("Greater", ["colhas", "half"], "colb")                  # bool [1,1,1,16]

    ramp_r = np.arange(GRID, dtype=np.float16).reshape(1, 1, GRID, 1)
    ramp_c = np.arange(GRID, dtype=np.float16).reshape(1, 1, 1, GRID)
    init("rr", ramp_r, np.float16)
    init("rc", ramp_c, np.float16)
    init("PBIG", np.array(BIG, np.float16), np.float16)

    n("Where", ["rowb", "rr", "PBIG"], "rsrc")                # [1,1,16,1] fp16
    n("ReduceMin", ["rsrc"], "r0_16", axes=[2], keepdims=1)   # [1,1,1,1] fp16
    n("Where", ["colb", "rc", "PBIG"], "csrc")                # [1,1,1,16] fp16
    n("ReduceMin", ["csrc"], "c0_16", axes=[3], keepdims=1)   # [1,1,1,1] fp16

    # ---- 5. Gather a 3x3 window of vf16 at (r0, c0) --------------------------
    init("shp1", np.array([1], np.int64), np.int64)
    n("Reshape", ["r0_16", "shp1"], "r0_1")                   # [1] fp16
    n("Reshape", ["c0_16", "shp1"], "c0_1")                   # [1] fp16
    n("Cast", ["r0_1"], "r0f", to=F32)
    n("Cast", ["c0_1"], "c0f", to=F32)

    base = np.arange(MINI, dtype=np.float32)
    init("base", base, np.float32)                            # [3]
    n("Add", ["base", "r0f"], "ridx_f")                       # [3] fp32
    n("Add", ["base", "c0f"], "cidx_f")
    n("Cast", ["ridx_f"], "ridx", to=I64)                     # [3] int64
    n("Cast", ["cidx_f"], "cidx", to=I64)

    n("Gather", ["vf16", "ridx"], "Vvr", axis=2)              # [1,1,3,16] fp16
    n("Gather", ["Vvr", "cidx"], "Vv", axis=3)                # [1,1,3,3] fp16
    n("Gather", ["hf16", "ridx"], "Vhr", axis=2)              # [1,1,3,16] fp16
    n("Gather", ["Vhr", "cidx"], "Vh", axis=3)                # [1,1,3,3] fp16

    # combine the two mirrors (fp32 Max on the tiny 3x3 window) ----------------
    n("Cast", ["Vv"], "Vvf", to=F32)
    n("Cast", ["Vh"], "Vhf", to=F32)
    n("Max", ["Vvf", "Vhf"], "Vs")                            # [1,1,3,3] fp32

    # ---- 6. label map -> Pad sentinel 10 -> Equal one-hot --------------------
    n("Cast", ["Vs"], "L33", to=U8)                           # [1,1,3,3] uint8
    init("u10", np.array(10, np.uint8), np.uint8)
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 30 - MINI, 30 - MINI], np.int64), np.int64)
    n("Pad", ["L33", "padpads", "u10"], "L", mode="constant")  # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                       # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task242", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
