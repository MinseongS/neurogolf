"""task177 (ARC-AGI 7468f01a) — crop the rectangle and mirror it horizontally.

Rule (from the generator):
  A solid `colors[0]` rectangle (tall x wide, both 4..8) sits at (rowoffset,
  coloffset) on a 0 background.  Inside it, a small connected creature is drawn in
  `colors[1]`.  Everything outside the rectangle is background 0, and the rectangle
  fully covers its own bounding box, so the bbox of all non-zero pixels == the
  rectangle.  The output is that rectangle CROPPED to the top-left of a fresh grid
  and FLIPPED LEFT-RIGHT (mirror columns):
      output[r][c] = input[min_row + r][min_col + (W-1-c)]   for r<H, c<W
  where (min_row,min_col,H,W) is the non-zero bounding box.  Output size = HxW.
  Colours are random per instance, so the per-cell colour value must be carried.

Encoding (Tier B-ish — crop+flip window + label map + final Equal):
  1. colplane[1,1,30,30] fp16 = sum_k k*input_k  (1x1 Conv, per-cell colour 0..9).
  2. bbox from 1-D any-nonzero occupancy profiles: min_row,min_col (first occupied),
     H,W (occupied count).
  3. row idx = min_row + arange(WORK); col idx (FLIPPED) = min_col + (W-1) - arange(WORK).
     Gather rows then cols -> small WORK x WORK colour window (already mirrored).
  4. label L[1,1,WORK,WORK] uint8 = colour where (r<H and c<W) else sentinel 10.
     Pad to 30x30 (sentinel 10) so off-grid cells are all-channels-False.
  5. output = Equal(L, arange[0..9]) -> free BOOL one-hot output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL

N = 30
WORK = 8  # max output dimension (wide,tall <= 8)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- per-cell colour index plane (1x1 Conv: sum_k k*input_k) -------------
    # Conv must match input dtype (fp32, FREE); casting input to fp16 would
    # materialize an 18000B fp16 copy of the 10-channel input, so keep the Conv
    # in fp32 (3600B) and gather on that plane directly, casting only the tiny
    # WORK x WORK window to uint8 at the end.
    w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("convw", w, np.float32)
    n("Conv", ["input", "convw"], "colf")          # [1,1,30,30] f32, values 0..9

    # ---- 1-D non-background occupancy profiles ------------------------------
    # Every background (colour 0) cell has channel-0 == 1, so reducing over ALL
    # channels would mark every cell occupied.  Non-background = any of channels
    # 1..9 present.  colf (= sum_k k*input_k) is >0 exactly on non-background
    # cells (colour 0 -> 0), so reuse it as the occupancy signal.
    init("ZEROF", np.array(0.0, np.float32), np.float32)
    n("ReduceMax", ["colf"], "rowprof_all", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["colf"], "colprof_all", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    n("Greater", ["rowprof_all", "ZEROF"], "rowb")   # bool [1,1,30,1]
    n("Greater", ["colprof_all", "ZEROF"], "colb")   # bool [1,1,1,30]

    # H = number of occupied rows ; W = number of occupied cols
    n("Cast", ["rowb"], "rowbf", to=F32)
    n("Cast", ["colb"], "colbf", to=F32)
    n("ReduceSum", ["rowbf"], "Hf", axes=[2], keepdims=1)   # [1,1,1,1] f32
    n("ReduceSum", ["colbf"], "Wf", axes=[3], keepdims=1)   # [1,1,1,1] f32

    # min_row / min_col = first occupied index
    init("BIG", np.array(1000.0, np.float32), np.float32)
    rr = np.arange(N, dtype=np.float32).reshape(1, 1, N, 1)
    cc = np.arange(N, dtype=np.float32).reshape(1, 1, 1, N)
    init("rr", rr, np.float32)
    init("cc", cc, np.float32)
    n("Where", ["rowb", "rr", "BIG"], "rsrc")
    n("ReduceMin", ["rsrc"], "minr", axes=[2], keepdims=1)   # [1,1,1,1]
    n("Where", ["colb", "cc", "BIG"], "csrc")
    n("ReduceMin", ["csrc"], "minc", axes=[3], keepdims=1)   # [1,1,1,1]

    # ---- build gather indices ------------------------------------------------
    init("baseW", np.arange(WORK, dtype=np.float32), np.float32)   # [WORK]
    init("shp1", np.array([1], np.int64), np.int64)
    init("ONEF", np.array(1.0, np.float32), np.float32)
    init("c0", np.array(0.0, np.float32), np.float32)
    init("c29", np.array(29.0, np.float32), np.float32)

    # row indices = min_row + arange(WORK)
    n("Reshape", ["minr", "shp1"], "minr1")          # [1]
    n("Add", ["baseW", "minr1"], "ridx_f")           # [WORK]
    n("Clip", ["ridx_f", "c0", "c29"], "ridx_cl")
    n("Cast", ["ridx_cl"], "ridx", to=I64)           # [WORK] int64

    # col indices (FLIPPED) = min_col + (W-1) - arange(WORK)
    n("Reshape", ["minc", "shp1"], "minc1")          # [1]
    n("Reshape", ["Wf", "shp1"], "W1")               # [1]
    n("Sub", ["W1", "ONEF"], "Wm1")                  # W-1
    n("Add", ["minc1", "Wm1"], "cbase")              # min_col + W - 1
    n("Sub", ["cbase", "baseW"], "cidx_f")           # cbase - arange(WORK)
    n("Clip", ["cidx_f", "c0", "c29"], "cidx_cl")
    n("Cast", ["cidx_cl"], "cidx", to=I64)           # [WORK] int64

    # ---- gather the mirrored WORK x WORK colour window -----------------------
    n("Gather", ["colf", "ridx"], "Vr", axis=2)      # [1,1,WORK,30] f32
    n("Gather", ["Vr", "cidx"], "Vw", axis=3)        # [1,1,WORK,WORK] f32
    n("Cast", ["Vw"], "Vu8", to=U8)                  # [1,1,WORK,WORK] uint8

    # ---- box mask (r < H) and (c < W) on WORK x WORK canvas ------------------
    rampw_r = np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1)
    rampw_c = np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK)
    init("wr", rampw_r, np.float32)
    init("wc", rampw_c, np.float32)
    n("Less", ["wr", "Hf"], "rmask")                 # [1,1,WORK,1] bool
    n("Less", ["wc", "Wf"], "cmask")                 # [1,1,1,WORK] bool
    n("And", ["rmask", "cmask"], "boxmask")          # [1,1,WORK,WORK] bool

    # ---- label map: colour inside box, sentinel 10 outside -------------------
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["boxmask", "Vu8", "u10"], "Lw")      # [1,1,WORK,WORK] uint8
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, N - WORK, N - WORK], np.int64), np.int64)
    n("Pad", ["Lw", "padpads", "u10"], "L", mode="constant")  # [1,1,30,30] uint8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")              # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, N, N])
    g = helper.make_graph(nodes, "task177", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
