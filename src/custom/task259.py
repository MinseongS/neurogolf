"""task259 (ARC-AGI a740d043) — crop the sprite's bounding box, blue background -> black.

Rule (from the generator):
  Input is a `width x height` (5..7 each) grid filled with BLUE (colour 1).  A small
  sprite (a `hollow_conway` blob, wide,tall in 2..3) is drawn on it at (rowoffset,
  coloffset); the sprite pixels take random non-blue colours (>=2).  The output is the
  sprite cropped to its bounding box (size wide x tall, <=3x3): every sprite pixel keeps
  its colour; every other cell inside the bbox (blue background / hollow gaps) becomes
  BLACK (colour 0).  Colours are random per instance so the per-cell value must be carried.

Encoding (small-canvas crop window + label map + final Equal):
  Background here is BLUE = colour 1 (NOT 0), so non-background == colf >= 2.
  The grid lives entirely in rows/cols 0..6, so slice the 10-ch input to [1,10,7,7]
  FIRST (escape (3): bounded active region) — the colour-index Conv then costs 196B
  instead of 3600B.
  1. colf[1,1,7,7] fp32 = sum_k k*input_k  (1x1 Conv, per-cell colour 0..9).
  2. occupancy = colf >= 2 (sprite).  bbox: min_row,min_col (first occupied),
     H=tall,W=wide (occupied counts) from 1-D occupancy profiles.
  3. value plane vmap = where(colf>=2, colf, 0)  -> blue/gap -> black(0).
  4. gather rows min_row+[0..WORK), then cols min_col+[0..WORK) -> WORK x WORK window.
  5. label L uint8 = window-colour where (r<H and c<W) else sentinel 10; Pad to 30x30
     (sentinel 10) so off-grid cells are all-channels-False.  Equal(L,arange(10)) -> BOOL.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL

N = 30
ACT = 7    # active grid is at most 7x7 (width,height in 5..7) at top-left
WORK = 3   # max output dim (wide,tall <= 3)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- slice the 10-ch input to the active 7x7 top-left region ------------
    init("st", np.array([0, 0], np.int64), np.int64)
    init("en", np.array([ACT, ACT], np.int64), np.int64)
    init("ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "st", "en", "ax"], "xin")        # [1,10,7,7] f32

    # ---- per-cell colour index plane (1x1 Conv: sum_k k*input_k) ------------
    w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("convw", w, np.float32)
    n("Conv", ["xin", "convw"], "colf")                   # [1,1,7,7] f32, 0..9

    # ---- occupancy (sprite = colour >= 2) -----------------------------------
    init("ONEHALF", np.array(1.5, np.float32), np.float32)
    n("ReduceMax", ["colf"], "rowmax", axes=[1, 3], keepdims=1)  # [1,1,7,1]
    n("ReduceMax", ["colf"], "colmax", axes=[1, 2], keepdims=1)  # [1,1,1,7]
    n("Greater", ["rowmax", "ONEHALF"], "rowb")           # bool [1,1,7,1]
    n("Greater", ["colmax", "ONEHALF"], "colb")           # bool [1,1,1,7]

    # bbox is the SPAN of occupied rows/cols (the sprite has hollow gaps so the
    # bbox is max-min+1, NOT the count of occupied lines):
    #   H = max_row - min_row + 1 ; W = max_col - min_col + 1
    init("BIG", np.array(1000.0, np.float32), np.float32)
    init("NEG", np.array(-1.0, np.float32), np.float32)
    init("ONEF", np.array(1.0, np.float32), np.float32)
    rr = np.arange(ACT, dtype=np.float32).reshape(1, 1, ACT, 1)
    cc = np.arange(ACT, dtype=np.float32).reshape(1, 1, 1, ACT)
    init("rr", rr, np.float32)
    init("cc", cc, np.float32)
    n("Where", ["rowb", "rr", "BIG"], "rsrc")
    n("ReduceMin", ["rsrc"], "minr", axes=[2], keepdims=1)   # [1,1,1,1]
    n("Where", ["rowb", "rr", "NEG"], "rsrcx")
    n("ReduceMax", ["rsrcx"], "maxr", axes=[2], keepdims=1)  # [1,1,1,1]
    n("Where", ["colb", "cc", "BIG"], "csrc")
    n("ReduceMin", ["csrc"], "minc", axes=[3], keepdims=1)   # [1,1,1,1]
    n("Where", ["colb", "cc", "NEG"], "csrcx")
    n("ReduceMax", ["csrcx"], "maxc", axes=[3], keepdims=1)  # [1,1,1,1]
    n("Sub", ["maxr", "minr"], "hd")
    n("Add", ["hd", "ONEF"], "Hf")                           # [1,1,1,1]
    n("Sub", ["maxc", "minc"], "wd")
    n("Add", ["wd", "ONEF"], "Wf")                           # [1,1,1,1]

    init("ZEROF", np.array(0.0, np.float32), np.float32)

    # ---- gather indices: rows min_row+[0..WORK), cols min_col+[0..WORK) -----
    init("baseW", np.arange(WORK, dtype=np.float32), np.float32)   # [WORK]
    init("shp1", np.array([1], np.int64), np.int64)
    init("c0", np.array(0.0, np.float32), np.float32)
    init("clo", np.array(float(ACT - 1), np.float32), np.float32)

    n("Reshape", ["minr", "shp1"], "minr1")               # [1]
    n("Add", ["baseW", "minr1"], "ridx_f")                # [WORK]
    n("Clip", ["ridx_f", "c0", "clo"], "ridx_cl")
    n("Cast", ["ridx_cl"], "ridx", to=I64)                # [WORK] int64

    n("Reshape", ["minc", "shp1"], "minc1")               # [1]
    n("Add", ["baseW", "minc1"], "cidx_f")                # [WORK]
    n("Clip", ["cidx_f", "c0", "clo"], "cidx_cl")
    n("Cast", ["cidx_cl"], "cidx", to=I64)                # [WORK] int64

    # ---- gather the WORK x WORK colour window, then blue/gap (<2) -> 0 ------
    n("Gather", ["colf", "ridx"], "Vr", axis=2)           # [1,1,WORK,7] f32
    n("Gather", ["Vr", "cidx"], "Vw", axis=3)             # [1,1,WORK,WORK] f32
    n("Greater", ["Vw", "ONEHALF"], "Vsprite")            # bool sprite mask
    n("Where", ["Vsprite", "Vw", "ZEROF"], "Vz")          # blue/gap -> 0
    n("Cast", ["Vz"], "Vu8", to=U8)                       # uint8

    # ---- box mask (r < H) and (c < W) on WORK x WORK ------------------------
    rampr = np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1)
    rampc = np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK)
    init("wr", rampr, np.float32)
    init("wc", rampc, np.float32)
    n("Less", ["wr", "Hf"], "rmask")                      # [1,1,WORK,1] bool
    n("Less", ["wc", "Wf"], "cmask")                      # [1,1,1,WORK] bool
    n("And", ["rmask", "cmask"], "boxmask")               # [1,1,WORK,WORK] bool

    # ---- label map: colour inside box, sentinel 10 outside ------------------
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["boxmask", "Vu8", "u10"], "Lw")           # [1,1,WORK,WORK] uint8
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, N - WORK, N - WORK], np.int64), np.int64)
    n("Pad", ["Lw", "padpads", "u10"], "L", mode="constant")  # [1,1,30,30] uint8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                   # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, N, N])
    g = helper.make_graph(nodes, "task259", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
