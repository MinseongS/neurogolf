"""task388 (ARC-AGI f5b8619d) — "2x2 tile a sparse grid + cyan vertical lines through pixel-columns".

Rule (from the generator):
  Input is a size x size grid (size in 2..6) with 1..size colored pixels of a single
  colour `color` (cyan excluded), on a 0 background.
  Output is a 2*size x 2*size grid:
    - For every input column c that contains >=1 colored pixel, the WHOLE vertical
      lines at output columns c and c+size are painted cyan(8) (all 2*size rows).
    - Each input pixel (r,c)=color is tiled into the 2x2 layout
        out[r][c]=out[r][c+size]=out[r+size][c]=out[r+size][c+size]=color,
      painted AFTER the cyan so the colour overwrites cyan.

  Per output cell (r,c), r,c in [0,2*size):
    out = color   if src[r mod size][c mod size] is a colored pixel
        = cyan(8)  else if column (c mod size) has any colored pixel
        = 0        else.

Encoding (Tier B-ish — small active canvas, double-MatMul 2x2 tiling, separable cyan):
  Work on W=12 active canvas (max output = 12x12).  NO 30x30 colour plane is ever
  materialised; the colour-collapse is replaced by a 1-channel bg slice + count vector:
    * src (colored-pixel occupancy on W x W) = (bg-channel==0) AND (r<size) AND (c<size).
      bg channel is sliced to [1,1,W,W] (576 B) — off-grid cells also have bg==0, hence
      the in-grid AND-gate.
    * k (colour value) = ReduceMax over channels of (cnt>0)*channel_ramp, where
      cnt = ReduceSum(input, axes=[2,3]) [1,10,1,1] (40 B) — ch0 ramp weight is 0 so bg
      never wins; only the single colour channel is nonzero -> k.
    * size = #occupied input rows = ReduceSum over rows of (rowany>0), where
      rowany = ReduceMax(input, axes=[1,3]) [1,1,30,1] (120 B), no occupancy plane.
  Tiling matrices (W x W, fp16, {0,1} exact):
    R[Rout,rin]  = (Rout==rin) OR (Rout==rin+size)
    C^T[cin,Cout]= (Cout==cin) OR (Cout==cin+size)
  tile  = R @ src @ C^T                       -> 2x2 tiled colour mask
  cyancol[1,1,1,W] = ReduceMax(src,rows) @ C^T  -> cyan output columns
  ingrid = (row<2*size) AND (col<2*size)
  Label L (uint8, W x W): k where tile, else 8 where cyancol & ingrid, else 99 off-grid.
  Pad to 30x30 with 99, output = Equal(L, arange[0..9]) -> BOOL.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

W = 12   # output active canvas (max output is 2*6 = 12)
IW = 6   # input active canvas (max input is 6x6)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("ZEROF", np.array(0.0, np.float32), np.float32)
    init("ZEROH", np.array(0.0, np.float16), np.float16)

    # ---- bg channel sliced to the IW x IW input canvas ---------------------
    init("bg_s", np.array([0, 0, 0], np.int64), np.int64)
    init("bg_e", np.array([1, IW, IW], np.int64), np.int64)
    init("bg_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "bg_s", "bg_e", "bg_ax"], "bg")   # [1,1,IW,IW] f32

    # ---- k = colour value (no plane) ---------------------------------------
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    n("Greater", ["cnt", "ZEROF"], "cntb")                     # bool [1,10,1,1]
    n("Cast", ["cntb"], "cntf", to=F32)
    init("chramp", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Mul", ["cntf", "chramp"], "kvec")                       # [1,10,1,1] (ch0->0)
    n("ReduceMax", ["kvec"], "kf", axes=[1], keepdims=1)       # [1,1,1,1] f32
    n("Cast", ["kf"], "ku8", to=U8)                            # scalar uint8

    # ---- size = #occupied input rows (no plane) ----------------------------
    n("ReduceMax", ["input"], "rowany", axes=[1, 3], keepdims=1)  # [1,1,30,1] f32
    n("Greater", ["rowany", "ZEROF"], "rowanyb")                  # bool [1,1,30,1]
    n("Cast", ["rowanyb"], "rowanyf", to=F32)
    n("ReduceSum", ["rowanyf"], "sizef", axes=[2], keepdims=1)    # [1,1,1,1] f32

    # ---- src = colored-pixel occupancy on IW x IW --------------------------
    # colored = (bg-channel == 0) AND in input-grid (r<size, c<size)
    axI2 = init("axI2", np.arange(IW, dtype=np.float32).reshape(1, 1, IW, 1), np.float32)
    axI3 = init("axI3", np.arange(IW, dtype=np.float32).reshape(1, 1, 1, IW), np.float32)
    axO2 = init("axO2", np.arange(W, dtype=np.float32).reshape(1, 1, W, 1), np.float32)
    axO3 = init("axO3", np.arange(W, dtype=np.float32).reshape(1, 1, 1, W), np.float32)
    n("Equal", ["bg", "ZEROF"], "notbg")               # bool [1,1,IW,IW]
    n("Less", ["axI2", "sizef"], "rin_in")             # bool [1,1,IW,1]  r<size
    n("Less", ["axI3", "sizef"], "cin_in")             # bool [1,1,1,IW]  c<size
    n("And", ["rin_in", "cin_in"], "ingrid_in")        # bool [1,1,IW,IW]
    n("And", ["notbg", "ingrid_in"], "srcb")           # bool [1,1,IW,IW]
    n("Cast", ["srcb"], "src", to=F16)                 # fp16 [1,1,IW,IW]

    # ---- tiling matrices R [Rout(W),rin(IW)], C^T [cin(IW),Cout(W)] --------
    # R[Rout(axis2), rin(axis3)] = (Rout==rin) OR (Rout==rin+size)
    n("Equal", ["axO2", "axI3"], "Reye")               # bool [1,1,W,IW]
    n("Add", ["axI3", "sizef"], "rin_sh")              # rin+size  [1,1,1,IW]
    n("Equal", ["axO2", "rin_sh"], "Rsh")              # bool [1,1,W,IW]
    n("Or", ["Reye", "Rsh"], "Rb")
    n("Cast", ["Rb"], "R", to=F16)                     # fp16 [1,1,W,IW]

    # C^T[cin(axis2), Cout(axis3)] = (Cout==cin) OR (Cout==cin+size)
    n("Equal", ["axO3", "axI2"], "Ceye")               # bool [1,1,IW,W]
    n("Add", ["axI2", "sizef"], "cin_sh")              # cin+size  [1,1,IW,1]
    n("Equal", ["axO3", "cin_sh"], "Csh")              # bool [1,1,IW,W]
    n("Or", ["Ceye", "Csh"], "CTb")
    n("Cast", ["CTb"], "CT", to=F16)                   # fp16 [1,1,IW,W]

    # ---- tile = R @ src @ C^T  (fp16, {0,1} exact) -------------------------
    n("MatMul", ["R", "src"], "rowtile")               # fp16 [1,1,W,IW]
    n("MatMul", ["rowtile", "CT"], "tilef")            # fp16 [1,1,W,W]
    n("Greater", ["tilef", "ZEROH"], "tileb")          # bool [1,1,W,W]

    # ---- cyancol = pixcol @ C^T --------------------------------------------
    n("ReduceMax", ["src"], "pixcol", axes=[2], keepdims=1)  # fp16 [1,1,1,IW]
    n("MatMul", ["pixcol", "CT"], "cyancolf")          # fp16 [1,1,1,W]
    n("Greater", ["cyancolf", "ZEROH"], "cyancolb")    # bool [1,1,1,W]

    # ---- ingrid output = (row<2*size) AND (col<2*size) ---------------------
    n("Add", ["sizef", "sizef"], "two_size")           # [1,1,1,1]
    n("Less", ["axO2", "two_size"], "rin_g")           # bool [1,1,W,1]
    n("Less", ["axO3", "two_size"], "cin_g")           # bool [1,1,1,W]
    n("And", ["rin_g", "cin_g"], "ingrid")             # bool [1,1,W,W]
    n("And", ["cyancolb", "ingrid"], "cyanb")          # bool [1,1,W,W]

    # ---- label map ---------------------------------------------------------
    init("V0", np.array(0, np.uint8), np.uint8)
    init("V8", np.array(8, np.uint8), np.uint8)
    init("V99", np.array(99, np.uint8), np.uint8)
    n("Where", ["cyanb", "V8", "V0"], "L1")            # cyan on bg canvas
    n("Where", ["tileb", "ku8", "L1"], "Lin")          # colour wins
    n("Where", ["ingrid", "Lin", "V99"], "Lw")         # off-output -> sentinel

    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    n("Pad", ["Lw", "pads", "V99"], "L", mode="constant")  # [1,1,30,30] uint8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task388", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
