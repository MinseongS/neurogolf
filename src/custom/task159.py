"""Task 159 (6b9890af): magnify a 3x3 sprite into a red-bordered box.

Rule (from ARC-GEN): the input contains a 3x3 sprite (single non-red color)
and a hollow red square outline of side outsize = 3*m+2 (m = magnifier 1..4).
The output is a fresh outsize x outsize grid: a red border ring, and inside it
the sprite magnified by m (each sprite cell -> an m x m block), placed at
offset +1 from the border. Interior non-sprite cells are background (color 0).
Everything outside the outsize x outsize rectangle is all-zero.

Recovery from input:
  - m = (#red cells - 4) / 12   (red box is a hollow ring, perimeter 12m+4)
  - sprite footprint = sum of channels except 0 (bg) and 2 (red)
  - (rmin,cmin) = top-left of the sprite footprint bbox (3x3)
  - sprite color = the single channel (not 0/2) that is non-empty

Construction (all integer math, exact in float32):
  Row expand/select matrix P[Ro,R] = [R == rmin + floor((Ro-1)/m)]; col Q
  likewise. magInterior = (P @ footprint @ Q^T) masked to the interior box
  [1..3m]^2. border ring + interior-bg fill channel 0, sprite color channel
  gets magInterior, red channel gets the ring. One Conv-free fully arithmetic
  graph writing straight into `output`.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    I32 = onnx.TensorProto.INT32
    F = onnx.TensorProto.FLOAT
    H = onnx.TensorProto.FLOAT16

    # ---- per-channel presence (sum over H,W) ----
    n("ReduceSum", ["input"], "chsum", axes=[2, 3], keepdims=1)   # [1,10,1,1] f
    # red count -> m = (redcount - 4)/12
    init("g2", np.array([2], np.int64), np.int64)
    n("Gather", ["chsum", "g2"], "redc", axis=1)                  # [1,1,1,1] f
    init("c4", np.array(4.0, np.float32))
    init("c12", np.array(12.0, np.float32))
    n("Sub", ["redc", "c4"], "rm4")
    n("Div", ["rm4", "c12"], "m")                                 # [1,1,1,1] f  scalar m

    # ---- sprite footprint plane via 1x1 Conv (sum channels != 0,2) ----
    keep = np.ones((1, 10, 1, 1), np.float32)
    keep[0, 0] = 0.0
    keep[0, 2] = 0.0
    init("keep", keep)
    keepW = keep.reshape(1, 10, 1, 1)
    init("keepW", keepW)
    n("Conv", ["input", "keepW"], "fpf")                          # [1,1,30,30] f32
    n("Cast", ["fpf"], "fp", to=H)                                # [1,1,30,30] fp16 footprint
    # (fpf reused for occupancy reductions below to keep them f32-exact)

    # ---- sprite color one-hot vector colvec[1,10,1,1] ----
    # channel present (chsum>0) AND kept
    n("Greater", ["chsum", "zeroS"], "present")                   # [1,10,1,1] b
    init("zeroS", np.array(0.0, np.float32))
    n("Cast", ["present"], "presentf", to=F)
    n("Mul", ["presentf", "keep"], "colvec")                      # [1,10,1,1] f

    # ---- rmin, cmin from footprint ----
    # row occupancy
    n("ReduceSum", ["fpf"], "rowsum", axes=[3], keepdims=1)       # [1,1,30,1] f
    n("ReduceSum", ["fpf"], "colsum", axes=[2], keepdims=1)       # [1,1,1,30] f
    n("Greater", ["rowsum", "zeroS"], "rocc")                     # [1,1,30,1] b
    n("Greater", ["colsum", "zeroS"], "cocc")                     # [1,1,1,30] b
    n("Cast", ["rocc"], "roccf", to=F)
    n("Cast", ["cocc"], "coccf", to=F)
    # bigR = R*occ + (1-occ)*99  -> min gives first occupied row index
    init("Rrow", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1))
    init("Ccol", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30))
    init("c99", np.array(99.0, np.float32))
    init("c1", np.array(1.0, np.float32))
    # rows
    n("Mul", ["Rrow", "roccf"], "Rocc")
    n("Sub", ["c1", "roccf"], "rinv")
    n("Mul", ["rinv", "c99"], "rinv99")
    n("Add", ["Rocc", "rinv99"], "Rbig")                          # [1,1,30,1]
    n("ReduceMin", ["Rbig"], "rmin", axes=[2], keepdims=1)        # [1,1,1,1]
    # cols
    n("Mul", ["Ccol", "coccf"], "Cocc")
    n("Sub", ["c1", "coccf"], "cinv")
    n("Mul", ["cinv", "c99"], "cinv99")
    n("Add", ["Cocc", "cinv99"], "Cbig")
    n("ReduceMin", ["Cbig"], "cmin", axes=[3], keepdims=1)        # [1,1,1,1]

    # ---- floor((idx-1)/m) target row/col index vectors ----
    # fdR[Ro] = floor((Ro-1)/m); tgtR[Ro] = rmin + fdR
    n("Sub", ["Rrow", "c1"], "Rm1")                               # [1,1,30,1]
    n("Div", ["Rm1", "m"], "Rdiv")
    n("Floor", ["Rdiv"], "Rfd")
    n("Add", ["Rfd", "rmin"], "tgtR")                             # [1,1,30,1] f
    n("Sub", ["Ccol", "c1"], "Cm1")
    n("Div", ["Cm1", "m"], "Cdiv")
    n("Floor", ["Cdiv"], "Cfd")
    n("Add", ["Cfd", "cmin"], "tgtC")                             # [1,1,1,30] f

    # ---- P[Ro,R] = [R == tgtR[Ro]] as [30,30]; Q[Co,C] = [C == tgtC] ----
    # build via broadcast Equal of an index row vector against tgtR column.
    # Ridx as [1,1,1,30] (the R axis), tgtR as [1,1,30,1] -> [1,1,30,30]
    n("Cast", ["tgtR"], "tgtRi", to=I32)                          # [1,1,30,1]
    n("Cast", ["Ccol"], "Ridxi", to=I32)                         # [1,1,1,30] reuse 0..29
    n("Equal", ["Ridxi", "tgtRi"], "Pb")                         # [1,1,30,30] b
    n("Cast", ["Pb"], "P", to=H)                                 # [1,1,30,30] fp16
    n("Cast", ["tgtC"], "tgtCi", to=I32)                          # [1,1,1,30]
    n("Cast", ["Rrow"], "Cidxi", to=I32)                         # [1,1,30,1] 0..29
    n("Equal", ["Cidxi", "tgtCi"], "Qb")                         # [1,1,30,30] b  Q[C,Co]
    n("Cast", ["Qb"], "Qf", to=H)                                # [1,1,30,30] fp16

    # ---- mag = P @ fp @ Q ----
    # P:[1,1,30(Ro),30(R)] @ fp:[1,1,30(R),30(C)] = [1,1,30(Ro),30(C)]
    n("MatMul", ["P", "fp"], "PF")                               # [1,1,30,30]
    # PF:[1,1,30(Ro),30(C)] @ Qf:[1,1,30(C),30(Co)] = [1,1,30(Ro),30(Co)]
    n("MatMul", ["PF", "Qf"], "mag")                             # [1,1,30,30]

    # ---- region masks from m ----
    # G = 3m+2 ; interior: 1<=R<=3m i.e. R>=1 and R<=3m -> R-1>=0 and R<=3m
    init("c3", np.array(3.0, np.float32))
    init("c2", np.array(2.0, np.float32))
    n("Mul", ["m", "c3"], "m3")                                  # 3m
    n("Add", ["m3", "c2"], "G")                                  # 3m+2
    # interiorR[Ro] = (Ro>=1) and (Ro<=3m)
    init("half", np.array(0.5, np.float32))
    n("Greater", ["Rrow", "half"], "Rge1")                       # Ro>=1  [1,1,30,1] b
    n("Add", ["m3", "half"], "m3h")
    n("Less", ["Rrow", "m3h"], "Rle3m")                          # Ro<=3m
    n("And", ["Rge1", "Rle3m"], "RintB")                         # [1,1,30,1] b
    n("Greater", ["Ccol", "half"], "Cge1")
    n("Less", ["Ccol", "m3h"], "Cle3m")
    n("And", ["Cge1", "Cle3m"], "CintB")                         # [1,1,1,30] b
    n("And", ["RintB", "CintB"], "interiorB")                    # [1,1,30,30] b
    n("Cast", ["interiorB"], "interior", to=F)                   # [1,1,30,30] f

    # magInterior = mag * interior  (cast mag fp16 -> f32 first)
    n("Cast", ["mag"], "magf", to=F)                             # [1,1,30,30] f
    n("Mul", ["magf", "interior"], "magI")                       # [1,1,30,30] f

    # ---- border ring ----
    # inGrid: Ro<G and Co<G ; borderR: Ro==0 or Ro==G-1
    n("Sub", ["G", "half"], "Gh")                                # G-0.5
    n("Less", ["Rrow", "Gh"], "RinB")                            # Ro<=G-1
    n("Less", ["Ccol", "Gh"], "CinB")
    n("And", ["RinB", "CinB"], "inGridB")                        # [1,1,30,30] b
    # Ro==0 : Ro<0.5 ; Ro==G-1 : Ro>G-1.5
    n("Less", ["Rrow", "half"], "Rzero")                         # Ro==0
    n("Sub", ["G", "c1"], "Gm1")
    n("Sub", ["Gm1", "half"], "Gm1h")
    n("Greater", ["Rrow", "Gm1h"], "Rlast")                      # Ro==G-1 (within grid)
    n("Or", ["Rzero", "Rlast"], "RborderB")                      # [1,1,30,1] b
    n("Less", ["Ccol", "half"], "Czero")
    n("Greater", ["Ccol", "Gm1h"], "Clast")
    n("Or", ["Czero", "Clast"], "CborderB")                      # [1,1,1,30] b
    n("Or", ["RborderB", "CborderB"], "anyborderB")              # [1,1,30,30] b
    n("And", ["anyborderB", "inGridB"], "borderB")               # [1,1,30,30] b
    n("Cast", ["borderB"], "border", to=F)                       # [1,1,30,30] f

    # ---- assemble output via a single 1x1 Conv into `output` (free) ----
    # stack3 = [magI, interior, border]; interior bg(ch0) = interior - magI
    # is computed inside the conv (W[0,interior]=1, W[0,magI]=-1).
    n("Concat", ["magI", "interior", "border"], "stack3", axis=1)  # [1,3,30,30]
    # runtime weight W3 [10,3,1,1]:
    #   slot0 (magI)     -> sprite color channel (W3[c,0]=colvec[c]); ch0-=magI
    #   slot1 (interior) -> channel 0            (W3[0,1]=1)
    #   slot2 (border)   -> channel 2            (W3[2,2]=1)
    init("shp_c", np.array([10, 1, 1, 1], np.int64), np.int64)
    n("Reshape", ["colvec", "shp_c"], "wcol")                    # [10,1,1,1]
    wfix = np.zeros((10, 2, 1, 1), np.float32)
    wfix[0, 0] = 1.0   # interior -> ch0
    wfix[2, 1] = 1.0   # border -> ch2
    init("wfix", wfix)
    n("Concat", ["wcol", "wfix"], "W3", axis=1)                  # [10,3,1,1]
    # subtract magI from ch0: W3[0,0] -= 1
    sub0 = np.zeros((10, 3, 1, 1), np.float32)
    sub0[0, 0] = -1.0
    init("sub0", sub0)
    n("Add", ["W3", "sub0"], "W3f")                              # [10,3,1,1]
    n("Conv", ["stack3", "W3f"], "output")                       # [1,10,30,30] free

    return _model(nodes, inits)
