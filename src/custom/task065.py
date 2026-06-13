"""Task 065 (2dc579da): fold the dotted quadrant onto an s x s grid.

Rule (from ARC-GEN): input is a (2s+1)x(2s+1) grid (s in 1..7) split into four
s x s quadrants by a central cross of `linecolor` (row s and col s). The whole
grid is background `b` except a single `dotcolor` pixel sitting in one quadrant.
Output is an s x s grid filled with `b` carrying that dot at the same in-quadrant
position. Three colors (line/dot/b) are always distinct, so:
  - dotcolor channel  = the unique color with cell-count 1
  - b channel         = the color with cell-count 4s^2-1 = tot - 2*sqrt(tot)
where tot = ReduceSum(input) = (2s+1)^2 and sqrt(tot)=2s+1 (exact in f32).

Graph (no canvas-sized intermediates except the final [1,2,30,30] feature map):
  s, s+1 derived from tot via Sqrt; per-color counts -> dot/b channel vectors;
  dot row/col profiles found via masked row/col reductions; each profile folded
  (add the shifted-by-(s+1) copy, then keep the first s entries) via a computed
  Gather index; region = rowmask (x) colmask, dot_pos = rowprof_f (x) colprof_f
  built jointly as a [1,2,30,30] map; a final 1x1 Conv with the runtime weight
  W[k,0]=bvec[k], W[k,1]=dotvec[k]-bvec[k] writes `output` directly (free):
    output[k] = region*bvec[k] + dot_pos*(dotvec[k]-bvec[k]).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    I32 = onnx.TensorProto.INT32
    F = onnx.TensorProto.FLOAT

    # ---- constants ----
    init("ARANGE", np.arange(30, dtype=np.int32), np.int32)     # [30]
    init("ARANGEF", np.arange(30, dtype=np.float32), np.float32)  # [30]
    init("c1i", np.array(1, np.int32), np.int32)
    init("c1f", np.array(1.0, np.float32), np.float32)
    init("c2f", np.array(2.0, np.float32), np.float32)
    init("chalf", np.array(0.5, np.float32), np.float32)
    init("Wshape", np.array([10, 1, 1, 1], np.int64), np.int64)

    # ---- size scalars ----
    n("ReduceSum", ["input"], "tot", keepdims=1)                # [1,1,1,1] (2s+1)^2
    n("Sqrt", ["tot"], "sq")                                    # 2s+1
    # s+1 = (sq+1)/2 ; s = (sq-1)/2
    n("Add", ["sq", "c1f"], "sqp1")
    n("Mul", ["sqp1", "chalf"], "sp1f")                         # s+1
    n("Sub", ["sq", "c1f"], "sqm1")
    n("Mul", ["sqm1", "chalf"], "sf")                           # s
    n("Cast", ["sf"], "s_4", to=I32)
    n("Squeeze", ["s_4"], "s", axes=[0, 1, 2, 3])               # scalar s int
    n("Squeeze", ["sp1f"], "sp1fs", axes=[0, 1, 2, 3])          # scalar s+1 float
    # b count = tot - 2*sq
    n("Mul", ["sq", "c2f"], "twosq")
    n("Sub", ["tot", "twosq"], "bcountf")                       # [1,1,1,1]
    n("Cast", ["bcountf"], "bcount", to=I32)                    # [1,1,1,1] int

    # ---- channel vectors: dotvec / bvec  [1,10,1,1] ----
    n("ReduceSum", ["input"], "cntf", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    n("Cast", ["cntf"], "cnt", to=I32)
    n("Equal", ["cnt", "c1i"], "dotvec_b")                      # count==1
    n("Equal", ["cnt", "bcount"], "bvec_b")                     # count==4s^2-1
    n("Cast", ["dotvec_b"], "dotvec", to=F)                     # [1,10,1,1]
    n("Cast", ["bvec_b"], "bvec", to=F)

    # ---- dot location mask -> row/col profiles ----
    # dotvec [1,10,1,1] doubles as a 1x1 Conv weight [out=1,in=10,1,1]:
    # dotmask[r,c] = 1 iff input cell (r,c) is the dot color
    n("Conv", ["input", "dotvec"], "dotmask")                   # [1,1,30,30]
    n("ReduceSum", ["dotmask"], "rowprof", axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceSum", ["dotmask"], "colprof", axes=[2], keepdims=1)  # [1,1,1,30]

    # ---- fold: profile + shifted-by-(s+1) copy, then keep first s ----
    # gidx = clip(arange + (s+1), max=29) (float, Clip supports float in op10)
    n("Add", ["ARANGEF", "sp1fs"], "gidx_rawf")                 # [30] float
    n("Clip", ["gidx_rawf"], "gidx_clipf", max=29.0)            # [30] float
    n("Cast", ["gidx_clipf"], "gidx", to=I32)                   # [30] int32
    n("Gather", ["rowprof", "gidx"], "row_sh", axis=2)          # [1,1,30,1]
    n("Gather", ["colprof", "gidx"], "col_sh", axis=3)          # [1,1,1,30]
    n("Add", ["rowprof", "row_sh"], "rowfold")                  # [1,1,30,1]
    n("Add", ["colprof", "col_sh"], "colfold")                  # [1,1,1,30]

    # masks (i < s) and apply to folded profiles
    n("Less", ["ARANGE", "s"], "mask_b")                        # [30] bool
    n("Cast", ["mask_b"], "mask1d", to=F)                       # [30]
    # reshape to row/col mask shapes
    init("rms", np.array([1, 1, 30, 1], np.int64), np.int64)
    init("cms", np.array([1, 1, 1, 30], np.int64), np.int64)
    n("Reshape", ["mask1d", "rms"], "rowmask")                  # [1,1,30,1]
    n("Reshape", ["mask1d", "cms"], "colmask")                  # [1,1,1,30]
    n("Mul", ["rowfold", "rowmask"], "rowprof_f")               # [1,1,30,1]
    n("Mul", ["colfold", "colmask"], "colprof_f")               # [1,1,1,30]

    # ---- joint [1,2,30,30] feature: ch0=region, ch1=dot_pos ----
    n("Concat", ["rowmask", "rowprof_f"], "RM", axis=1)         # [1,2,30,1]
    n("Concat", ["colmask", "colprof_f"], "CM", axis=1)         # [1,2,1,30]
    n("Mul", ["RM", "CM"], "feat")                              # [1,2,30,30]

    # ---- final Conv weight built from channel vectors ----
    n("Sub", ["dotvec", "bvec"], "dmb")                         # [1,10,1,1]
    n("Reshape", ["bvec", "Wshape"], "bvec_r")                  # [10,1,1,1]
    n("Reshape", ["dmb", "Wshape"], "dmb_r")                    # [10,1,1,1]
    n("Concat", ["bvec_r", "dmb_r"], "W", axis=1)               # [10,2,1,1]
    n("Conv", ["feat", "W"], "output")                          # [1,10,30,30] free

    return _model(nodes, inits)
