"""task298 (ARC-AGI bda2d7a6) — cyclic recolor of concentric square rings.

Rule (from the generator):
  A size x size grid (size = 2*half, half in {3,4}) is 4-fold symmetric (mirror in
  both axes).  Cell (r,c) belongs to ring m = min(r, c) (distance from the nearest
  outer edge of the top-left quadrant, mirrored into all four quadrants).  Three
  DISTINCT colors c0,c1,c2 = colors[0],colors[1],colors[2] (sample of 3 from 0..9,
  black allowed) are assigned by ring:
      input  ring m :  colors[m % 3]
      output ring m :  colors[(m + 2) % 3]
  So at every cell, output_color = the color one step BACK in the 3-cycle:
      input c0 -> output c2 ,  input c1 -> output c0 ,  input c2 -> output c1.
  This is a pure per-instance color permutation (position-independent).  The whole
  grid is filled (no padding inside the grid); off-grid cells (rows/cols >= size)
  are all-zero one-hot and must stay all-zero.

  The ring order is recoverable spatially: the outer/corner ring is c0 (cell (0,0)),
  the next ring c1 (cell (1,1)), the inner ring c2 (cell (2,2)) -- positions that
  always exist since size >= 6.

Encoding (route the 10-ch expansion into the FREE Where output):
  colf  = sum_k k*input_k   (1x1 conv)               [1,1,30,30] fp32  (entry plane)
  ingrid= ReduceMax(input over channel) > 0          [1,1,30,30]       (1 in-grid incl
          black, 0 off-grid -- a black cell sets ch0=1, off-grid sets nothing)
  c0s,c1s,c2s = colf at (0,0),(1,1),(2,2)            scalars (color indices)
  mask_cX = And(Equal(colf, cXs), ingrid)            [1,1,30,30] bool
  cX_onehot = input[:, :, p:p+1, p:p+1]              [1,10,1,1]  (one-hot of color cX)
  output = Where(mask_c0, c2_onehot,
              Where(mask_c1, c0_onehot,
                Where(mask_c2, c1_onehot, ZERO)))     FREE [1,10,30,30]
  The Where broadcasts [1,1,30,30] cond and [1,10,1,1] value into the free output,
  so the only full-canvas intermediates are colf and the three bool masks.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    W = 8  # active canvas: grid size is 2*half in {6,8}, so 8x8 covers all in-grid

    # ---- slice the input to the 8x8 active region [1,10,8,8] ---------------
    init("in_s", np.array([0, 0, 0], np.int64), np.int64)
    init("in_e", np.array([10, W, W], np.int64), np.int64)
    init("in_ax", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "in_s", "in_e", "in_ax"], "inp")  # [1,10,8,8] fp32

    # ---- colf = sum_k k * input_k  (1x1 conv) -> [1,1,8,8] fp32 -> fp16 ----
    w = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("colw", w, np.float32)
    n("Conv", ["inp", "colw"], "colf32")  # [1,1,8,8] fp32
    n("Cast", ["colf32"], "colf", to=F16)  # [1,1,8,8] fp16

    # ---- ingrid = ReduceMax over channel axis > 0 -> bool ------------------
    n("ReduceMax", ["inp"], "occ", axes=[1], keepdims=1)  # [1,1,8,8] f32
    init("ZF", np.array(0.0, np.float32), np.float32)
    n("Greater", ["occ", "ZF"], "ingrid")  # [1,1,8,8] bool

    # ---- color-index scalars at ring positions (0,0),(1,1),(2,2) -----------
    def scalar_at(p, name):
        init(f"{name}_s", np.array([0, p, p], np.int64), np.int64)
        init(f"{name}_e", np.array([1, p + 1, p + 1], np.int64), np.int64)
        init(f"{name}_ax", np.array([1, 2, 3], np.int64), np.int64)
        return n("Slice", ["colf", f"{name}_s", f"{name}_e", f"{name}_ax"], name)

    c0s = scalar_at(0, "c0s")  # [1,1,1,1] fp16
    c1s = scalar_at(1, "c1s")
    c2s = scalar_at(2, "c2s")

    # ---- build a single OUTPUT color-index plane Lout8, then expand -------
    # in-grid: output index = c2 where input==c0, c0 where input==c1,
    #          c1 where input==c2 (disjoint partition, exactly one hit).
    # off-grid: Lout = -1 (matches no channel -> all-false output).
    # Lout8 = m0*c2s + m1*c0s + m2*c1s - (1 - ingrid_f)
    # where mX = Cast(Equal(colf,cXs), fp16) gated by ingrid; off-grid colf=0
    # so a black color (cXs==0) could falsely match off-grid -- gate by ingrid.
    n("Cast", ["ingrid"], "ingrid_f", to=F16)  # [1,1,8,8] fp16 {0,1}

    def maskf(cs, name):
        eq = n("Equal", ["colf", cs], f"{name}_eq")          # bool [1,1,8,8]
        ef = n("Cast", [eq], f"{name}_ef", to=F16)           # fp16 {0,1}
        return n("Mul", [ef, "ingrid_f"], name)              # fp16, gated in-grid

    m0 = maskf(c0s, "m0")  # cells of color c0  -> emit c2 index
    m1 = maskf(c1s, "m1")  # cells of color c1  -> emit c0 index
    m2 = maskf(c2s, "m2")  # cells of color c2  -> emit c1 index

    n("Mul", [m0, c2s], "t0")  # [1,1,8,8] fp16
    n("Mul", [m1, c0s], "t1")
    n("Mul", [m2, c1s], "t2")

    # off-grid penalty: -(1 - ingrid) so off-grid Lout = -1
    init("ONE_H", np.array(1.0, np.float16), np.float16)
    n("Sub", ["ONE_H", "ingrid_f"], "offg")   # 1 off-grid, 0 in-grid
    # Lout = t0 + t1 + t2 - offg
    n("Sum", ["t0", "t1", "t2"], "tsum")       # [1,1,8,8] fp16
    n("Sub", ["tsum", "offg"], "Lout8")        # fp16, in {-1, 0..9}

    # ---- one-hot expand on the small 8x8 region, then pad to 30x30 --------
    # Equal(Lout8, arange) -> [1,10,8,8] bool (off-grid Lout=-1 -> all-false).
    init("arange16", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)
    n("Equal", ["Lout8", "arange16"], "oh8")          # [1,10,8,8] bool
    U8 = TensorProto.UINT8
    n("Cast", ["oh8"], "oh8u", to=U8)                 # [1,10,8,8] uint8
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("ZU8", np.array(0, np.uint8), np.uint8)
    n("Pad", ["oh8u", "pads", "ZU8"], "output", mode="constant")  # FREE [1,10,30,30] uint8

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task298", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
