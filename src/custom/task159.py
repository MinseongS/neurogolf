"""Task 159 (6b9890af): magnify a 3x3 sprite into a red-bordered box.

Rule (from ARC-GEN): the input contains a 3x3 sprite (single non-red color)
and a hollow red square outline of side outsize = 3*m+2 (m = magnifier 1..4).
The output is a fresh outsize x outsize grid: a red border ring, and inside it
the sprite magnified by m (each sprite cell -> an m x m block), placed at
offset +1 from the border. Interior non-sprite cells are background (color 0).
Everything outside the outsize x outsize rectangle is all-zero.

Recovery from input (1-D reductions, cheap):
  - m = (#red cells - 4) / 12       (red ring perimeter is 12m+4)
  - footprint = sum of channels except 0 (bg) and 2 (red)
  - (rmin,cmin) = top-left of the sprite footprint bbox
  - sprite colour = the single channel (not 0/2) that is non-empty

Memory floor-break (small 14x14 canvas + uint8 label map + final Equal):
  outsize = 3m+2 <= 14, so the whole output fits the top-left 14x14 corner.
  The single canvas-sized float is the footprint Conv plane fpf [1,1,30,30].
  The EXACT 3x3 sprite S is gathered straight from fpf (3 rows then 3 cols at
  the bbox top-left) -> a [3,3] bool, never a 14x14 footprint copy.  Magnify is
  two tiny Gathers: gidx[i]=clip(floor((i-1)/m),0,2) over the 14-vector, then
  out[i,j]=S[gidx[i],gidx[j]] via Gather(S,axis=0)+Gather(.,axis=1) -> [14,14]
  bool (no int64 index plane, no fp32 14x14 plane).  Region geometry stays 1-D
  (in-grid/border edge tests on [1,1,14,1]/[1,1,1,14]) and only the masks
  broadcast to 14x14.  A uint8 label L (priority offgrid>border>sprite>bg) is
  padded to 30x30 (sentinel 10) and finished with Equal(L, arange[0..9]) into
  the free BOOL `output`.

  Memory ~7.7KB: dominated by the irreducible fpf Conv plane (3600B, the one
  fp32 per-cell reduction of the 10-ch input) + the 30x30 uint8 label (900B,
  output-shaping floor) + the [1,1,3,30] row-gather (360B); everything else is
  the small 14x14 working set + 1-D recovery vectors.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 14  # outsize = 3*m+2 <= 14 for m <= 4


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
    I64 = TensorProto.INT64

    # ---- per-channel presence (sum over H,W) ----
    n("ReduceSum", ["input"], "chsum", axes=[2, 3], keepdims=1)   # [1,10,1,1]
    init("g2", np.array([2], np.int64), np.int64)
    n("Gather", ["chsum", "g2"], "redc", axis=1)                  # [1,1,1,1]
    init("c4", np.array(4.0, np.float32), np.float32)
    init("c12", np.array(12.0, np.float32), np.float32)
    n("Sub", ["redc", "c4"], "rm4")
    n("Div", ["rm4", "c12"], "m")                                 # scalar m

    # ---- footprint plane (sum channels != 0,2) via 1x1 Conv ----
    keepW = np.ones((1, 10, 1, 1), np.float32)
    keepW[0, 0] = 0.0
    keepW[0, 2] = 0.0
    init("keepW", keepW, np.float32)
    init("keep", keepW.reshape(1, 10, 1, 1), np.float32)
    n("Conv", ["input", "keepW"], "fpf")                          # [1,1,30,30] f32

    # ---- sprite colour one-hot vector -> scalar colour id (uint8) ----
    init("zeroS", np.array(0.0, np.float32), np.float32)
    n("Greater", ["chsum", "zeroS"], "present")                   # [1,10,1,1] b
    n("Cast", ["present"], "presentf", to=F)
    n("Mul", ["presentf", "keep"], "colvec")                      # [1,10,1,1]
    arc = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("arc", arc, np.float32)
    n("Mul", ["colvec", "arc"], "colid_parts")
    n("ReduceSum", ["colid_parts"], "colid_f", axes=[1, 2, 3], keepdims=1)
    n("Cast", ["colid_f"], "colid_u8", to=TensorProto.UINT8)      # scalar colour

    # ---- rmin, cmin from footprint (1-D occupancy, Where idiom) ----
    n("ReduceMax", ["fpf"], "rowmax", axes=[3], keepdims=1)       # [1,1,30,1]
    n("ReduceMax", ["fpf"], "colmax", axes=[2], keepdims=1)       # [1,1,1,30]
    n("Greater", ["rowmax", "zeroS"], "rocc")                     # bool
    n("Greater", ["colmax", "zeroS"], "cocc")
    init("Rrow", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1), np.float32)
    init("Ccol", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30), np.float32)
    init("c99", np.array(99.0, np.float32), np.float32)
    init("c1", np.array(1.0, np.float32), np.float32)
    n("Where", ["rocc", "Rrow", "c99"], "Rbig")                  # [1,1,30,1]
    n("Where", ["cocc", "Ccol", "c99"], "Cbig")                  # [1,1,1,30]
    n("ReduceMin", ["Rbig"], "rmin", axes=[2], keepdims=1)        # [1,1,1,1]
    n("ReduceMin", ["Cbig"], "cmin", axes=[3], keepdims=1)        # [1,1,1,1]

    # ---- gather the exact 3x3 sprite (3 rows then 3 cols) from fpf ----
    # rows rmin+[0,1,2], cols cmin+[0,1,2]; sprite always covers all 3 rows/cols.
    init("step3", np.array([0.0, 1.0, 2.0], np.float32), np.float32)
    init("sh1", np.array([1], np.int64), np.int64)
    n("Reshape", ["rmin", "sh1"], "rmin1")       # [1]
    n("Reshape", ["cmin", "sh1"], "cmin1")
    n("Add", ["rmin1", "step3"], "ridx_f")       # [3]
    n("Add", ["cmin1", "step3"], "cidx_f")
    init("c0f", np.array(0.0, np.float32), np.float32)
    init("c29f", np.array(29.0, np.float32), np.float32)
    n("Clip", ["ridx_f", "c0f", "c29f"], "ridx_c")
    n("Clip", ["cidx_f", "c0f", "c29f"], "cidx_c")
    n("Cast", ["ridx_c"], "ridx", to=I64)        # [3]
    n("Cast", ["cidx_c"], "cidx", to=I64)
    n("Gather", ["fpf", "ridx"], "occr", axis=2)   # [1,1,3,30]
    n("Gather", ["occr", "cidx"], "S", axis=3)     # [1,1,3,3]
    init("s33", np.array([3, 3], np.int64), np.int64)
    n("Reshape", ["S", "s33"], "S33f")             # [3,3] fp32
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["S33f", "half"], "S33")          # [3,3] bool

    # ---- magnify on a 14x14 canvas via two small Gathers ----
    # out[i,j] = S33[ri[i], cj[j]] where ri[i]=clip(floor((i-1)/m),0,2).
    # The interior/border masks below overwrite cells outside the sprite block,
    # so out-of-block index values are harmless.
    arW = np.arange(WORK, dtype=np.float32).reshape(WORK)   # [WORK]
    init("arW", arW, np.float32)
    n("Sub", ["arW", "c1"], "ar0")               # [WORK]  (i-1)
    n("Div", ["ar0", "m"], "ard")
    n("Floor", ["ard"], "arfl")
    init("c2f", np.array(2.0, np.float32), np.float32)
    n("Clip", ["arfl", "c0f", "c2f"], "arcl")    # broadcast [1,1,1,WORK] in 0..2
    init("shW", np.array([WORK], np.int64), np.int64)
    n("Reshape", ["arcl", "shW"], "arcl1")       # [WORK]
    n("Cast", ["arcl1"], "gidx", to=I64)         # [WORK] int64
    n("Gather", ["S33", "gidx"], "Srow", axis=0)  # [WORK,3] bool
    n("Gather", ["Srow", "gidx"], "magB", axis=1)  # [WORK,WORK] bool
    # (kept rank-2; broadcasts against [1,1,WORK,WORK] masks below)

    # ---- region geometry on 14x14 (index comparisons, mostly 1-D) ----
    # All comparisons stay 1-D ([1,1,WORK,1] / [1,1,1,WORK]); only the final
    # masks broadcast to WORKxWORK.  outsize G = 3m+2; in-grid <=> idx<=G-1;
    # border <=> idx==0 or idx==G-1 (within grid).
    arWr = np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1)
    arWc = np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK)
    init("arWr", arWr, np.float32)
    init("arWc", arWc, np.float32)
    init("c3", np.array(3.0, np.float32), np.float32)
    init("c2", np.array(2.0, np.float32), np.float32)
    n("Mul", ["m", "c3"], "m3")                  # 3m
    n("Add", ["m3", "c2"], "G")                  # outsize = 3m+2
    n("Sub", ["G", "half"], "Gh")                # G-0.5  (in-grid <=> idx<Gh)
    n("Sub", ["G", "c1"], "Gm1")                 # G-1 = last ring index
    n("Sub", ["Gm1", "half"], "Gm1h")            # G-1.5 (last <=> idx>Gm1h)
    # in-grid (separable)
    n("Less", ["arWr", "Gh"], "RinB"); n("Less", ["arWc", "Gh"], "CinB")
    n("And", ["RinB", "CinB"], "inGridB")        # [1,1,WORK,WORK]
    # border edges (1-D): idx==0 (idx<0.5) or idx==G-1 (idx>G-1.5)
    n("Less", ["arWr", "half"], "Rzero"); n("Greater", ["arWr", "Gm1h"], "Rlast")
    n("Or", ["Rzero", "Rlast"], "RbB")           # [1,1,WORK,1]
    n("Less", ["arWc", "half"], "Czero"); n("Greater", ["arWc", "Gm1h"], "Clast")
    n("Or", ["Czero", "Clast"], "CbB")           # [1,1,1,WORK]
    n("Or", ["RbB", "CbB"], "borderB")           # [1,1,WORK,WORK] ring within box

    # magnified sprite mask restricted to in-grid (border applied AFTER, so it
    # overrides any sprite-on cell that falls on the ring -> no interior mask).
    n("And", ["magB", "inGridB"], "spriteB")     # [1,1,WORK,WORK]

    # ---- uint8 label map on 14x14 (priority high->low: offgrid>border>sprite>bg)
    init("v0", np.array(0, np.uint8), np.uint8)
    init("v2", np.array(2, np.uint8), np.uint8)
    init("v10", np.array(10, np.uint8), np.uint8)
    n("Where", ["spriteB", "colid_u8", "v0"], "L0")   # sprite colour over bg(0)
    n("Where", ["borderB", "v2", "L0"], "L1")         # red ring over sprite/bg
    n("Where", ["inGridB", "L1", "v10"], "L14")       # off-grid sentinel 10 wins

    # ---- pad to 30x30 (sentinel 10), final Equal ----
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L14", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30]
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")              # free BOOL

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task159", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
