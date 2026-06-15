"""Task 019 (10fcaaa3): 2x2 tile with cyan diagonal halo.

Rule (from ARC-GEN generator): input is an H x W grid (H,W in 2..6) holding a
few cells of a single non-cyan color.  Output is 2H x 2W: the input grid is
tiled 2x2 (color stamps at (r,c),(r+H,c),(r,c+W),(r+H,c+W)); cyan(8) is painted
on the 4 diagonal neighbours of every stamped colour cell (in full output
coords, clipped to the 2H x 2W region), but the colour overwrites cyan.

Memory floor-break.  Two ideas stack:

1.  Label map + final Equal.  Instead of materialising a [1,10,30,30] one-hot
    stack, build a SINGLE-channel label map L giving the output colour at each
    cell (10 = outside the 2H x 2W region, 0 = in-grid background, 8 = cyan,
    k = the colour).  The FINAL op is Equal(L, arange[0..9]) writing straight
    into the free BOOL `output` (opset 11), so there is never a 10-channel
    intermediate.  Outside cells use sentinel 10 (>= 10 never matches channels
    0..9 -> all-false there, as required).  L is built with 3 Where ops, the
    minimum for a 4-valued label.

2.  Work on a 12 x 12 canvas.  The whole 2H x 2W output (H,W <= 6) fits inside
    the top-left 12 x 12 corner, so every per-cell intermediate is sliced to
    12 x 12 (144 elements) instead of 30 x 30 (900).  Only the very first colour
    Conv must emit 30 x 30 (its input is the fixed 30 x 30 grid); it is
    immediately Sliced to 12 x 12.  The label L is Padded back to 30 x 30 (with
    the sentinel 10) just before the final Equal.

The tile maps simply wrap (R % D) on the 12-wide canvas; rows/cols at or beyond
2H/2W are repaired by ANDing the colour stamp mask with the in-grid rectangle Gb
(built separably from the tiled 1-D occupancy), so wraparound never leaks a
spurious stamp.  All values (0/1 masks, 0..4 conv sums, labels 0..10) are small
integers, exact in fp16 / uint8.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

WORK = 12  # tiled working canvas side (2*max(H,W) = 12)
SRC = 6    # input grid side bound (max H,W); colour mask only needs SRC x SRC


def _idx_tables():
    """[5,WORK] int32 tile maps for D = H or W = 2..6: idx[D-2][R] = R % D.

    Wraparound is safe because cells with R >= 2D lie outside the 2H x 2W
    rectangle and are masked out by the in-grid rectangle Gb downstream."""
    return np.array(
        [[R % D for R in range(WORK)] for D in range(2, 7)], np.int32)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants ----
    init("IDX", _idx_tables(), np.int32)                          # [5,WORK]
    init("Wk", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1),
         np.float32)                                              # k accumulator
    init("diag", np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                          np.float16).reshape(1, 1, 3, 3), np.float16)
    init("two", np.array(2.0, np.float32), np.float32)
    init("half", np.array(0.5, np.float32), np.float32)
    init("ar_row", np.arange(WORK, dtype=np.float32).reshape(1, 1, WORK, 1),
         np.float32)                                             # row indices
    init("ar_col", np.arange(WORK, dtype=np.float32).reshape(1, 1, 1, WORK),
         np.float32)                                             # col indices
    init("half16", np.array(0.5, np.float16), np.float16)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    # Slice the background channel of the top-left SRC x SRC corner (axes
    # [1,2,3], [0,0,0] -> [1,SRC,SRC]).  The tile Gathers only ever read rows
    # 0..H-1 / cols 0..W-1 (H,W <= SRC), so the colour mask needs only SRC x SRC
    # = 6x6 (36B) rather than the full grid.
    init("bg_st", np.array([0, 0, 0], np.int64), np.int64)
    init("bg_en", np.array([1, SRC, SRC], np.int64), np.int64)
    init("bg_ax", np.array([1, 2, 3], np.int64), np.int64)

    # ---- recover H, W directly from 1-D occupancy (no `exists` canvas) ----
    n("ReduceMax", ["input"], "rowocc", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    n("ReduceMax", ["input"], "colocc", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    n("ReduceSum", ["rowocc"], "Hf", keepdims=0)                  # scalar H
    n("ReduceSum", ["colocc"], "Wf", keepdims=0)                  # scalar W
    n("Sub", ["Hf", "two"], "Hm")
    n("Sub", ["Wf", "two"], "Wm")
    n("Cast", ["Hm"], "Hi0", to=TensorProto.INT32)               # H-2 in 0..4
    n("Cast", ["Wm"], "Wi0", to=TensorProto.INT32)
    n("Squeeze", ["Hi0"], "Hi", axes=[0])                        # -> scalar index
    n("Squeeze", ["Wi0"], "Wi", axes=[0])
    n("Gather", ["IDX", "Hi"], "idxr", axis=0)                   # [WORK] int32
    n("Gather", ["IDX", "Wi"], "idxc", axis=0)

    # ---- in-grid rectangle Gb = (row < 2H) & (col < 2W) ----
    # Built from index comparisons (NOT tiled occupancy): on the wrapped WORK
    # canvas every tiled row is "occupied", so the rectangle must come from the
    # explicit 2H/2W bound instead. twoH = 2*H, twoW = 2*W (both scalar floats).
    n("Add", ["Hf", "Hf"], "twoH")
    n("Add", ["Wf", "Wf"], "twoW")
    n("Less", ["ar_row", "twoH"], "rrect")                       # [1,1,WORK,1] bool
    n("Less", ["ar_col", "twoW"], "crect")                       # [1,1,1,WORK] bool
    n("And", ["rrect", "crect"], "Gb")                          # [1,1,WORK,WORK]

    # ---- colour stamp mask Cb (the one unavoidable 2-D tile) ----
    # colb = colour cell = background channel 0 is OFF.  We only ever Gather colb
    # at in-grid positions (idxr/idxc map into rows 0..H-1, cols 0..W-1), where
    # channel 0 == background, so "ch0 off" == "coloured" there.  This SRC x SRC
    # Slice ([1,1,6,6] = 36 elem) replaces a full 30x30 colour Conv (3600B).
    n("Slice", ["input", "bg_st", "bg_en", "bg_ax"], "bg6")      # [1,1,SRC,SRC]
    n("Less", ["bg6", "half"], "colb")                          # bool: ch0 off (6x6)
    n("Gather", ["colb", "idxr"], "cr", axis=2)                 # [1,1,WORK,SRC] bool
    n("Gather", ["cr", "idxc"], "Craw", axis=3)                # [1,1,WORK,WORK] bool
    n("And", ["Craw", "Gb"], "Cb")                             # clip wrap to grid

    # ---- diag-neighbour mask via a single diagonal Conv ----
    n("Cast", ["Cb"], "Cf", to=TensorProto.FLOAT16)            # fp16 for Conv
    n("Conv", ["Cf", "diag"], "raw", pads=[1, 1, 1, 1])         # 0..4 fp16
    n("Greater", ["raw", "half16"], "rawpos")                  # bool diag-nbr
    n("And", ["rawpos", "Gb"], "cyanish")                      # bool cyan (pre-override)

    # ---- colour index k (uint8 scalar, broadcasts) ----
    n("ReduceMax", ["input"], "pres", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    n("Mul", ["pres", "Wk"], "kparts")
    n("ReduceSum", ["kparts"], "kf", keepdims=1)                # [1,1,1,1] f32 = k
    n("Cast", ["kf"], "ki", to=TensorProto.UINT8)              # uint8 k (broadcast)
    init("v8", np.array(8, np.uint8), np.uint8)
    init("v0", np.array(0, np.uint8), np.uint8)
    init("vout", np.array(10, np.uint8), np.uint8)             # outside sentinel
    # Pad inputs (opset 11): pad each spatial axis up to 30, fill = sentinel 10.
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64),
         np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)

    # ---- L = colour? k : cyan? 8 : in-grid? 0 : 10  (uint8 12x12, 3 Where) ----
    # uint8 is the smallest dtype ORT implements for Where/Equal -> 144B planes.
    n("Where", ["Gb", "v0", "vout"], "Lg")                     # 0 in-grid else 10
    n("Where", ["cyanish", "v8", "Lg"], "Lc")                  # cyan = 8
    n("Where", ["Cb", "ki", "Lc"], "L12")                      # colour overrides

    # ---- pad L back to 30x30 with the outside sentinel, then final Equal ----
    n("Pad", ["L12", "padpads", "padval"], "L", mode="constant")  # uint8 [1,1,30,30]
    n("Equal", ["L", "chan"], "output")                         # -> free BOOL output

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
