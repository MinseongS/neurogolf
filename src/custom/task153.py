"""task153 (ARC-AGI 681b3aeb) — reassemble two 3x3 "halves" into the 3x3 output.

Rule (from the generator, verified fresh):
  A 10x10 grid holds TWO 3x3 windows at random non-overlapping positions
  (|row gap|>2, |col gap|>2).  Window `idx` is filled with colour `colors[idx]`
  exactly at the cells where the 3x3 partition mask `idxs` equals `idx`.  Group 1
  (`idxs==1`) is a `continuous_creature` grown from (0,0): it is 4-connected and
  ALWAYS contains its window's top-left corner, so its bbox-min == window anchor.
  Group 0 is the complement (forced diagonally connected).  Both halves use a
  distinct colour from `random_colors(2)` (both nonzero).  The 3x3 output is the
  superposition: out[r][c] = colour of whichever window covers (r,c) (the two
  groups partition all 9 cells, so every output cell is filled exactly once).

  Key reconstruction (verified 6000/6000):
    * The CREATURE colour, placed with its bbox top-left at output (0,0), lands
      exactly on its output cells; the OTHER colour fills the remaining cells.
    * To identify the creature among the two colours: place colour C at its
      bbox-TL; the complement region (3x3 minus C's pattern), re-normalised to
      its own bbox-TL, equals the OTHER colour's bbox-TL-normalised pattern iff
      C is the creature.  (When both colours satisfy this, both placements give
      the identical output, so either may be chosen.)

  ONNX (opset 11), all tensors on a tiny 10x10 / 3x3 canvas:
    1. Slice input -> [1,10,10,10]; colf = sum_k k*input_k -> [1,1,10,10] f32.
    2. cmax = max colour; maskA = (colf==cmax); maskB = (colf>0 & !=cmax);
       cmin = max(colf*maskB).
    3. For each mask: first occupied row/col (ArgMax of 1-D occupancy), Gather a
       3x3 bbox-TL window -> patA, patB (bool 3x3).
    4. compA = ~patA; renormalise compA to its bbox-TL; isCreatA = (renorm==patB).
    5. candA = Where(patA, cmax, cmin); candB = Where(patB, cmin, cmax);
       out3 = Where(isCreatA, candA, candB)  (3x3 colour index).
    6. Pad out3 to 30x30 with sentinel 10; output = Equal(L, arange(10)) -> BOOL.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I64 = TensorProto.INT64
U8 = TensorProto.UINT8
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- 1. collapse to a colour-index plane via 1x1 Conv, then slice 10x10 --
    # Conv weight [1,10,1,1] = k -> sum_k k*input_k, output [1,1,30,30] (no
    # [1,10,30,30] product materialised), then crop to the 10x10 active canvas.
    kw = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("kw", kw, np.float32)
    n("Conv", ["input", "kw"], "colf30")                         # [1,1,30,30] f32
    init("s_starts", np.array([0, 0], np.int64), np.int64)
    init("s_ends", np.array([10, 10], np.int64), np.int64)
    init("s_axes", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["colf30", "s_starts", "s_ends", "s_axes"], "colf")  # [1,1,10,10]

    # ---- 2. the two colours and their masks ----------------------------------
    n("ReduceMax", ["colf"], "cmax", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]
    n("Equal", ["colf", "cmax"], "maskA")                        # bool [1,1,10,10]
    init("zero", np.array(0.0, np.float32), np.float32)
    n("Greater", ["colf", "zero"], "nonbg")                      # bool present
    n("Not", ["maskA"], "notA")
    n("And", ["nonbg", "notA"], "maskB")                         # bool
    # cmin = max over (colf where maskB)
    n("Cast", ["maskB"], "maskBf", to=F32)
    n("Mul", ["colf", "maskBf"], "colfB")
    n("ReduceMax", ["colfB"], "cmin", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]

    # ---- 3. bbox-TL 3x3 window extractor (helper, returns pat name) ----------
    init("base3", np.arange(3, dtype=np.float32), np.float32)    # [3]
    init("shp1", np.array([1], np.int64), np.int64)
    init("c0", np.array(0.0, np.float32), np.float32)
    init("c12", np.array(12.0, np.float32), np.float32)         # clip idx<=12 (13-pad)
    # zero-pad mask canvas to 13x13 so a bbox-min near the edge can still gather
    # 3 rows/cols (the padded rows are zero -> harmless duplicates).
    init("pad13", np.array([0, 0, 0, 0, 0, 0, 3, 3], np.int64), np.int64)
    init("fz16", np.array(0.0, np.float16), np.float16)
    init("fzero", np.array(0.0, np.float32), np.float32)

    def window(mask, tag):
        # mask: bool [1,1,10,10]  -> work in fp16 (planes counted at half)
        mf = n("Cast", [mask], tag + "_mf", to=F16)              # fp16 0/1
        n("Pad", [mf, "pad13", "fz16"], tag + "_mp", mode="constant")  # [1,1,13,13]
        n("ReduceMax", [mf], tag + "_rany", axes=[3], keepdims=1)  # [1,1,10,1]
        n("ReduceMax", [mf], tag + "_cany", axes=[2], keepdims=1)  # [1,1,1,10]
        n("ArgMax", [tag + "_rany"], tag + "_fr", axis=2, keepdims=1)  # [1,1,1,1] i64
        n("ArgMax", [tag + "_cany"], tag + "_fc", axis=3, keepdims=1)  # [1,1,1,1] i64
        n("Cast", [tag + "_fr"], tag + "_frf", to=F32)
        n("Cast", [tag + "_fc"], tag + "_fcf", to=F32)
        n("Reshape", [tag + "_frf", "shp1"], tag + "_frs")       # [1]
        n("Reshape", [tag + "_fcf", "shp1"], tag + "_fcs")
        n("Add", ["base3", tag + "_frs"], tag + "_ridf")         # [3]
        n("Add", ["base3", tag + "_fcs"], tag + "_cidf")
        n("Clip", [tag + "_ridf", "c0", "c12"], tag + "_ridc")
        n("Clip", [tag + "_cidf", "c0", "c12"], tag + "_cidc")
        n("Cast", [tag + "_ridc"], tag + "_rid", to=I64)         # [3]
        n("Cast", [tag + "_cidc"], tag + "_cid", to=I64)
        n("Gather", [tag + "_mp", tag + "_rid"], tag + "_w2", axis=2)  # [1,1,3,13]
        n("Gather", [tag + "_w2", tag + "_cid"], tag + "_w", axis=3)  # [1,1,3,3]
        init(tag + "_half", np.array(0.5, np.float16), np.float16)
        return n("Greater", [tag + "_w", tag + "_half"], tag + "_pat")  # bool [1,1,3,3]

    patA = window("maskA", "A")   # high colour pattern, bbox-TL
    patB = window("maskB", "B")   # low  colour pattern, bbox-TL

    # ---- 4. is colour A the creature? ----------------------------------------
    # compA = NOT patA ; renormalise compA to its own bbox-TL ; compare to patB
    n("Not", [patA], "compA")                                    # bool [1,1,3,3]
    n("Cast", ["compA"], "compAf", to=F32)
    # zero-pad compA to 5x5 so a bbox-min up to row/col 2 can still gather 3 rows
    n("Pad", ["compAf", "pad13", "fzero"], "compAp", mode="constant")  # [1,1,6,6]
    # first occupied row/col of compA on the original 3x3 canvas
    n("ReduceMax", ["compAf"], "cA_rany", axes=[3], keepdims=1)  # [1,1,3,1]
    n("ReduceMax", ["compAf"], "cA_cany", axes=[2], keepdims=1)  # [1,1,1,3]
    n("ArgMax", ["cA_rany"], "cA_fr", axis=2, keepdims=1)
    n("ArgMax", ["cA_cany"], "cA_fc", axis=3, keepdims=1)
    n("Cast", ["cA_fr"], "cA_frf", to=F32)
    n("Cast", ["cA_fc"], "cA_fcf", to=F32)
    n("Reshape", ["cA_frf", "shp1"], "cA_frs")
    n("Reshape", ["cA_fcf", "shp1"], "cA_fcs")
    n("Add", ["base3", "cA_frs"], "cA_ridf")
    n("Add", ["base3", "cA_fcs"], "cA_cidf")
    init("c5", np.array(5.0, np.float32), np.float32)            # clip idx<=5 (6-pad)
    n("Clip", ["cA_ridf", "c0", "c5"], "cA_ridc")
    n("Clip", ["cA_cidf", "c0", "c5"], "cA_cidc")
    n("Cast", ["cA_ridc"], "cA_rid", to=I64)
    n("Cast", ["cA_cidc"], "cA_cid", to=I64)
    n("Gather", ["compAp", "cA_rid"], "cA_w2", axis=2)           # [1,1,3,6]
    n("Gather", ["cA_w2", "cA_cid"], "cA_w", axis=3)             # [1,1,3,3]
    init("half2", np.array(0.5, np.float32), np.float32)
    n("Greater", ["cA_w", "half2"], "compA_n")                   # bool [1,1,3,3]
    # equal to patB at all 9 cells -> isCreatA
    n("Equal", ["compA_n", patB], "creq")                        # bool [1,1,3,3]
    n("Cast", ["creq"], "creqf", to=F32)
    n("ReduceMin", ["creqf"], "isCreatAf", axes=[1, 2, 3], keepdims=1)  # [1,1,1,1]
    init("halff", np.array(0.5, np.float32), np.float32)
    n("Greater", ["isCreatAf", "halff"], "isCreatA")             # bool [1,1,1,1]

    # ---- 5. build the two candidate 3x3 colour-index planes & select ---------
    n("Where", [patA, "cmax", "cmin"], "candA")                  # [1,1,3,3] f32
    n("Where", [patB, "cmin", "cmax"], "candB")                  # [1,1,3,3] f32
    n("Where", ["isCreatA", "candA", "candB"], "out3")           # [1,1,3,3] f32

    # ---- 6. pad to 30x30 with sentinel 10 (uint8), then Equal -> bool output -
    n("Cast", ["out3"], "out3u", to=U8)                          # uint8 colour index
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64), np.int64)
    init("sent", np.array(10, np.uint8), np.uint8)
    n("Pad", ["out3u", "padpads", "sent"], "L", mode="constant")  # [1,1,30,30] u8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                          # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task153", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
