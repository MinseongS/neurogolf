"""task312 (ARC-AGI c9f8e694) — recolour each gray box by its row's column-0 colour.

Rule (exact, from the ARC-GEN generator, verified fresh):
  Grid is size=12 (12x12, placed at the top-left of the 30x30 canvas).  Column 0 of
  every row r holds a per-row pattern colour pattern[r] (drawn from 2-3 random NON-gray
  colours; with start=1 row 0 is background 0).  The grid also holds 3-4 axis-aligned
  rectangular GRAY (colour 5) boxes at columns col = randint(2, size-wide) >= 2, so the
  boxes NEVER touch column 0.

  OUTPUT(r,c) = pattern[r] where input(r,c) is gray, else input(r,c) unchanged.

  This is the separable per-ROW-colour-vector (X) box-mask escape:
      output = Where(gray_mask[1,1,30,30],  col0[1,10,30,1],  input[1,10,30,30])
  col0 = the column-0 one-hot slice of the input (the per-row colour vector, broadcast
  over columns); gray_mask = the gray channel (broadcast over the 10 colour channels);
  their product / select lands DIRECTLY in the FREE `output` — no [1,1,30,30] box plane
  of colours and no [1,10,30,30] carrier ever materialises.

  Channel 5 (gray) is correctly zeroed in the output: where gray, col0[5]=0 (column 0 is
  never gray); where not gray, input[5]=0 (the cell is not a gray box).

Memory note (beats the 5700B `Where(slice_ch5_full, col0, input)` baseline):
  The dominant plane in the naive net is the FULL fp32 channel-5 slice (3600B).  Because
  the grid is fixed 12x12 at the top-left, we slice channel 5 AND crop to 12x12 in ONE
  Slice (576B f32), Cast to uint8 (144B), Pad uint8 back to 30x30 (900B), then Greater->
  BOOL condition (900B).  col0 stays the [1,10,30,1] f32 slice (1200B).  Total working
  memory ~ 576+144+900+900+1200 = 3720B  (pts ~ 16.83, +~0.48 over 16.35).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

WORK = 12  # grid is fixed 12x12 at the top-left


def build(task):
    inits, nodes = [], []

    _NP = {F16: np.float16, F32: np.float32, BOOL: np.bool_,
           U8: np.uint8, I64: np.int64}

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=_NP[dtype]), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # --- gray box mask: slice channel 5 AND crop to the 12x12 active grid in ONE Slice
    init("g_st", np.array([5, 0, 0], np.int64), I64)          # ch5, row0, col0
    init("g_en", np.array([6, WORK, WORK], np.int64), I64)    # ch6, row12, col12
    init("g_ax", np.array([1, 2, 3], np.int64), I64)          # channel, row, col axes
    n("Slice", ["input", "g_st", "g_en", "g_ax"], "gray_f")   # [1,1,12,12] f32

    # Cast to uint8 (144B), Pad back to 30x30 uint8 (opset-11 Pad accepts uint8),
    # then Greater -> bool condition.
    n("Cast", ["gray_f"], "gray_u8", to=U8)                   # [1,1,12,12] uint8
    init("p_full", np.array([0, 0, 0, 0, 0, 0, 30 - WORK, 30 - WORK], np.int64), I64)
    init("pv_u8", np.array(0, np.uint8), U8)
    n("Pad", ["gray_u8", "p_full", "pv_u8"], "gray_u8_30", mode="constant")  # [1,1,30,30] u8
    init("zero_u8", np.array(0, np.uint8), U8)
    n("Greater", ["gray_u8_30", "zero_u8"], "gray_b")         # [1,1,30,30] bool

    # --- per-row colour vector: the column-0 one-hot slice [1,10,30,1] f32
    init("c0_st", np.array([0], np.int64), I64)
    init("c0_en", np.array([1], np.int64), I64)
    init("c0_ax", np.array([3], np.int64), I64)
    n("Slice", ["input", "c0_st", "c0_en", "c0_ax"], "col0")  # [1,10,30,1] f32

    # --- route into the FREE output: paint pattern[r] over gray boxes, else keep input
    n("Where", ["gray_b", "col0", "input"], "output")         # [1,10,30,30] FREE

    graph = helper.make_graph(
        nodes, "task312",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
