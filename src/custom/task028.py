"""task028 (ARC-AGI 1bfc4729) — fixed two-colour frame template.

Rule (from the generator, size=10):
  Input has exactly two coloured pixels: colour0 at row 2 (some column),
  colour1 at row size-3=7 (some column).  The COLUMN positions do NOT affect
  the output.  The output is a FIXED 10x10 label template:
    lab grid (1=colour0, 2=colour1, 0=background):
      1111111111
      1000000001
      1111111111
      1000000001
      1000000001
      2000000002
      2000000002
      2222222222
      2000000002
      2222222222

Encoding (floor-break: build ONE colour-index plane L then route the 10-ch
expansion into the FREE bool output via Equal):
  - colour0 scalar = Sum_k k * onehot_k where onehot = ReduceMax over cols of
    input row 2 (its only nonzero cell is colour0).  Same for colour1 (row 7).
    (background channel ch0 contributes weight 0, so the all-bg cells in the
    row are harmless.)
  - L[1,1,30,30] fp16 = lab1*c0 + lab2*c1 + offgrid*99   (on-grid bg stays 0).
  - output = Equal(L, arange[0..9]) : on-grid bg(0) -> ch0; frame -> its colour;
    off-grid(99) matches nothing -> all-zero (correct, off-grid output is bg-less).
  Only L (1800B fp16) + a few tiny single-channel planes are materialised;
  the 10-channel output is the FREE final tensor.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
I64 = TensorProto.INT64

SIZE = 10  # active canvas (generator always size=10)


def _lab():
    lab = np.zeros((SIZE, SIZE), dtype=int)
    for r in range(0, SIZE // 2):
        lab[r][0] = lab[r][SIZE - 1] = 1
        lab[0][r] = lab[0][SIZE - 1 - r] = 1
        lab[2][r] = lab[2][SIZE - 1 - r] = 1
        lab[SIZE - 1 - r][0] = lab[SIZE - 1 - r][SIZE - 1] = 2
        lab[SIZE - 1][r] = lab[SIZE - 1][SIZE - 1 - r] = 2
        lab[SIZE - 3][r] = lab[SIZE - 3][SIZE - 1 - r] = 2
    return lab


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    lab = _lab()

    # ---- recover the two colour scalars from rows 2 and 7 -------------------
    # row r slice over channels/cols -> [1,10,1,30]; ReduceMax over cols -> [1,10,1,1]
    init("r2_s", np.array([2, 0], np.int64), np.int64)
    init("r2_e", np.array([3, SIZE], np.int64), np.int64)
    init("r7_s", np.array([7, 0], np.int64), np.int64)
    init("r7_e", np.array([8, SIZE], np.int64), np.int64)
    init("rc_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "r2_s", "r2_e", "rc_ax"], "row2")    # [1,10,1,30]
    n("Slice", ["input", "r7_s", "r7_e", "rc_ax"], "row7")    # [1,10,1,30]
    n("ReduceMax", ["row2"], "c0_oh", axes=[3], keepdims=1)    # [1,10,1,1]
    n("ReduceMax", ["row7"], "c1_oh", axes=[3], keepdims=1)    # [1,10,1,1]

    # colour scalar = Sum_k k * onehot_k  (k=channel index; ch0 weight 0)
    karange = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("karange", karange, np.float32)
    n("Mul", ["c0_oh", "karange"], "c0_w")                    # [1,10,1,1]
    n("Mul", ["c1_oh", "karange"], "c1_w")                    # [1,10,1,1]
    n("ReduceSum", ["c0_w"], "c0s", axes=[1], keepdims=1)     # [1,1,1,1]
    n("ReduceSum", ["c1_w"], "c1s", axes=[1], keepdims=1)     # [1,1,1,1]
    n("Cast", ["c0s"], "c0s16", to=F16)
    n("Cast", ["c1s"], "c1s16", to=F16)

    # ---- fixed label masks on the 10x10 ACTIVE region (fp16) ----------------
    lab1 = (lab == 1).astype(np.float16).reshape(1, 1, SIZE, SIZE)
    lab2 = (lab == 2).astype(np.float16).reshape(1, 1, SIZE, SIZE)
    init("lab1", lab1, np.float16)
    init("lab2", lab2, np.float16)

    # ---- L10 = lab1*c0 + lab2*c1  (10x10; on-grid bg stays 0) ---------------
    n("Mul", ["lab1", "c0s16"], "t0")        # [1,1,10,10] fp16
    n("Mul", ["lab2", "c1s16"], "t1")        # [1,1,10,10] fp16
    n("Add", ["t0", "t1"], "L10")            # [1,1,10,10] fp16 colour-index

    # ---- cast to uint8 (half the bytes), pad to 30x30 with sentinel 99 ------
    U8 = TensorProto.UINT8
    n("Cast", ["L10"], "L10u8", to=U8)       # [1,1,10,10] uint8
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - SIZE, 30 - SIZE], np.int64), np.int64)
    init("c99", np.array(99, np.uint8), np.uint8)
    n("Pad", ["L10u8", "pads", "c99"], "L", mode="constant")  # [1,1,30,30] uint8

    # ---- output = Equal(L, arange[0..9]) : FREE 10-ch bool output -----------
    arangeu8 = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("arangeu8", arangeu8, np.uint8)
    n("Equal", ["L", "arangeu8"], "output")  # [1,10,30,30] bool (FREE)

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task028", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
