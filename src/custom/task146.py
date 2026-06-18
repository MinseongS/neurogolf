"""task146 (ARC-AGI 662c240a) — output the asymmetric 3x3 block.

Rule (from the generator):
  The input is a 9x3 grid = three 3x3 colour blocks stacked vertically
  (block i = input rows 3i..3i+2, cols 0..2).  Exactly TWO of the three
  blocks are symmetric along the main diagonal (block[r][c]==block[c][r])
  and exactly ONE is asymmetric.  The 3x3 output is the asymmetric block.

Net (opset 11, tiny working canvas):
  1. Slice the FREE input to the 9x3 active region -> [1,10,9,3] (1080B).
  2. Conv (sum_k k*input_k) -> colour-index plane colf9 [1,1,9,3] (108B).
  3. Within-block transpose: reshape colf9 -> [1,3,3,3] (block,r,c),
     Transpose r<->c -> [1,3,3,3], reshape back -> colf9t [1,1,9,3].
  4. asym per cell = (colf9 != colf9t); per-block asym count via reshape
     [1,3,9] + ReduceSum; ArgMax over the 3 blocks -> idx scalar.
     (exactly one block has nonzero asym count -> unique, exact.)
  5. row indices [3*idx, 3*idx+1, 3*idx+2]; Gather those rows of colf9
     -> box [1,1,3,3].
  6. Cast box -> uint8, Pad to [1,1,30,30] uint8 (900B, bg pad value 0),
     output = Equal(lpad, arange[1,10,1,1]) -> FREE BOOL one-hot output.

Dominant intermediates: cropped input 1080B + label pad 900B; everything
else is <200B.  No full-canvas fp32 plane is ever materialised.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
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

    # ---- 1. crop FREE input to 9x3 active region ----
    # crop channels 1..9 (ch0=background is never set in the 9x3 grid) and the
    # 9x3 active region in ONE Slice -> [1,9,9,3] (972B).
    init("st", np.array([0, 1, 0, 0], np.int64), np.int64)
    init("en", np.array([1, 10, 9, 3], np.int64), np.int64)
    init("ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "st", "en", "ax"], "crop")        # [1,9,9,3] f32 972B

    # ---- 2. colour-index plane (fp16; values 0..9 are fp16-exact) ----
    kw = np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1)
    init("kw", kw, np.float32)
    n("Conv", ["crop", "kw"], "colf9f", kernel_shape=[1, 1])  # [1,1,9,3] f32
    n("Cast", ["colf9f"], "colf9", to=TensorProto.FLOAT16)    # [1,1,9,3] fp16 54B

    # ---- 3. within-block transpose (fp16) ----
    init("sh3333", np.array([1, 3, 3, 3], np.int64), np.int64)
    n("Reshape", ["colf9", "sh3333"], "blk")               # [1,3,3,3] (block,r,c)
    n("Transpose", ["blk"], "blkt", perm=[0, 1, 3, 2])     # swap r<->c
    init("sh1193", np.array([1, 1, 9, 3], np.int64), np.int64)
    n("Reshape", ["blkt", "sh1193"], "colf9t")             # [1,1,9,3] fp16

    # ---- 4. per-block asymmetry -> idx (the block with fewest EQUAL cells) ----
    # symmetric block: all 9 cells equal their transpose (count 9);
    # asymmetric block: <9 -> unique min -> ArgMin, no Sub needed.
    n("Equal", ["colf9", "colf9t"], "eq")                  # bool [1,1,9,3] 27B
    init("sh139", np.array([1, 3, 9], np.int64), np.int64)
    n("Reshape", ["eq", "sh139"], "eqr")                   # bool [1,3,9] 27B
    n("Cast", ["eqr"], "eq39", to=TensorProto.FLOAT16)     # fp16 [1,3,9] 54B
    n("ReduceSum", ["eq39"], "sym", axes=[2], keepdims=0)  # [1,3] equal-count per block
    n("ArgMin", ["sym"], "idx", axis=1, keepdims=0)        # [1] int64
    # idx is shape [1]; squeeze to scalar
    n("Squeeze", ["idx"], "idxs", axes=[0])                # scalar int64

    # ---- 5. gather the 3 rows of the selected block ----
    # rowidx = 3*idx + [0,1,2]
    init("three", np.array(3, np.int64), np.int64)
    init("base012", np.array([0, 1, 2], np.int64), np.int64)
    n("Mul", ["idxs", "three"], "base")                    # scalar 3*idx
    n("Add", ["base", "base012"], "rowidx")                # [3] int64
    n("Gather", ["colf9", "rowidx"], "box", axis=2)        # [1,1,3,3] f32

    # ---- 6. cast + pad + Equal -> output ----
    n("Cast", ["box"], "boxu", to=U8)                      # [1,1,3,3] uint8
    # pad to [1,1,30,30]: pad ends on axes 2,3 by 27
    init("pads", np.array([0, 0, 0, 0, 0, 0, 27, 27], np.int64), np.int64)
    # pad with sentinel 255 (not in 0..9) so cells outside the 3x3 grid match
    # NO channel -> all-zero (the harness expects unset cells outside the grid).
    init("sentinel", np.array(255, np.uint8), np.uint8)
    n("Pad", ["boxu", "pads", "sentinel"], "lpad", mode="constant")  # [1,1,30,30] u8 900B
    init("ar", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["lpad", "ar"], "output")                   # [1,10,30,30] bool

    graph = helper.make_graph(
        nodes, "task146",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", B, [1, 10, 30, 30])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
