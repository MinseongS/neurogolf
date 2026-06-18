"""task393 (ARC-AGI f8ff0b80) — "sort the box colours by descending pixel count".

Rule (from the generator):
  The INPUT renders num_boxes=3 solid "creatures" (contiguous pixel blobs) of
  DISTINCT sizes onto a size x size canvas (size=12, harness-padded to 30x30).
  The counts are sampled (without replacement) from range(3,15) and sorted
  DESCENDING, so box 0 has the most pixels, box 2 the fewest.  Each box i uses
  one colour colors[i] in 1..9; background is colour 0 (channel 0), which is the
  most frequent colour overall.

  OUTPUT is a 3x1 column: output[i] = the colour of the box with the i-th
  largest pixel count.  output[0] = most-frequent box colour, output[2] = least.

Encoding (no per-cell plane — everything is tiny [1,10,..] tensors; the COUNT-RANK
lever, ported from task391):
  cnt_k = ReduceSum(input_k) over space -> [1,10,1,1] f32.  Channel 0 (canvas
  background) is the MOST frequent colour and must be excluded from the ranking,
  so mask it to 0 (chmask: ch0=0, ch1..9=1).  Then rank_k = #{ j : cnt_j > cnt_k }
  via a [10,10] Greater + ReduceSum.  Counts are distinct, so the three box
  colours get unique ranks 0,1,2 (the masked ch0 and the absent colours all sit
  at cnt 0 and rank 3, never matched).  Build H[1,10,30,1] = Equal(rank,
  target[1,1,30,1]) with target=[0,1,2,99,99,...], AND with a col-0 mask
  [1,1,1,30] -> the FREE bool output [1,10,30,30].  No [1,10,30,30] intermediate
  is ever materialised; no iterative argmax / penalty / Pad needed.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- per-colour pixel counts (input is FREE), keep everything fp16 ----
    n("ReduceSum", ["input"], "cnt0", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    chmask = np.ones((1, 10, 1, 1), np.float16)
    chmask[0, 0, 0, 0] = 0.0  # exclude background channel 0
    init("chmask", chmask, np.float16)
    n("Cast", ["cnt0"], "cnt0h", to=TensorProto.FLOAT16)
    n("Mul", ["cnt0h", "chmask"], "cnt")  # [1,10,1,1] f16, ch0 zeroed

    # ---- iterative ArgMax (3 picks, descending count) --------------------
    # cnt is [1,10,1,1] f16 with ch0 masked to 0.  Channel == colour (the
    # harness one-hots colour value -> channel), so ArgMax over channels gives
    # the box colour directly.  Subtract a large penalty at each pick to take
    # the next-largest.  All tensors are [1,10,1,1]/scalars -> a few bytes each.
    arange = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("arange", arange, np.float16)         # channel index ramp
    init("neg", np.array(-1.0, np.float16), np.float16)  # penalty fill

    def pick(src, idx):
        am = n("ArgMax", [src], f"pick{idx}", axis=1, keepdims=1)  # [1,1,1,1] i64
        amf = n("Cast", [am], f"pickf{idx}", to=TensorProto.FLOAT16)
        oh = n("Equal", [amf, "arange"], f"oh{idx}")               # [1,10,1,1] bool
        return oh

    # masked counts are >=0; replacing the picked channel with -1 makes the next
    # ArgMax skip it (Where lands its output directly, dropping the onehot*Mul+Sub
    # penalty chain -> 2 fewer fp16 planes per step).
    oh0 = pick("cnt", 0)
    n("Where", ["oh0", "neg", "cnt"], "cnt1")   # [1,10,1,1] f16
    oh1 = pick("cnt1", 1)
    n("Where", ["oh1", "neg", "cnt1"], "cnt2")  # [1,10,1,1] f16
    oh2 = pick("cnt2", 2)

    # ---- assemble [1,10,3,1] one-hot (rows = rank order) -----------------
    n("Cast", [oh0], "u0", to=TensorProto.UINT8)  # [1,10,1,1] uint8
    n("Cast", [oh1], "u1", to=TensorProto.UINT8)
    n("Cast", [oh2], "u2", to=TensorProto.UINT8)
    n("Concat", ["u0", "u1", "u2"], "Hu", axis=2)  # [1,10,3,1] uint8 (30B)

    # Pad the [1,10,3,1] block into the [1,10,30,30] canvas (rows 3->30 below,
    # cols 1->30 right). Pad on uint8 runs under ORT_DISABLE_ALL.
    pads = np.array([0, 0, 0, 0, 0, 0, 27, 29], np.int64)  # begin x4, end x4
    init("pads", pads, np.int64)
    n("Pad", ["Hu", "pads"], "output", mode="constant")  # [1,10,30,30] uint8 (free)

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task393", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
