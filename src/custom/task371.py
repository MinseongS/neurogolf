"""Task 371 (ARC-AGI e9614598): green plus at the midpoint of two blue dots.

Rule (orientation-equivariant, so xpose is handled for free): the input has
exactly two blue(1) cells, separated by 2*space along ONE axis. The output keeps
both blue cells and draws a green(3) plus (centre + 4 orthogonal neighbours) at
their MIDPOINT. Because the two dots are symmetric about the centre, the centroid
(mean blue row, mean blue col) IS the plus centre in both orientations, and both
means are integer (the dots share one coordinate and straddle the other by 2*space).

Active region: width in {10,12,14}, height in {6..14}, and xpose only swaps them,
so every grid fits inside 14x14. We work on a 14x14 slice of channel 1, keeping
all full planes small, and pad the final boolean plus-mask back to 30x30.

Encoding (no [1,10,H,W] plane; profiles, then one 14x14 fp16 plane):
  blue = input channel 1 cropped to 14x14 (fp32 [1,1,14,14])
  pr = ReduceSum(blue,axis=3) [1,1,14,1], pc = ReduceSum(blue,axis=2) [1,1,1,14]
  mean_row = Sum_r r*pr / 2 , mean_col = Sum_c c*pc / 2   (count==2; tiny scalars)
  dr=|rowidx-mr| [1,1,14,1], dc=|colidx-mc| [1,1,1,14]
  mask14 = (dr+dc) < 1.5    (radius-1 L1 ball = a plus; only 14x14 fp16 plane)
  cond = Pad(mask14 -> 30x30) as bool
  output = Where(cond, green_onehot[1,10,1,1], input)   -> FREE 10-ch output.

Blue dots sit at L1 distance space>=3 from the centre, never in the plus, so they
survive the Where; input has no green so the value is a constant one-hot (chan 3).
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model

F16 = TensorProto.FLOAT16
F32 = TensorProto.FLOAT
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8

K = 14  # active canvas


def build(task):
    inits = []
    nodes = []
    vinfos = []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # blue = channel 1, cropped to KxK -> fp32 [1,1,K,K]
    init("b_s", np.array([0, 1, 0, 0], np.int64), np.int64)
    init("b_e", np.array([1, 2, K, K], np.int64), np.int64)
    init("b_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "b_s", "b_e", "b_ax"], "blue")
    vi("blue", F32, [1, 1, K, K])

    # row / col profiles (fp32, tiny)
    n("ReduceSum", ["blue"], "pr", axes=[3], keepdims=1)   # [1,1,K,1]
    vi("pr", F32, [1, 1, K, 1])
    n("ReduceSum", ["blue"], "pc", axes=[2], keepdims=1)   # [1,1,1,K]
    vi("pc", F32, [1, 1, 1, K])

    # profile-weighting coords on the KxK crop (fp32, tiny)
    rcoordK = np.arange(K, dtype=np.float32).reshape(1, 1, K, 1)
    ccoordK = np.arange(K, dtype=np.float32).reshape(1, 1, 1, K)
    init("rcoordK", rcoordK)
    init("ccoordK", ccoordK)

    # weighted profile sums on TINY tensors -> 2*mean_row, 2*mean_col
    n("Mul", ["pr", "rcoordK"], "wr")                      # [1,1,K,1]
    vi("wr", F32, [1, 1, K, 1])
    n("Mul", ["pc", "ccoordK"], "wc")                      # [1,1,1,K]
    vi("wc", F32, [1, 1, 1, K])
    n("ReduceSum", ["wr"], "sr", axes=[2], keepdims=1)     # [1,1,1,1]
    vi("sr", F32, [1, 1, 1, 1])
    n("ReduceSum", ["wc"], "sc", axes=[3], keepdims=1)     # [1,1,1,1]
    vi("sc", F32, [1, 1, 1, 1])
    init("HALF", np.array(0.5, np.float32))
    n("Mul", ["sr", "HALF"], "mr32")                       # mean_row
    vi("mr32", F32, [1, 1, 1, 1])
    n("Mul", ["sc", "HALF"], "mc32")                       # mean_col
    vi("mc32", F32, [1, 1, 1, 1])
    n("Cast", ["mr32"], "mr", to=F16)
    vi("mr", F16, [1, 1, 1, 1])
    n("Cast", ["mc32"], "mc", to=F16)
    vi("mc", F16, [1, 1, 1, 1])

    # L1 distance over the full 30x30 ramps in fp16 (centre always < 14 < 30,
    # so off-grid cells get large dist -> never plus)
    rcoord = np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1)
    ccoord = np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30)
    init("rcoord", rcoord, np.float16)
    init("ccoord", ccoord, np.float16)
    n("Sub", ["rcoord", "mr"], "rd0")
    vi("rd0", F16, [1, 1, 30, 1])
    n("Abs", ["rd0"], "rd")
    vi("rd", F16, [1, 1, 30, 1])
    n("Sub", ["ccoord", "mc"], "cd0")
    vi("cd0", F16, [1, 1, 1, 30])
    n("Abs", ["cd0"], "cd")
    vi("cd", F16, [1, 1, 1, 30])
    n("Add", ["rd", "cd"], "dist")                         # [1,1,30,30] fp16 (broadcast)
    vi("dist", F16, [1, 1, 30, 30])

    # plus mask = dist < 1.5 -> bool [1,1,30,30]  (centre + 4 neighbours)
    init("th15", np.array(1.5, np.float16), np.float16)
    n("Less", ["dist", "th15"], "cond")
    vi("cond", BOOL, [1, 1, 30, 30])

    # output = cond ? green_onehot : input
    green = np.zeros((1, 10, 1, 1), np.float32)
    green[0, 3, 0, 0] = 1.0
    init("green", green)
    n("Where", ["cond", "green", "input"], "output")

    return _model(nodes, inits, vinfos)
