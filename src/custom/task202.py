"""Task 202 (ARC-AGI 855e0971): per-strata black columns.

True rule (verified on all stored + fresh arc-gen instances, both orientations
of the generator's xpose flag):

  The grid is fully painted with horizontal colored "strata" bands (each band a
  block of `height` rows of one DISTINCT color). Sparse black(0) pixels sit
  inside the bands. For every black pixel at column c within a band, the whole
  vertical extent of that band at column c is painted black in the output.

Because `common.random_colors` returns DISTINCT colors, every color identifies
exactly one band, so two rows are in the same band iff they share a non-black
color. With
  Rrow[k,r] = (row r contains color k>=1),  black[r,c] = input channel 0,
the non-transposed output-black mask is
  obR[r,c] = sum_{r'} (sum_k Rrow[k,r]Rrow[k,r']) black[r',c]
           = sum_k Rrow[k,r] * colblk[k,c],   colblk = Rrow @ black   ([10,30])
i.e. obR = Rrow^T @ (Rrow @ black).  Routing the contraction through the small
[10,30] band-by-column count `colblk` avoids ever materialising a [30,30]
band-similarity matrix.

The xpose=1 case is the dual along columns:
  obC[r,c] = sum_k rowblk[r,k] * Rcol[k,c],   rowblk = black @ Rcol^T  ([30,10])
i.e. obC = (black @ Rcol^T) @ Rcol.

Orientation is detected from whether filled rows are monochrome: max per-row
distinct-color count == 1 -> not transposed; >= 2 -> transposed.

Memory levers vs the previous adopted graph (mem 27063):
  * black extracted by a single fp32 Slice of channel 0 (3600B) + cast to fp16,
    not Conv->Reshape->Cast (which paid 3600+3600+1800).
  * contraction routed through the [10,30] colblk/rowblk counts (600B each)
    instead of two [30,30] band-similarity matrices (1800B each).
  * every full-canvas working tensor is fp16 (half the static plane cost).
The only fp32 tensors are the three free-input reductions/slice (2x [1,10,30]
plus one [1,1,30,30]); everything afterwards is fp16/bool.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits, nodes, vinfos = [], [], []

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

    F16 = TensorProto.FLOAT16
    F32 = TensorProto.FLOAT

    # --- per-color row / col occupancy (1 iff color present in that row/col) ---
    n("ReduceMax", ["input"], "mrf", axes=[3], keepdims=0)   # [1,10,30] f32
    vi("mrf", F32, [1, 10, 30])
    n("ReduceMax", ["input"], "mcf", axes=[2], keepdims=0)   # [1,10,30] f32
    vi("mcf", F32, [1, 10, 30])

    # cast to fp16; fold channel-0 (black) zeroing into a single fp16 multiply
    # so it never couples bands.  The occupancy stays RANK-3 ([1,10,30]) with NO
    # reshape (ReduceMax already yields that shape) and `black` stays its native
    # 4D [1,1,30,30]; ONNX MatMul rank-broadcasting (rank3 vs rank4) batches the
    # contraction without materialising a single extra [30,30] reshape.
    cmask = np.ones((1, 10, 1), np.float16)
    cmask[0, 0, 0] = 0.0
    init("cmask", cmask, np.float16)

    n("Cast", ["mrf"], "mrh", to=F16)
    vi("mrh", F16, [1, 10, 30])
    n("Mul", ["mrh", "cmask"], "Rrow")                       # [1,10,30] color x row
    vi("Rrow", F16, [1, 10, 30])

    n("Cast", ["mcf"], "mch", to=F16)
    vi("mch", F16, [1, 10, 30])
    n("Mul", ["mch", "cmask"], "Rcol")                       # [1,10,30] color x col
    vi("Rcol", F16, [1, 10, 30])

    # black[r,c] = input channel 0, extracted by one fp32 slice -> fp16 [1,1,30,30].
    init("st0", np.array([0], np.int64), np.int64)
    init("en1", np.array([1], np.int64), np.int64)
    init("ax1", np.array([1], np.int64), np.int64)
    n("Slice", ["input", "st0", "en1", "ax1"], "blkslice")   # [1,1,30,30] f32
    vi("blkslice", F32, [1, 1, 30, 30])
    n("Cast", ["blkslice"], "black", to=F16)                 # [1,1,30,30] f16
    vi("black", F16, [1, 1, 30, 30])

    # non-transposed: obR = Rrow^T @ (Rrow @ black)
    # Rrow[1,10,30] x black[1,1,30,30] -> rank-broadcast batch -> [1,1,10,30]
    n("MatMul", ["Rrow", "black"], "colblk")                 # [1,1,10,30] band x col
    vi("colblk", F16, [1, 1, 10, 30])
    n("Transpose", ["Rrow"], "RrowT", perm=[0, 2, 1])        # [1,30,10] row x band
    vi("RrowT", F16, [1, 30, 10])
    n("MatMul", ["RrowT", "colblk"], "obR")                  # [1,1,30,30]
    vi("obR", F16, [1, 1, 30, 30])

    # transposed: obC = (black @ Rcol^T) @ Rcol
    n("Transpose", ["Rcol"], "RcolT", perm=[0, 2, 1])        # [1,30,10] col x band
    vi("RcolT", F16, [1, 30, 10])
    n("MatMul", ["black", "RcolT"], "rowblk")                # [1,1,30,10] row x band
    vi("rowblk", F16, [1, 1, 30, 10])
    n("MatMul", ["rowblk", "Rcol"], "obC")                   # [1,1,30,30]
    vi("obC", F16, [1, 1, 30, 30])

    # orientation: max per-row distinct-color count.  ==1 -> not transposed.
    n("ReduceSum", ["Rrow"], "rowcc", axes=[1], keepdims=0)  # [1,30] per-row count
    vi("rowcc", F16, [1, 30])
    n("ReduceMax", ["rowcc"], "maxcc", axes=[1], keepdims=1)  # [1,1]
    vi("maxcc", F16, [1, 1])
    init("th15", np.array([1.5], np.float16), np.float16)
    n("Less", ["maxcc", "th15"], "nonxpose")                 # [1,1] bool
    vi("nonxpose", TensorProto.BOOL, [1, 1])

    n("Where", ["nonxpose", "obR", "obC"], "ob")             # [1,1,30,30]
    vi("ob", F16, [1, 1, 30, 30])

    init("zero", np.array(0.0, np.float16), np.float16)
    n("Greater", ["ob", "zero"], "mask")                     # [1,1,30,30] bool
    vi("mask", TensorProto.BOOL, [1, 1, 30, 30])

    # output = mask ? black_onehot : input  (black cell = channel-0 one-hot)
    blackoh = np.zeros((1, 10, 1, 1), np.float32)
    blackoh[0, 0, 0, 0] = 1.0
    init("blackoh", blackoh)
    n("Where", ["mask", "blackoh", "input"], "output")

    return _model(nodes, inits, vinfos)
