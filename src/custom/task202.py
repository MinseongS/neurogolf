"""Task 202 (ARC-AGI 855e0971): per-strata black columns.

True rule (verified on all 230 stored + 300 fresh arc-gen instances, both
orientations of the generator's xpose flag):

  The grid is fully painted with horizontal colored "strata" bands (each band a
  block of `height` rows of one distinct color). Sparse black(0) pixels sit
  inside the bands. For every black pixel at column c within a band, the whole
  vertical extent of that band at column c is painted black in the output.

Because `common.random_colors` returns DISTINCT colors, every color identifies
exactly one band. So a cell (r,c) becomes black iff some row r' that shares the
band-color of row r has a black input pixel in column c. Equivalently, with
  Rrow[k,r] = (row r contains color k),  black[r,c] = input channel 0,
  same[r,r'] = sum_k Rrow[k,r]*Rrow[k,r']   (1 iff r,r' in the same band)
the output-black mask is (same @ black) > 0.

When the generator transposes (xpose=1) the bands run vertically, so the dual
mask (black @ sameCol) applies. Orientation is detected from whether filled
rows are monochrome (max per-row color count == 1 -> not transposed).

Graph keeps every aggregate 1-D or [30,30]; the only canvas-sized tensors are
two [30,30] floats and the final bool mask. Output written via one Where.
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
    # --- per-color row / col occupancy (1 iff color present in that row/col) ---
    # ReduceMax directly on the (free) float32 input -> small [1,10,30] tensors,
    # then cast to fp16 so every downstream intermediate is half-width. All
    # values are 0/1 or small integer sums (<= 30), exact in fp16.
    n("ReduceMax", ["input"], "mrf", axes=[3], keepdims=0)  # [1,10,30] f32
    vi("mrf", TensorProto.FLOAT, [1, 10, 30])
    n("ReduceMax", ["input"], "mcf", axes=[2], keepdims=0)  # [1,10,30] f32
    vi("mcf", TensorProto.FLOAT, [1, 10, 30])
    n("Cast", ["mrf"], "mr", to=F16)
    vi("mr", F16, [1, 10, 30])
    n("Cast", ["mcf"], "mc", to=F16)
    vi("mc", F16, [1, 10, 30])

    # zero out channel 0 (black) so it never couples bands.
    cmask = np.ones((1, 10, 1), np.float16)
    cmask[0, 0, 0] = 0.0
    init("cmask", cmask, np.float16)
    n("Mul", ["mr", "cmask"], "Rrow")                       # [1,10,30] color x row
    vi("Rrow", F16, [1, 10, 30])
    n("Mul", ["mc", "cmask"], "Rcol")                       # [1,10,30] color x col
    vi("Rcol", F16, [1, 10, 30])

    init("sh2", np.array([10, 30], np.int64), np.int64)
    n("Reshape", ["Rrow", "sh2"], "Rr2")                    # [10,30]
    vi("Rr2", F16, [10, 30])
    n("Reshape", ["Rcol", "sh2"], "Rc2")                    # [10,30]
    vi("Rc2", F16, [10, 30])

    # same[r,r'] = sum_k Rrow[k,r]*Rrow[k,r'] = Rrow^T @ Rrow  -> [30,30]
    n("Transpose", ["Rr2"], "RrT", perm=[1, 0])             # [30,30] row x color
    vi("RrT", F16, [30, 10])
    n("MatMul", ["RrT", "Rr2"], "sameR")                    # [30,30]
    vi("sameR", F16, [30, 30])
    n("Transpose", ["Rc2"], "RcT", perm=[1, 0])
    vi("RcT", F16, [30, 10])
    n("MatMul", ["RcT", "Rc2"], "sameC")                    # [30,30]
    vi("sameC", F16, [30, 30])

    # black[r,c] = input channel 0, as [30,30]
    Wb = np.zeros((1, 10, 1, 1), np.float32)
    Wb[0, 0, 0, 0] = 1.0
    init("Wb", Wb)
    n("Conv", ["input", "Wb"], "blackf")                    # [1,1,30,30] f32
    vi("blackf", TensorProto.FLOAT, [1, 1, 30, 30])
    init("sh30", np.array([30, 30], np.int64), np.int64)
    n("Reshape", ["blackf", "sh30"], "black32")             # [30,30] f32
    vi("black32", TensorProto.FLOAT, [30, 30])
    n("Cast", ["black32"], "black", to=F16)
    vi("black", F16, [30, 30])

    # non-transposed: outblack = (sameR @ black) > 0
    n("MatMul", ["sameR", "black"], "obR")                  # [30,30]
    vi("obR", F16, [30, 30])
    # transposed: outblack = (black @ sameC) > 0
    n("MatMul", ["black", "sameC"], "obC")                  # [30,30]
    vi("obC", F16, [30, 30])

    # orientation: max per-row color count.  ==1 (filled monochrome rows) -> not
    # transposed; >=2 -> transposed.
    n("ReduceSum", ["Rr2"], "rowcc", axes=[0], keepdims=0)  # [30] per-row count
    vi("rowcc", F16, [30])
    n("ReduceMax", ["rowcc"], "maxcc", axes=[0], keepdims=1)  # [1]
    vi("maxcc", F16, [1])
    init("th15", np.array([1.5], np.float16), np.float16)
    n("Less", ["maxcc", "th15"], "nonxpose")                # [1] bool
    vi("nonxpose", TensorProto.BOOL, [1])

    n("Where", ["nonxpose", "obR", "obC"], "ob")            # [30,30]
    vi("ob", F16, [30, 30])

    init("zero", np.array(0.0, np.float16), np.float16)
    n("Greater", ["ob", "zero"], "obb")                     # [30,30] bool
    vi("obb", TensorProto.BOOL, [30, 30])
    init("sh1130", np.array([1, 1, 30, 30], np.int64), np.int64)
    n("Reshape", ["obb", "sh1130"], "mask")                 # [1,1,30,30] bool
    vi("mask", TensorProto.BOOL, [1, 1, 30, 30])

    # output = mask ? black_onehot : input  (black cell = channel 0 one-hot)
    blackoh = np.zeros((1, 10, 1, 1), np.float32)
    blackoh[0, 0, 0, 0] = 1.0
    init("blackoh", blackoh)
    n("Where", ["mask", "blackoh", "input"], "output")

    return _model(nodes, inits, vinfos)
