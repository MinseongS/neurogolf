"""Task 384 (f25fbde4): 2x nearest-neighbor upscale of the yellow bounding box.

Rule (from the ARC-GEN generator): the input (always 9x9, color 4 = yellow)
contains a set of yellow pixels. Crop to their bounding box (height h, width w)
and upscale 2x so each cell becomes a 2x2 block; the output grid is 2h x 2w.

Output cell (R, C) = input(minr + R//2, minc + C//2). Inside the 2h x 2w output
box every cell is yellow (channel 4) or background (channel 0); outside the box
all channels are 0.

The official scorer thresholds the output at `> 0`, so only the SIGN matters and
we encode both colors in ONE signed channel:

    v = (channel 4) - (channel 0)   ->  +1 yellow, -1 grid-background, 0 elsewhere

(the 9x9 grid is top-left, so rows/cols 9..29 are all-zero in every channel; the
source-index Gather redirects out-of-box cells to the empty edge row/col 29, so
the gathered value there is 0.)  v is built by a single 1x1 Conv on the input.

  * yellow row/col extent (the bbox) comes from int8 reduce-max of the yellow
    mask `v > 0`; min/max -> 2x-upscale source-index vectors
    base = [0,0,1,1,...] + min, redirected to 29 past max.
  * v is cast to int8 (900 B/channel) and gathered along rows then cols, giving
    x in {+1, -1, 0}.
  * a final Conv with bias writes `output` directly (free):
        ch0 = -x - 0.5  (>0 only on background)
        ch4 =  x        (>0 only on yellow)
        all other channels 0.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    base = np.repeat(np.arange(15), 2).astype(np.int32)   # [0,0,1,1,...,14,14]
    init("BASE", base, np.int32)                          # [30]
    init("RIDX", np.arange(30, dtype=np.int32), np.int32)  # [30]
    init("BIG", np.full(30, 999, dtype=np.int32), np.int32)
    init("NEG", np.full(30, -1, dtype=np.int32), np.int32)
    init("C29", np.full(30, 29, dtype=np.int32), np.int32)
    init("zero_f", np.array(0.0, dtype=np.float32), np.float32)

    # v = ch4 - ch0  (single 1x1 conv over the input)
    wv = np.zeros((1, 10, 1, 1), dtype=np.float32)
    wv[0, 4, 0, 0] = 1.0
    wv[0, 0, 0, 0] = -1.0
    init("WV", wv, np.float32)

    # output weight + bias (bias trick over x in {+1,-1,0})
    wo = np.zeros((10, 1, 1, 1), dtype=np.float32)
    wo[4, 0, 0, 0] = 1.0    # ch4 = x
    wo[0, 0, 0, 0] = -1.0   # ch0 = -x - 0.5
    bo = np.zeros((10,), dtype=np.float32)
    bo[0] = -0.5
    init("WO", wo, np.float32)
    init("BO", bo, np.float32)

    n("Conv", ["input", "WV"], "v_f")            # float [1,1,30,30]
    n("Cast", ["v_f"], "v_i8", to=onnx.TensorProto.INT8)  # int8 900 (for gather)

    # row/col max of v in {+1,-1,0}: a row/col contains yellow iff its max > 0.
    n("ReduceMax", ["v_f"], "rowmax", axes=[0, 1, 3], keepdims=0)  # [30] float
    n("ReduceMax", ["v_f"], "colmax", axes=[0, 1, 2], keepdims=0)  # [30] float

    def bbox(occ, lo_name, hi_name):
        n("Greater", [occ, "zero_f"], occ + "_m")          # bool [30]
        n("Where", [occ + "_m", "RIDX", "BIG"], occ + "_lo")
        n("ReduceMin", [occ + "_lo"], lo_name, axes=[0], keepdims=0)  # scalar
        n("Where", [occ + "_m", "RIDX", "NEG"], occ + "_hi")
        n("ReduceMax", [occ + "_hi"], hi_name, axes=[0], keepdims=0)  # scalar

    bbox("rowmax", "minr", "maxr")
    bbox("colmax", "minc", "maxc")

    def srcvec(mn, mx, name):
        n("Add", ["BASE", mn], name + "_raw")              # [30]
        n("Greater", [name + "_raw", mx], name + "_over")  # bool [30]
        n("Where", [name + "_over", "C29", name + "_raw"], name)  # int32 [30]
        return name

    srcvec("minr", "maxr", "ridx")
    srcvec("minc", "maxc", "cidx")

    # gather v (int8) -> x in {+1,-1,0}
    n("Gather", ["v_i8", "ridx"], "vr", axis=2)   # int8 [1,1,30,30]
    n("Gather", ["vr", "cidx"], "vc", axis=3)     # int8 [1,1,30,30]
    n("Cast", ["vc"], "x_f", to=onnx.TensorProto.FLOAT)  # float 3600

    n("Conv", ["x_f", "WO", "BO"], "output")
    return _model(nodes, inits)
