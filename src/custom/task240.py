"""Task 240 (ARC-AGI 9d9215db): 4-fold mirror a diagonal color bitmap and draw
dotted-square rings between diagonal-adjacent colors.

Rule (verified exact on all 266 stored examples, and derived from the ARC-GEN
generator). The 19x19 input holds a small (length 2-4) diagonal of colored
pixels at odd coordinates (2r+1, 2c+1), possibly with a "nextdoor" color one
step along the diagonal, and the whole grid may be h/v-flipped. The output is
fully determined, per chebyshev ring d in {1,3,5,7} (distance from the border),
by exactly two colors:

  * Cd = the diagonal-corner color of ring d  (painted on the 4 ring corners)
  * Kd = the nextdoor/ring color of ring d    (painted on all non-corner odd
          cells of ring d as a dotted square; 0 => those cells stay background)

Because the flips and the mirroring are 4-fold symmetric, Cd and Kd can be read
straight off the *unflipped-canonical* positions: fold the color-index plane by
taking, for each canonical top-left position, the max over its 4 mirror cells.
Canonical positions: corner d at (d,d); ring d at (d,d+2). Center (9,9) is never
used (length<=4 => rings only reach d=7).

Graph (no big float canvases, no symmetrization matmul):
  1. Conv input one-hot -> color-index plane cidx [1,1,30,30].
  2. Flatten, Gather 32 fixed cells (8 groups x 4 mirrors), max each group of 4
     -> 8 group colors. Prepend [-1, 0] -> gcolor_vec[10]
     (index 0 = outside-region sentinel -1, index 1 = in-region background 0,
      indices 2..5 = corner colors of rings 1,3,5,7, 6..9 = ring colors).
  3. A fixed groupid plane [1,1,30,30] maps every canvas cell to its group.
     outidx = Gather(gcolor_vec, groupid)  (=-1 outside, 0 background, color>0).
  4. output = Cast(Equal(outidx, [0..9] ramp)) -> one-hot. The -1 sentinel
     matches no channel so cells outside the 19x19 grid stay all-zero, while
     in-region background maps to channel 0.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def _groupid_plane():
    gid = np.zeros((30, 30), np.int32)
    ring_index = {1: 0, 3: 1, 5: 2, 7: 3}
    for y in range(30):
        for x in range(30):
            if y >= 19 or x >= 19:
                gid[y, x] = 0          # outside grid -> sentinel
                continue
            g = 1                       # in-region background
            ymin = min(y, 18 - y)
            xmin = min(x, 18 - x)
            ring = min(ymin, xmin)
            if (y % 2 == 1) and (x % 2 == 1) and ring in ring_index:
                ri = ring_index[ring]
                g = (2 + ri) if ymin == xmin else (6 + ri)
            gid[y, x] = g
    return gid


def _mirror_flat_indices():
    canon = [(d, d) for d in (1, 3, 5, 7)] + [(d, d + 2) for d in (1, 3, 5, 7)]
    idx = []
    for (y, x) in canon:
        for (my, mx) in [(y, x), (y, 18 - x), (18 - y, x), (18 - y, 18 - x)]:
            idx.append(my * 30 + mx)
    return np.array(idx, np.int64)        # 32 = 8 groups x 4 mirrors


def build(task):
    inits = []
    nodes = []
    vinfos = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # 1. color-index plane: Conv with per-channel weight [0,1,...,9].
    ramp = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("Wramp", ramp, np.float32)
    n("Conv", ["input", "Wramp"], "cidxf")               # [1,1,30,30] f32
    vi("cidxf", TensorProto.FLOAT, [1, 1, 30, 30])
    n("Cast", ["cidxf"], "cidx8", to=TensorProto.INT8)   # [1,1,30,30] i8 (900B)
    vi("cidx8", TensorProto.INT8, [1, 1, 30, 30])

    # 2. flatten (i8 -> 900B) + gather 32 mirror cells; cast to i32 for the
    #    per-group max (ReduceMax rejects int8 but the tensors are tiny here).
    init("flat900", np.array([900], np.int64), np.int64)
    n("Reshape", ["cidx8", "flat900"], "cflat")          # [900] i8
    vi("cflat", TensorProto.INT8, [900])
    init("mirIdx", _mirror_flat_indices(), np.int64)
    n("Gather", ["cflat", "mirIdx"], "g32")              # [32] i8
    vi("g32", TensorProto.INT8, [32])
    n("Cast", ["g32"], "g32i", to=TensorProto.INT32)     # [32] i32 (tiny)
    vi("g32i", TensorProto.INT32, [32])
    init("shp84", np.array([8, 4], np.int64), np.int64)
    n("Reshape", ["g32i", "shp84"], "g84")               # [8,4] i32
    vi("g84", TensorProto.INT32, [8, 4])
    n("ReduceMax", ["g84"], "gcol", axes=[1], keepdims=0)  # [8] i32
    vi("gcol", TensorProto.INT32, [8])

    # gcolor_vec[10] = concat([-1, 0], gcol)
    init("head", np.array([-1, 0], np.int32), np.int32)
    n("Concat", ["head", "gcol"], "gcvec", axis=0)       # [10] i32
    vi("gcvec", TensorProto.INT32, [10])

    # 3. map every canvas cell to its group's color (gather straight into the
    #    [1,1,30,30] shape via a pre-shaped groupid index tensor).
    init("groupid", _groupid_plane().reshape(1, 1, 30, 30), np.int32)
    n("Gather", ["gcvec", "groupid"], "outidx")          # [1,1,30,30] i32
    vi("outidx", TensorProto.INT32, [1, 1, 30, 30])

    # 4. one-hot via Equal against [0..9] ramp (broadcast over channels). The
    #    -1 sentinel (outside grid) matches no channel -> stays all-zero.
    init("rampch", np.arange(10, dtype=np.int32).reshape(1, 10, 1, 1),
         np.int32)
    n("Equal", ["outidx", "rampch"], "eq")               # [1,10,30,30] bool
    vi("eq", TensorProto.BOOL, [1, 10, 30, 30])
    n("Cast", ["eq"], "output", to=TensorProto.FLOAT)    # [1,10,30,30] f32

    return _model(nodes, inits, vinfos)
