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

Floor-break re-encoding (old mem 18360 -> ~5400):
  * `output` is BOOL (opset 11) so the final Equal writes straight into the free
    output -- no materialised [1,10,30,30] one-hot (saved 9000B).
  * the per-cell label plane is uint8 with sentinel 10 for outside-grid cells
    (Equal vs the [0..9] channel ramp never matches -> all-false there), so it
    costs 900B instead of an int32 3600B plane.
  * tiny [8,4]/[10] color math stays int32 (ReduceMax rejects int8).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def _groupid_plane():
    gid = np.zeros((30, 30), np.int64)
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

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # 1. color-index plane: Conv with per-channel weight [0,1,...,9], cast
    #    straight to uint8 (900B).  Only 32 cells are read from it downstream,
    #    but the Conv collapses the 10 channels so its output is a single 30x30
    #    plane (3600B f32) -- far cheaper than reshaping the full input.
    ramp = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("Wramp", ramp, np.float32)
    n("Conv", ["input", "Wramp"], "cidxf")               # [1,1,30,30] f32
    n("Cast", ["cidxf"], "cidx8", to=TensorProto.UINT8)  # [1,1,30,30] u8 (900B)

    # 2. flatten + gather 32 mirror cells; cast to i32 for the per-group max
    #    (ReduceMax rejects uint8 but the gathered tensors are tiny).
    init("flat900", np.array([900], np.int64), np.int64)
    n("Reshape", ["cidx8", "flat900"], "cflat")          # [900] u8
    init("mirIdx", _mirror_flat_indices(), np.int64)
    n("Gather", ["cflat", "mirIdx"], "g32")              # [32] u8
    n("Cast", ["g32"], "g32i", to=TensorProto.INT32)     # [32] i32 (tiny)
    init("shp84", np.array([8, 4], np.int64), np.int64)
    n("Reshape", ["g32i", "shp84"], "g84")               # [8,4] i32
    n("ReduceMax", ["g84"], "gcol", axes=[1], keepdims=0)  # [8] i32

    # gcolor_vec[10] = concat([10, 0], gcol) ; index 0 = outside sentinel 10.
    init("head", np.array([10, 0], np.int32), np.int32)
    n("Concat", ["head", "gcol"], "gcvec", axis=0)       # [10] i32
    n("Cast", ["gcvec"], "gcvec8", to=TensorProto.UINT8)  # [10] u8

    # 3. map every canvas cell to its group's color (gather into [1,1,30,30]).
    init("groupid", _groupid_plane().reshape(1, 1, 30, 30), np.int64)
    n("Gather", ["gcvec8", "groupid"], "outidx")         # [1,1,30,30] u8 (900B)

    # 4. one-hot via Equal vs the [0..9] channel ramp, straight into free BOOL
    #    output. Sentinel 10 (outside grid) matches no channel -> all-zero.
    init("rampch", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["outidx", "rampch"], "output")           # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task240", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
