"""Task 240 (ARC-AGI 9d9215db): 4-fold mirror a diagonal color bitmap and draw
dotted-square rings between diagonal-adjacent colors.

Rule (verified exact on all 266 stored examples, derived from the ARC-GEN
generator). The 19x19 input holds a small (length 2-4) diagonal of colored
pixels at odd coordinates (2r+1, 2c+1), optionally with a "nextdoor" color one
step along the diagonal, and the whole grid may be h/v-flipped. The output is
fully determined, per chebyshev ring d in {1,3,5,7} (distance from the border),
by exactly two colors:

  * Cd = the diagonal-corner color of ring d  (painted on the 4 ring corners)
  * Kd = the nextdoor/ring color of ring d    (painted on all non-corner odd
          cells of ring d as a dotted square; 0 => those cells stay background)

Because the flips and the mirroring are 4-fold symmetric, Cd and Kd are read off
the unflipped-canonical positions by folding (max over the 4 mirror cells).
Canonical positions: corner d at (d,d); ring d at (d,d+2).

Floor-break re-encoding (mem 6670 -> ~2900, params 957 -> ~380):
  * The color-index plane is produced by a STRIDE-2 Conv (kernel 1x1 = sum_k
    k*input_k) with pads top/left=1, so the only intermediate full plane is a
    16x16 fp32 plane (1024B) covering ONLY the odd input coords -- the cells
    that can ever hold a colour -- instead of the full 30x30 (3600B). out[i,j]
    = input[2i-1, 2j-1].
  * The 30x30 carrier is rebuilt SEPARABLY from a 16x16 odd-cell colour plane by
    two 1-D Gathers (row then col), each driven by a 30-int index map (60 params
    total) instead of a 900-int 30x30 group plane.
  * `output` is BOOL (opset 11) so the final Equal writes straight into the free
    output -- no materialised [1,10,30,30] one-hot.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

# The strided color plane is 16x16: planecell (iy,jx) holds input[2*iy-1, 2*jx-1]
# (iy,jx = 0 reads padded zeros). A canonical odd coord y maps to iy = (y+1)//2.
_RING_INDEX = {1: 0, 3: 1, 5: 2, 7: 3}


def _group_of(y, x):
    """Group id of original-coord cell (y, x): 0=outside/even-bg sentinel,
    1=in-grid background, 2..5=corner rings d=1,3,5,7, 6..9=dotted rings."""
    if y < 0 or x < 0 or y > 18 or x > 18:
        return 0
    if not (y % 2 == 1 and x % 2 == 1):
        return 0                       # even cells are always background
    ymin = min(y, 18 - y)
    xmin = min(x, 18 - x)
    ring = min(ymin, xmin)
    if ring in _RING_INDEX:
        ri = _RING_INDEX[ring]
        return (2 + ri) if ymin == xmin else (6 + ri)
    return 0                            # in-grid odd cell not on a ring -> bg


def _subgroupid_plane():
    """11x11 group-id plane indexed by (iy,jx), iy,jx in 0..10:
       0..9 -> original coord (2*iy-1, 2*jx-1)  (iy=0 => coord -1 = background);
       10   -> OUT-OF-GRID sentinel (group id 10 -> sentinel color 10).
    Even/non-ring/outside cells -> group 0 (background color 0)."""
    g = np.zeros((11, 11), np.int64)
    for iy in range(11):
        for jx in range(11):
            if iy == 10 or jx == 10:
                g[iy, jx] = 10                       # out-of-grid sentinel
            else:
                g[iy, jx] = _group_of(2 * iy - 1, 2 * jx - 1)
    return g


def _mirror_flat_indices():
    """Flat indices into the 256-cell (16x16) strided plane for the 8 canonical
    groups x 4 mirror cells = 32."""
    def pidx(y, x):
        return ((y + 1) // 2) * 16 + ((x + 1) // 2)
    canon = [(d, d) for d in (1, 3, 5, 7)] + [(d, d + 2) for d in (1, 3, 5, 7)]
    idx = []
    for (y, x) in canon:
        for (my, mx) in [(y, x), (y, 18 - x), (18 - y, x), (18 - y, 18 - x)]:
            idx.append(pidx(my, mx))
    return np.array(idx, np.int64)        # 32 = 8 groups x 4 mirrors


def _rowmap():
    """30-int map: original coord Y -> index into the 11-row subcol plane.
       Y >= 19 (off the 19x19 grid)  -> 10 (out-of-grid sentinel row);
       even in-grid Y                -> 0  (coord -1 == background row);
       odd  in-grid Y (1..17)        -> (Y+1)//2  (1..9)."""
    m = np.zeros(30, np.int64)
    for Y in range(30):
        if Y >= 19:
            m[Y] = 10
        elif Y % 2 == 0:
            m[Y] = 0
        else:
            m[Y] = (Y + 1) // 2
    return m


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

    # 1. STRIDE-2 colour-index plane: Conv kernel 1x1 = sum_k k*input_k, stride 2,
    #    pads top/left=1 so out[i,j] = input[2i-1, 2j-1] -> a 16x16 fp32 plane
    #    (1024B) covering only the odd input coords (the only cells that can hold
    #    a colour).  Far cheaper than the full 30x30 (3600B) plane.
    ramp = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("Wramp", ramp, np.float32)
    n("Conv", ["input", "Wramp"], "cidxf",
      strides=[2, 2], pads=[1, 1, 0, 0])                  # [1,1,16,16] f32 1024B
    n("Cast", ["cidxf"], "cidx8", to=TensorProto.UINT8)   # [1,1,16,16] u8 256B

    # 2. flatten + gather the 32 mirror cells; cast to i32 for the per-group max.
    init("flat256", np.array([256], np.int64), np.int64)
    n("Reshape", ["cidx8", "flat256"], "cflat")           # [256] u8
    init("mirIdx", _mirror_flat_indices(), np.int64)
    n("Gather", ["cflat", "mirIdx"], "g32")               # [32] u8
    n("Cast", ["g32"], "g32i", to=TensorProto.INT32)      # [32] i32
    init("shp84", np.array([8, 4], np.int64), np.int64)
    n("Reshape", ["g32i", "shp84"], "g84")                # [8,4] i32
    n("ReduceMax", ["g84"], "gcol", axes=[1], keepdims=0)  # [8] i32

    # colour vector [11]: index 0,1 = background (color 0), indices 2..9 = the 8
    #   group colors, index 10 = out-of-grid sentinel colour 10 (matches no
    #   channel in the final Equal -> off-grid cells stay all-zero).
    init("head", np.array([0, 0], np.int32), np.int32)
    init("tail", np.array([10], np.int32), np.int32)
    n("Concat", ["head", "gcol", "tail"], "gcvec", axis=0)  # [11] i32
    n("Cast", ["gcvec"], "gcvec8", to=TensorProto.UINT8)   # [11] u8

    # 3. 11x11 group-color plane (shaped [1,1,11,11]): gather colour vec by gid.
    init("subgid", _subgroupid_plane().reshape(1, 1, 11, 11), np.int64)
    n("Gather", ["gcvec8", "subgid"], "subcol")           # [1,1,11,11] u8 121B

    # 4. rebuild the [1,1,30,30] carrier SEPARABLY via chained 1-D Gathers on the
    #    spatial axes (row then col).  Because we gather directly on axes 2 and 3
    #    the result is already [1,1,30,30] -- no separate reshape plane needed.
    #    rowmap maps each original coord to its subcol index (even in-grid -> the
    #    background row 0; off-grid -> the sentinel index 10).
    init("rowmap", _rowmap(), np.int64)                    # [30] i64 (row==col map)
    n("Gather", ["subcol", "rowmap"], "byrow", axis=2)    # [1,1,30,11] u8 330B
    n("Gather", ["byrow", "rowmap"], "outidx", axis=3)    # [1,1,30,30] u8 900B

    # 5. one-hot via Equal vs the [0..9] channel ramp -- straight into the free
    #    BOOL output (sentinel colour 10 matches no channel -> off-grid all-zero).
    init("rampch", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["outidx", "rampch"], "output")            # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task240", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
