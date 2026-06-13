"""Task 190 (7ddcd7ec): extend diagonal rays from a 2x2 box.

Rule (ARC-GEN generator): the input has a solid 2x2 box of one colour plus, for
each chosen diagonal direction, a single 'seed' cell placed diagonally adjacent
to the matching box corner.  The output extends every seed into a full diagonal
ray going outward (away from the box) to the grid edge; the box is preserved.

Every grid is 10x10 and fully populated, so the grid-occupancy mask is the
constant all-ones 10x10 and the coloured mask is just (1 - background channel).
We work entirely on the top-left 10x10 region (a 10x10 float plane is 400B vs
3600B on the 30x30 canvas, and the half-line ray conv self-truncates at the
10x10 edge so no clipping is needed), then Pad the 10x10 result up to the 30x30
`output` (free).

Masks ([1,1,10,10]):
  ch0  = Slice(input, channel 0, top-left 10x10)   (background)
  cm   = 1 - ch0                                    (coloured)
  box  = cells in a full 2x2 window
  seeds= cm - box
Directional rays: for diagonal d=(dr,dc) the seed of type d is a seed whose box
corner sits at (r-dr,c-dc).  A 1->4 conv shifts box into the 4 diagonal
neighbours; AND with seeds gives the 4 seed-type planes; a grouped 19x19
diagonal-half-line conv extends each into its ray; ReduceSum + threshold gives
A (the full coloured-cell mask).

Final assembly is one 1x1 Conv with runtime weights W[10,2,1,1] on
S=concat(ones, A): out[k]=ones*W[k,0]+A*W[k,1] with W[:,0]=e0, W[:,1]=colour-e0,
so channel 0 = 1-A (background), the colour channel = A, others 0.  Pad lifts
the 10x10 result onto the 30x30 `output`.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]   # NW NE SW SE

    init("c0_5", np.array(0.5, np.float32), np.float32)
    init("c1", np.array(1.0, np.float32), np.float32)

    # isolated-cell kernel: orth-neighbour sum minus 2*centre.
    #   value = (#orthogonal coloured neighbours) - 2*cm
    #   an isolated coloured cell (cm=1, 0 neighbours) gives -2 (< -1.5);
    #   box / non-coloured cells give >= -1.  -> seed = value < -1.5
    isoW = np.zeros((1, 1, 3, 3), np.float32)
    for (u, v) in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        isoW[0, 0, u, v] = 1.0
    isoW[0, 0, 1, 1] = -2.0
    init("isoW", isoW, np.float32)
    init("cm1_5", np.array(-1.5, np.float32), np.float32)

    # 1->4 shift kernel: out_d(r,c) = cm(r-dr, c-dc)
    shiftW = np.zeros((4, 1, 3, 3), np.float32)
    for i, (dr, dc) in enumerate(dirs):
        shiftW[i, 0, 1 - dr, 1 - dc] = 1.0
    init("shiftW", shiftW, np.float32)

    # ray kernel as a single out-channel conv summing the 4 directional
    # half-lines: rayW[0, d] is direction d's outward half-line.  The generator
    # draws row,col in [2, size-4]=[2,6], so a seed sits at offset 1..5 from an
    # edge: a ray beyond the seed is at most 5 cells, hence K=11 (offset 5).
    K, cc = 11, 5
    rayW = np.zeros((1, 4, K, K), np.float32)
    for i, (dr, dc) in enumerate(dirs):
        for k in range(cc + 1):
            rayW[0, i, cc - k * dr, cc - k * dc] = 1.0
    init("rayW", rayW, np.float32)

    init("e0", np.array([1] + [0] * 9, np.float32).reshape(10, 1, 1, 1), np.float32)
    init("bias0", np.array([1] + [0] * 9, np.float32), np.float32)       # conv bias -> ch0=1
    init("zero0", np.array([0] + [1] * 9, np.float32).reshape(1, 10, 1, 1), np.float32)
    init("shp10", np.array([10, 1, 1, 1], np.int64), np.int64)

    # combined channel+spatial slice -> background plane at 10x10
    init("b_s", np.array([0, 0, 0], np.int64), np.int64)
    init("b_e", np.array([1, 10, 10], np.int64), np.int64)
    init("b_a", np.array([1, 2, 3], np.int64), np.int64)

    # ---- masks ----
    n("Slice", ["input", "b_s", "b_e", "b_a"], "ch0")                    # [1,1,10,10]
    n("Sub", ["c1", "ch0"], "cm")                                        # coloured mask (1 - bg)

    # seeds = isolated coloured cells (no orthogonal coloured neighbour)
    n("Conv", ["cm", "isoW"], "isoscore", kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    n("Less", ["isoscore", "cm1_5"], "seed_b")
    n("Cast", ["seed_b"], "seeds", to=TensorProto.FLOAT)                 # isolated coloured

    n("Conv", ["cm", "shiftW"], "cmshift", kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    n("Mul", ["cmshift", "seeds"], "seedtype")                           # [1,4,10,10]
    # single out-channel conv: sums the 4 directional half-line rays at once
    n("Conv", ["seedtype", "rayW"], "raysum", kernel_shape=[K, K],
      pads=[cc, cc, cc, cc])                                             # [1,1,10,10]
    n("Add", ["cm", "raysum"], "Asum")
    n("Greater", ["Asum", "c0_5"], "A_b")
    n("Cast", ["A_b"], "A", to=TensorProto.FLOAT)                        # [1,1,10,10]

    # ---- colour vector -> 1x1 conv weight + bias ----
    # out[k] = A * (colour-onehot - e0)[k] + e0[k]
    #   channel 0      : A*(0-1) + 1   = 1 - A   (background)
    #   colour channel : A*(1-0) + 0   = A
    #   other channels : 0
    n("ReduceMax", ["input"], "cvec_raw", axes=[2, 3], keepdims=1)       # [1,10,1,1]
    n("Mul", ["cvec_raw", "zero0"], "cvecP")                             # zero channel 0
    n("Reshape", ["cvecP", "shp10"], "cvecR")                            # [10,1,1,1]
    n("Sub", ["cvecR", "e0"], "Wc")                                      # [10,1,1,1]

    n("Conv", ["A", "Wc", "bias0"], "out10", kernel_shape=[1, 1])        # [1,10,10,10]
    n("Pad", ["out10"], "output", mode="constant",
      pads=[0, 0, 0, 0, 0, 0, 20, 20], value=0.0)

    return _model(nodes, inits)
