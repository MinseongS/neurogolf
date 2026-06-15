"""Task 190 (7ddcd7ec): extend diagonal rays from a 2x2 box.

Rule (ARC-GEN generator): the input has a solid 2x2 box of one colour plus, for
each chosen diagonal direction, a single 'seed' cell placed diagonally adjacent
to the matching box corner.  The output extends every seed into a full diagonal
ray going outward (away from the box) to the grid edge; the box is preserved.

Every grid is 10x10 and fully populated, so the grid-occupancy mask is the
constant all-ones 10x10 and the coloured mask is just (1 - background channel).
We work entirely on the top-left 10x10 region.

Memory floor-break (label map + final Equal, fp16 direction channels):

  Old: Conv(A, Wc, bias0) -> out10 [1,10,10,10] f32 (4000B) then Pad -> output.
  New: Build A [1,1,10,10] bool (coloured mask for output), then:
       ki (uint8 scalar) = colour index from input
       L10 = Where(A, ki, 0) as uint8 [1,1,10,10] (100B)
       L30 = Pad(L10, sentinel 0, to 30x30) -> [1,1,30,30] uint8 (900B)
       output = Equal(L30, arange[1,10,1,1]) -> free BOOL output (opset 11)

  Additional: use fp16 for cmshift and seedtype [1,4,10,10] (1600B -> 800B each).

  Masks ([1,1,10,10]):
    ch0  = Slice(input, channel 0, top-left 10x10)   (background)
    cm   = 1 - ch0                                    (coloured)
  Seed detection (isolated coloured cell, no orth coloured neighbour):
    isoscore = Conv(cm_f16, isoW_f16) with orth kernel - 2*centre
    seeds = isoscore < -1.5  -> bool [1,1,10,10]
  Directional rays:
    cmshift = Conv(cm_f16, shiftW)  [1,4,10,10] fp16 (shifted box mask per direction)
    seedtype = cmshift * seeds_f16  [1,4,10,10] fp16 (seeds tagged with their direction)
    raysum = Conv(seedtype, rayW)   [1,1,10,10] f32  (ray coverage for all directions)
    A = (cm_f32 + raysum > 0.5) bool [1,1,10,10]

  Label map: L10 = Where(A_bool, ki, v0) uint8 [1,1,10,10]
  Pad L10 to 30x30 (zero fill, 0 = background = ch0 on all outside cells) then Equal.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


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
    init("c0_5_f16", np.array(0.5, np.float16), np.float16)

    # isolated-cell kernel (orth-neighbour sum minus 2*centre, fp16)
    isoW = np.zeros((1, 1, 3, 3), np.float16)
    for (u, v) in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        isoW[0, 0, u, v] = 1.0
    isoW[0, 0, 1, 1] = -2.0
    init("isoW", isoW, np.float16)
    init("cm1_5", np.array(-1.5, np.float32), np.float32)

    # 1->4 shift kernel (fp16): out_d(r,c) = cm(r-dr, c-dc)
    shiftW = np.zeros((4, 1, 3, 3), np.float16)
    for i, (dr, dc) in enumerate(dirs):
        shiftW[i, 0, 1 - dr, 1 - dc] = 1.0
    init("shiftW", shiftW, np.float16)

    # ray kernel (fp16): single output channel summing 4 directional half-lines
    K, cc = 11, 5
    rayW = np.zeros((1, 4, K, K), np.float16)
    for i, (dr, dc) in enumerate(dirs):
        for k in range(cc + 1):
            rayW[0, i, cc - k * dr, cc - k * dc] = 1.0
    init("rayW", rayW, np.float16)

    # channel slice to get background at [1,1,10,10]
    init("b_s", np.array([0, 0, 0], np.int64), np.int64)
    init("b_e", np.array([1, 10, 10], np.int64), np.int64)
    init("b_a", np.array([1, 2, 3], np.int64), np.int64)

    # ---- masks ----
    n("Slice", ["input", "b_s", "b_e", "b_a"], "ch0")              # [1,1,10,10] f32
    n("Sub", ["c1", "ch0"], "cm_f")                                 # coloured mask f32
    n("Cast", ["cm_f"], "cm_f16", to=TensorProto.FLOAT16)           # fp16

    # seeds = isolated coloured cells
    n("Conv", ["cm_f16", "isoW"], "isoscore_f16",
      kernel_shape=[3, 3], pads=[1, 1, 1, 1])                       # [1,1,10,10] fp16
    n("Cast", ["isoscore_f16"], "isoscore", to=TensorProto.FLOAT)   # f32
    n("Less", ["isoscore", "cm1_5"], "seed_b")                      # [1,1,10,10] bool
    n("Cast", ["seed_b"], "seeds_f16", to=TensorProto.FLOAT16)      # fp16

    # directional shift of cm -> [1,4,10,10] fp16; multiply by seeds -> seedtype fp16
    n("Conv", ["cm_f16", "shiftW"], "cmshift",
      kernel_shape=[3, 3], pads=[1, 1, 1, 1])                       # [1,4,10,10] fp16
    n("Mul", ["cmshift", "seeds_f16"], "seedtype")                  # [1,4,10,10] fp16

    # ray conv: [1,4,10,10] fp16 -> [1,1,10,10] fp16
    n("Conv", ["seedtype", "rayW"], "raysum_f16",
      kernel_shape=[K, K], pads=[cc, cc, cc, cc])                   # [1,1,10,10] fp16
    n("Cast", ["raysum_f16"], "raysum", to=TensorProto.FLOAT)       # f32

    n("Add", ["cm_f", "raysum"], "Asum")
    n("Greater", ["Asum", "c0_5"], "A_bool")                        # [1,1,10,10] bool

    # ---- colour index ki (uint8 scalar) from input --------------------------
    # The single colour k is the unique non-background channel present
    # ki = ReduceSum(presf * arange) where presf[c] = max(input[c]) > 0
    init("arange10", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1),
         np.float32)
    n("ReduceMax", ["input"], "presf", axes=[2, 3], keepdims=1)     # [1,10,1,1]
    n("Mul", ["presf", "arange10"], "kparts")
    n("ReduceSum", ["kparts"], "k_f")                               # scalar f32
    n("Cast", ["k_f"], "ki", to=TensorProto.UINT8)                  # uint8

    # ---- uint8 label map L [1,1,10,10] -> Pad to [1,1,30,30] -> Equal ------
    # Background cells (A=0) -> 0; coloured cells (A=1) -> ki
    init("v0", np.array(0, np.uint8), np.uint8)
    n("Where", ["A_bool", "ki", "v0"], "L10")                       # [1,1,10,10] u8

    # Pad to 30x30 with sentinel 10 (> any channel index 0..9, so Equal gives
    # all-False there = all-channels-off, which is correct for cells outside the
    # 10x10 active grid).
    init("padpads",
         np.array([0, 0, 0, 0, 0, 0, 20, 20], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L10", "padpads", "padval"], "L30", mode="constant")  # [1,1,30,30] u8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L30", "chan"], "output")                            # BOOL [1,10,30,30]

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task190", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
