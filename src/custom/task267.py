"""Task 267 (ARC-AGI aabf363d) — recolour the creature to the marker colour.

Rule (from the generator):
  * A contiguous "creature" of 12-15 pixels is drawn in colour `colors[0]`.
    `continuous_creature(.,5,5)` lives in a 5x5 box (coords 0..4), shifted +1,
    so the creature ALWAYS occupies rows 1..5, cols 1..5.
  * A single marker pixel `grid[6][0] = colors[1]` sits at row 6, col 0 in the
    INPUT only; it tells us the target colour.
  * OUTPUT = a fresh blank 7x7 grid (size=7) in which every creature cell is set
    to `colors[1]`.  The marker is NOT copied (output[6][0] is background); every
    other in-grid cell is background (channel-0 one-hot).  The harness pads the
    7x7 grid to 30x30 with the all-zero off-grid sentinel.

Encoding (key: the marker pixel IS the colors[1] one-hot, and the creature only
ever lives in the inner 5x5 block, so only that block needs runtime data):
  * marker one-hot = input[:, :, 6, 0] > 0  -> [1,10,1,1] BOOL (the target colour).
  * creature mask on the inner 5x5 = (ch0 == 0)  (background channel is 0 only on
    occupied creature cells inside the inner block).  Pad it back to the 7x7 grid
    (begins=[1,1]) -> the surrounding ring stays False = background.
  * output block = Where(mask7, marker_onehot, bg_onehot) where bg_onehot is the
    constant channel-0 one-hot.  -> [1,10,7,7] BOOL (the lone 490B plane).
  * Pad to 30x30 (off-grid all-zero) into the FREE output.

This avoids the full 7x7 fp32 background slice (only the inner 5x5 = 100B is read)
and avoids the colour-index multiply/Equal chain entirely; the marker one-hot
feeds Where directly.  Dominant intermediate: the [1,10,7,7] bool one-hot (490B),
irreducible because the 7x7 grid has in-grid background cells (all 10 channels
present) and routing a single-channel index plane through Pad to 30x30 is dearer.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
B = TensorProto.BOOL
U8 = TensorProto.UINT8


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- marker one-hot: input[:, :, 6, 0] > 0  -> [1,10,1,1] BOOL ----
    init("m_st", np.array([6, 0], np.int64), np.int64)
    init("m_en", np.array([7, 1], np.int64), np.int64)
    init("axes_hw", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["input", "m_st", "m_en", "axes_hw"], "marker")  # [1,10,1,1] f32
    init("zero", np.array(0.0, np.float32), np.float32)
    n("Greater", ["marker", "zero"], "marker_b")            # [1,10,1,1] BOOL
    n("Cast", ["marker_b"], "marker_oh", to=U8)             # [1,10,1,1] uint8

    # ---- creature mask on the inner 5x5 block (rows1..5, cols1..5) ----
    # background channel ch0 == 0 exactly on occupied creature cells.
    init("c0_st", np.array([0, 1, 1], np.int64), np.int64)
    init("c0_en", np.array([1, 6, 6], np.int64), np.int64)
    init("axes_chw", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "c0_st", "c0_en", "axes_chw"], "ch0in")  # [1,1,5,5] f32
    n("Equal", ["ch0in", "zero"], "mask5")                  # [1,1,5,5] BOOL
    # pad the 5x5 mask back to the 7x7 grid position; ring stays False (background)
    init("mpads", np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), np.int64)
    n("Pad", ["mask5", "mpads"], "mask7", mode="constant")  # [1,1,7,7] BOOL

    # ---- compose: creature -> marker colour, else in-grid background ----
    bg = np.zeros((1, 10, 1, 1), np.uint8); bg[0, 0, 0, 0] = 1
    init("bg_oh", bg, np.uint8)
    n("Where", ["mask7", "marker_oh", "bg_oh"], "block")    # [1,10,7,7] uint8 (490B)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 23, 23], np.int64), np.int64)
    n("Pad", ["block", "pads"], "output", mode="constant")  # [1,10,30,30] uint8

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task267", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 13)])
