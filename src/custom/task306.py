"""Task 306 (ARC-AGI c444b776) — tile a single filled quadrant across the grid.

Rule (from the generator):
  * The canvas is a `width x height` array of 9x9 quadrants separated by single
    yellow(4) gridlines.  height is ALWAYS 2 and size is ALWAYS 9, so the grid
    is 19 rows tall and width*10-1 wide with width in {1,2,3} -> W in {9,19,29}.
    Vertical gridlines sit at c in {9,19} (only when width>=2); the horizontal
    gridline at row 9 ALWAYS exists and is yellow across every in-grid column.
  * The INPUT has the (up to 10) coloured pixels in exactly ONE quadrant; every
    other quadrant is empty.  The colours are drawn excluding yellow, so the
    content pixels are in {1,2,3,5,6,7,8,9} (never 0=bg, never 4=yellow).
  * The OUTPUT stamps that same 9x9 pattern into EVERY quadrant (the gridlines
    are unchanged).

So the output colour at content cell (r,c) is the single colour that appears at
local position (r%10, c%10) across the quadrants, or background if none.

Construction (label map + final Equal, opset 11, BOOL output):
  * label grid Lin[1,1,30,30] = sum_k k*input[k]  (background 0, colours 1..9,
    yellow 4).  This is the one fp32 30x30 plane (3600 B, the floor).
  * FOLD over quadrants by Max of the content row-bands (rows 0..8 / 10..18) and
    content col-bands (cols 0..8 / 10..18 / 20..28) -> patt[1,1,9,9] of colours.
    Gridline rows/cols are excluded from the bands, so only colours/bg fold in;
    Max recovers the colour because exactly one quadrant is filled (>0 wins).
  * UNFOLD: Gather patt with a [19] row index map and a [29] col index map to
    replicate the 9x9 pattern into every quadrant -> content[1,1,19,29].
  * Overlay yellow(4) on the gridline cells (r==9 or c in {9,19}); mask off-grid
    columns to a >=10 sentinel using the recovered in-grid column mask
    (input[4,9,c]); rows are always in-grid (H=19).
  * Pad to 30x30 with the sentinel and Equal(L, arange[1,10,1,1]) -> BOOL output.

Memory floor: the lone 30x30 plane is Lin (fp32 = 3600 B); the unfolded content
plane is [1,1,19,29] (fp32 551 -> uint8 sentinel work small); everything else is
1-D / 9x9 tiny.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- label grid Lin[1,1,30,30] = sum_k k * input[k] via a 1x1 Conv ----
    # A Conv with a [1,10,1,1] weight = arange(10) collapses the one-hot to the
    # colour label in ONE op (no [1,10,30,30] intermediate).
    W1 = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("W1", W1, np.float32)
    n("Conv", ["input", "W1"], "Lin")                  # [1,1,30,30] fp32 (3600)

    # ---- fold over quadrants: Max of the up-to-6 content blocks ----
    # ORT Max rejects uint8, so the fold stays fp32 on tiny 9x9 blocks; patt is
    # cast to uint8 afterwards (everything downstream is then 1-byte).
    # block(h,w) = Lin[h*10:h*10+9, w*10:w*10+9] (skips gridline row 9 / cols 9,19).
    # Off-grid blocks (width<3) are all-zero, harmless under Max.
    init("ax4", np.array([0, 1, 2, 3], np.int64), np.int64)
    blocks = []
    for h in range(2):
        for w in range(3):
            r0, c0 = h * 10, w * 10
            sname, ename = f"bs{h}{w}", f"be{h}{w}"
            init(sname, np.array([0, 0, r0, c0], np.int64), np.int64)
            init(ename, np.array([1, 1, r0 + 9, c0 + 9], np.int64), np.int64)
            blk = f"blk{h}{w}"
            n("Slice", ["Lin", sname, ename, "ax4"], blk)  # [1,1,9,9] fp32
            blocks.append(blk)
    # variadic Max folds all 6 blocks in ONE op (no intermediate accumulators)
    nodes.append(helper.make_node("Max", blocks, ["pattf"]))   # [1,1,9,9] fp32
    n("Cast", ["pattf"], "patt", to=U8)                        # [1,1,9,9] uint8

    # ---- unfold: gather patt to replicate the 9x9 pattern over all quadrants ----
    # row index map [19]: local row = r%10 for content rows; row 9 -> dummy 0
    # (overwritten by the gridline overlay).
    row_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0,
                        0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    # col index map [29]: local col = c%10 for content cols; cols 9,19 -> dummy 0.
    col_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0,
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 0,
                        0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    init("row_idx", row_idx, np.int64)
    init("col_idx", col_idx, np.int64)
    n("Gather", ["patt", "row_idx"], "gr", axis=2)     # [1,1,19,9] uint8
    n("Gather", ["gr", "col_idx"], "content", axis=3)  # [1,1,19,29] uint8

    # ---- overlay yellow(4) on gridline cells ----
    # gridline mask [1,1,19,29] bool: r==9 or c in {9,19}
    gl = np.zeros((1, 1, 19, 29), dtype=bool)
    gl[0, 0, 9, :] = True
    gl[0, 0, :, 9] = True
    gl[0, 0, :, 19] = True
    init("glmask", gl, np.bool_)
    init("yel", np.array(4, np.uint8), np.uint8)
    n("Where", ["glmask", "yel", "content"], "Lg")     # [1,1,19,29] uint8

    # ---- mask off-grid columns to sentinel 10 ----
    # in-grid columns: input[4, 9, c] == yellow (the always-present row-9 line).
    # Slice channel 4, row 9, cols 0..28 -> [1,1,1,29]; >0.5 => in-grid.
    init("ms", np.array([4, 9, 0], np.int64), np.int64)
    init("me", np.array([5, 10, 29], np.int64), np.int64)
    init("ax013", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "ms", "me", "ax013"], "colline")  # [1,1,1,29] fp32
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["colline", "half"], "ingrid")        # [1,1,1,29] bool
    init("sent", np.array(10, np.uint8), np.uint8)
    n("Where", ["ingrid", "Lg", "sent"], "Lu")         # [1,1,19,29] uint8

    # ---- pad to 30x30 with sentinel, final Equal ----
    init("u10", np.array(10, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 11, 1], np.int64), np.int64)
    n("Pad", ["Lu", "pads", "u10"], "L", mode="constant")  # [1,1,30,30] uint8

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")                # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task306", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
