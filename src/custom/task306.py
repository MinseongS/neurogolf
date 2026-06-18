"""Task 306 (ARC-AGI c444b776) — tile a single filled quadrant across the grid.

Rule (from the generator):
  * The canvas is a `width x height` array of 9x9 quadrants separated by single
    yellow(4) gridlines.  height is ALWAYS 2 and size is ALWAYS 9, so the grid
    is 19 rows tall and width*10-1 wide with width in {1,2,3} -> W in {9,19,29}.
    Vertical gridlines sit at c in {9,19} (only when width>=2); the horizontal
    gridline at row 9 ALWAYS exists, yellow across every in-grid column.
  * The INPUT has the (up to 10) coloured pixels in exactly ONE quadrant; every
    other quadrant is empty.  Colours exclude yellow, so content pixels are in
    {1,2,3,5,6,7,8,9} (never 0=bg, never 4=yellow).
  * The OUTPUT stamps that same 9x9 pattern into EVERY quadrant (gridlines kept).

So output colour at content cell (r,c) = the single colour at local position
(r%10, c%10) across the quadrants, or background if none.

Escape-1 SPATIAL-COPY (fixed lattice remap):
  1. FOLD the one-hot quadrants in ONE op: a DEPTHWISE Conv with a 3x3 all-ones
     kernel DILATED by 10 sums input[k, lr+10i, lc+10j] over the 3x3 quadrant grid
     -> foldoh[1,10,10,10].  Channels 0 (bg) and 4 (yellow) get weight 0 so only
     the eight CONTENT colours fold; exactly one quadrant is filled so the sum
     equals the donor's one-hot (no double-count).  (bg/yellow are double-summed
     across quadrants, hence excluded and reconstructed in the table below.)
  2. COLLAPSE to a value plane: 1x1 Conv (sum_k k) -> patt[1,1,10,10]; local rows/
     cols 0..8 = donor colour (0 = bg).
  3. Build an 11x11 lookup table from patt[:, :, :9, :9]: index 9 = gridline lane
     (value 4), index 10 = off-grid sentinel (99), via two Pads.
  4. UNFOLD by a 2-D Gather (col map then row map [30]) replicating the 9x9 pattern
     into every quadrant AND overlaying gridlines from the same table.  Rows 19..29
     map to the off-grid index; off-grid COLUMNS (width-dependent) are masked to the
     sentinel via the always-present row-9 gridline presence input[4,9,c] BEFORE the
     row-gather so no full extra carrier plane is needed.
  5. Equal(L, arange_ch[1,10,1,1]) -> BOOL output (10-ch expansion lands in the
     FREE output; 4 -> yellow, 99 -> nothing -> off-grid all-zero one-hot).

Dominant intermediate: the uint8 30x30 carrier L (900 B) feeding the final Equal
(an Equal index-feeder is pinned at one full 30x30 plane); the fold+collapse is a
single 10x10 fp32 Conv plane (400 B) and the 10-ch output expansion is FREE.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
I64 = TensorProto.INT64
U8 = TensorProto.UINT8
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins if isinstance(ins, list) else [ins],
                                      [out], **attrs))
        return out

    # ---- 1. FUSED dilated fold + channel-collapse in ONE Conv ----
    # A 3x3 kernel DILATED by 10 sums input[k, lr+10i, lc+10j] over the 3x3
    # quadrant grid; weight[0,k,i,j] = k (the colour value) for the 8 content
    # channels, 0 for bg(0) and yellow(4).  Exactly one quadrant is filled so the
    # sum collapses directly to the donor colour value -> patt[1,1,10,10].
    # (bg/yellow are double-summed across quadrants, hence excluded and
    # reconstructed in the table below.)
    Wf = np.zeros((1, 10, 3, 3), dtype=np.float32)
    for k in range(10):
        if k not in (0, 4):
            Wf[0, k, :, :] = float(k)
    init("Wf", Wf, np.float32)
    n("Conv", ["input", "Wf"], "patt", dilations=[10, 10],
      pads=[0, 0, 0, 0])                                # [1,1,10,10] fp32 (400)

    # ---- 3. build 11x11 lookup table ----
    n("Cast", ["patt"], "pu8", to=U8)                  # [1,1,10,10] uint8
    init("ss", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("se", np.array([1, 1, 9, 9], np.int64), np.int64)
    init("ax4", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["pu8", "ss", "se", "ax4"], "p9")       # [1,1,9,9] uint8
    init("yel", np.array(4, np.uint8), np.uint8)
    init("padgl", np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64), np.int64)
    n("Pad", ["p9", "padgl", "yel"], "tbl0", mode="constant")  # [1,1,10,10] u8
    init("sent", np.array(99, np.uint8), np.uint8)
    init("padog", np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64), np.int64)
    n("Pad", ["tbl0", "padog", "sent"], "tbl", mode="constant")  # [1,1,11,11] u8

    # ---- 4. unfold via 2-D Gather (cols first, mask off-grid cols, then rows) ----
    col_idx = [c % 10 for c in range(30)]              # gridline cols -> local 9
    row_idx = [(r % 10 if r < 19 else 10) for r in range(30)]  # rows>=19 off-grid
    init("col_idx", np.array(col_idx, np.int64), np.int64)
    init("row_idx", np.array(row_idx, np.int64), np.int64)
    n("Gather", ["tbl", "col_idx"], "gc", axis=3)      # [1,1,11,30] uint8

    # mask off-grid columns on the SMALL gc plane (avoids a full extra carrier).
    # in-grid columns = where the always-present row-9 gridline is yellow:
    # input[4,9,c] == 1.  Slice channel 4, row 9 -> [1,1,1,30].
    init("ms", np.array([4, 9, 0], np.int64), np.int64)
    init("me", np.array([5, 10, 30], np.int64), np.int64)
    init("ax123", np.array([1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "ms", "me", "ax123"], "colline")  # [1,1,1,30] fp32
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["colline", "half"], "ingrid")        # [1,1,1,30] bool
    init("sent2", np.array(99, np.uint8), np.uint8)
    n("Where", ["ingrid", "gc", "sent2"], "gcm")       # [1,1,11,30] uint8 broadcast
    n("Gather", ["gcm", "row_idx"], "L", axis=2)       # [1,1,30,30] uint8

    # ---- 5. final Equal -> BOOL output ----
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task306", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
