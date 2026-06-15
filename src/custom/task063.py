"""Task 063: green-fill empty interior rows/columns of a bordered square grid.

True rule (ARC-GEN 2bee17df): The input is a square grid whose perimeter ring
is fully colored (red=2/cyan=8) and whose interior contains scattered colored
cells. For every interior row that is entirely background, the whole interior of
that row is painted green=3; likewise for every entirely-background interior
column.

Compact characterization: let rc_colored[r] = number of non-background cells in
row r (channels 1..9), and cc_colored[c] likewise for columns.  Because the
perimeter endpoints always contribute exactly 2 colored cells to any in-grid
row/column, an interior row/col is "all background" iff its colored count == 2.

Memory floor-break: avoid materialising the full [1,1,30,30] occ tensor (3600B
f32).  Instead:
  - Extract channel-0 of input via a 1×1 Conv → ch0_f [1,1,30,30] f32 = 3600B.
    Note: ch0[r,c] = 1 iff the cell is a background cell AND in-grid (off-grid
    cells have ALL channels 0, so ch0 = 0 there too).  This gives us:
      bg = (ch0_f > 0.5) = [1,1,30,30] bool 900B — naturally excludes off-grid
      cells, so we do NOT need a separate ingrid mask.
  - Row/col colored counts: rc_colored = rc_all - rc_ch0, where rc_all comes
    from ReduceSum(input, [1,3]) (free, 0 extra params) and rc_ch0 from
    ReduceSum(ch0_f, [3]) (reusing ch0_f).
  - rceq = Or(rowfill_1d, colfill_1d) broadcasts directly to [1,1,30,30] 900B.
  - fillb = And(rceq, bg) [1,1,30,30] bool 900B.
  - Where(fillb, green, input) → output.

Total [1,1,30,30] intermediates: ch0_f (3600B f32) + bg + rceq + fillb (3×900B
bool) = 6300B vs old 8580B (occ_f32 + 5 bool 30×30).  Params: ~23 (W_ch0,
green, thresholds) vs old 24.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # ---- channel-0 extractor: 1x1 Conv picking only ch0 ----
    # ch0_f[r,c] = 1.0 iff background (ch0=1); 0.0 for colored or off-grid.
    W_ch0 = np.zeros([1, 10, 1, 1], np.float32)
    W_ch0[0, 0, 0, 0] = 1.0
    init("W_ch0", W_ch0)
    n("Conv", ["input", "W_ch0"], "ch0_f")         # [1,1,30,30] f32 = 3600B

    # ---- bg mask from ch0 (also encodes ingrid: off-grid ch0==0 → bg False) ----
    init("half", np.array(0.5, np.float32))
    n("Greater", ["ch0_f", "half"], "bg")          # [1,1,30,30] bool = 900B

    # ---- row/col colored counts via ReduceSum (0 extra params) ----
    # rc_all[r] = total in-grid cells in row r (sum over all channels)
    # For any in-grid cell exactly one channel is 1 → sum = 1.
    # rc_ch0[r] = count of background cells in row r.
    # rc_colored[r] = rc_all[r] - rc_ch0[r].
    n("ReduceSum", ["input"], "rc_all", axes=[1, 3], keepdims=1)   # [1,1,30,1]
    n("ReduceSum", ["ch0_f"], "rc_ch0", axes=[3], keepdims=1)      # [1,1,30,1]
    n("Sub", ["rc_all", "rc_ch0"], "rc")                           # [1,1,30,1] f32
    n("ReduceSum", ["input"], "cc_all", axes=[1, 2], keepdims=1)   # [1,1,1,30]
    n("ReduceSum", ["ch0_f"], "cc_ch0", axes=[2], keepdims=1)      # [1,1,1,30]
    n("Sub", ["cc_all", "cc_ch0"], "cc")                           # [1,1,1,30] f32

    # ---- empty-row/col indicator: rc == 2 ↔ only perimeter endpoints colored ----
    init("th15", np.array(1.5, np.float32))
    init("th25", np.array(2.5, np.float32))
    n("Greater", ["rc", "th15"], "rc_g")      # [1,1,30,1] bool
    n("Less", ["rc", "th25"], "rc_l")         # [1,1,30,1] bool
    n("And", ["rc_g", "rc_l"], "rowfill")     # [1,1,30,1] bool (rc == 2)
    n("Greater", ["cc", "th15"], "cc_g")      # [1,1,1,30] bool
    n("Less", ["cc", "th25"], "cc_l")         # [1,1,1,30] bool
    n("And", ["cc_g", "cc_l"], "colfill")     # [1,1,1,30] bool (cc == 2)

    # ---- fill condition: broadcast 1D conditions to [1,1,30,30] ----
    # Or(rowfill[r], colfill[c]) broadcasts [1,1,30,1] OR [1,1,1,30] → [1,1,30,30]
    n("Or", ["rowfill", "colfill"], "rceq")   # [1,1,30,30] bool = 900B

    # ---- final fill mask: in-grid background cells in empty rows/cols ----
    n("And", ["rceq", "bg"], "fillb")         # [1,1,30,30] bool = 900B

    # ---- paint green (channel 3) on fill cells; keep input elsewhere ----
    green = np.zeros([1, 10, 1, 1], np.float32)
    green[0, 3, 0, 0] = 1.0
    init("green", green)
    n("Where", ["fillb", "green", "input"], "output")

    return _model(nodes, inits)
