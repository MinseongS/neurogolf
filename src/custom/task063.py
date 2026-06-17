"""Task 063: green-fill empty interior rows/columns of a bordered square grid.

True rule (ARC-GEN 2bee17df): a square size×size grid (size ∈ {10,12,14}) whose
perimeter ring is fully coloured (red=2 / cyan=8) by inward-pointing teeth, with
scattered coloured interior cells. For every INTERIOR row r (1..size-2) whose
interior cells (cols 1..size-2) are entirely background, the whole interior of
that row is painted green=3; likewise for every interior column whose interior
is entirely background. (flip / transpose are applied afterwards but preserve
the row/column structure.)

Compact characterisation (no per-cell occupancy plane needed):

  * Per-row coloured count rc[r] = #cells in row r with colour in channels 1..9.
    Because the two perimeter endpoints (cols 0 and size-1) are always coloured,
    an interior row is "all-background-interior" iff rc[r] == 2. Same for cols.
  * A cell is filled green iff it is INTERIOR on both axes AND (its row is free
    OR its column is free):  fill = interior_r ∧ interior_c ∧ (rowfree ∨ colfree).
    The Or already implies the cell is background (a free row has an all-bg
    interior, ditto a free col), so NO separate background mask is required.

Memory: the only [1,1,30,30] tensors are three bool masks (900B each) plus the
FREE output; everything else is a 120B row/col profile vector. The previous
version paid a 3600B fp32 ch0 occupancy plane — removed here by deriving the
coloured counts directly with two channel-weighted no-pad Convs (ch0 weight 0)
and the in-grid / interior masks from 1-D neighbour Convs on those profiles.
"""

import numpy as np
from onnx import helper, numpy_helper

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

    # ---- per-row coloured count rc[r] in ONE no-pad Conv ----
    # Wrow[1,10,1,30]: weight 1 on channels 1..9 (colours), 0 on channel 0 (bg).
    # Conv(input, Wrow) -> [1,1,30,1]; each output = sum over cols of coloured.
    Wrow = np.ones([1, 10, 1, 30], np.float32)
    Wrow[0, 0] = 0.0
    init("Wrow", Wrow)
    n("Conv", ["input", "Wrow"], "rc")        # [1,1,30,1] f32 = 120B

    # ---- per-col coloured count cc[c] in ONE no-pad Conv ----
    Wcol = np.ones([1, 10, 30, 1], np.float32)
    Wcol[0, 0] = 0.0
    init("Wcol", Wcol)
    n("Conv", ["input", "Wcol"], "cc")        # [1,1,1,30] f32 = 120B

    # ---- free row/col: coloured count == 2 (only the two perimeter endpoints).
    # In-grid rows always have rc >= 2 (the two perimeter endpoints), so
    # rc < 2.5 <=> rc == 2 <=> free.  (off-grid rows have rc == 0 but are removed
    # by the interior mask below, so a single Less suffices — opset-10 Equal
    # does not accept float operands.)
    init("c25", np.array(2.5, np.float32))
    n("Less", ["rc", "c25"], "rowfree")       # [1,1,30,1] bool
    n("Less", ["cc", "c25"], "colfree")       # [1,1,1,30] bool

    # ---- in-grid indicator (count > 0) as float for neighbour Conv ----
    init("zero", np.array(0.0, np.float32))
    n("Greater", ["rc", "zero"], "rany_b")    # [1,1,30,1] bool
    n("Greater", ["cc", "zero"], "cany_b")    # [1,1,1,30] bool
    n("Cast", ["rany_b"], "rany_f", to=1)     # f32 0/1  [1,1,30,1]
    n("Cast", ["cany_b"], "cany_f", to=1)     # f32 0/1  [1,1,1,30]

    # ---- interior: a row is interior iff it AND both neighbours are in-grid;
    # neighbour-sum via a length-3 same-pad Conv on the 1-D profile == 3, i.e.
    # > 2.5 (only the true interior rows reach 3). ----
    Krow = np.ones([1, 1, 3, 1], np.float32)
    init("Krow", Krow)
    n("Conv", ["rany_f", "Krow"], "rnb", pads=[1, 0, 1, 0])  # [1,1,30,1]
    Kcol = np.ones([1, 1, 1, 3], np.float32)
    init("Kcol", Kcol)
    n("Conv", ["cany_f", "Kcol"], "cnb", pads=[0, 1, 0, 1])  # [1,1,1,30]
    n("Greater", ["rnb", "c25"], "intr")      # [1,1,30,1] bool (interior row)
    n("Greater", ["cnb", "c25"], "intc")      # [1,1,1,30] bool (interior col)

    # ---- fill = interior_r ∧ interior_c ∧ (rowfree ∨ colfree) ----
    n("Or", ["rowfree", "colfree"], "freeor")  # [1,1,30,30] bool = 900B
    n("And", ["intr", "intc"], "intr2")        # [1,1,30,30] bool = 900B
    n("And", ["freeor", "intr2"], "fillb")     # [1,1,30,30] bool = 900B

    # ---- paint green (channel 3); keep input elsewhere ----
    green = np.zeros([1, 10, 1, 1], np.float32)
    green[0, 3, 0, 0] = 1.0
    init("green", green)
    n("Where", ["fillb", "green", "input"], "output")

    return _model(nodes, inits)
