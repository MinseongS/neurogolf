"""Task 063: green-fill empty interior rows/columns of a bordered square grid.

True rule (ARC-GEN 2bee17df, invariant to the generator's final flip/xpose
since both grid and output are transformed identically):

  The input is a square grid whose perimeter ring is fully colored (red=2 /
  cyan=8) and whose interior contains scattered colored cells. For every
  interior row that is entirely background, the whole interior of that row is
  painted green=3; likewise for every entirely-background interior column.

Compact characterization used here (verified on all 266 stored examples and
fresh arc-gen instances): let occ = colored-cell indicator (1 - channel0).
  rowcount[r] = sum_c occ ,  colcount[c] = sum_r occ .
Because the two perimeter endpoints always contribute exactly 2 colored cells
to any in-grid row/column, an interior row/column is "all background" iff its
count == 2. The grid region is (rowcount>0) & (colcount>0) (perimeter is full,
off-grid canvas is background). A background cell is painted green iff it is
in-grid and (rowcount==2 or colcount==2).

Graph: occ -> row/col reductions -> bool masks -> fill[1,1,30,30] ->
delta(-1 on ch0, +1 on ch3) (x) fill -> Add(input, .) -> output.
One [1,10,30,30] float intermediate; everything else 1-D or bool.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits = []
    nodes = []
    vinfos = []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # --- occupancy: occ = sum of color channels 1..9  (1 where colored, else 0).
    # NB: off-grid canvas cells are all-zero across EVERY channel (incl. ch0),
    # so summing channels 1..9 correctly marks them background; using 1-ch0
    # would wrongly count off-grid cells as colored.
    Wocc = np.ones((1, 10, 1, 1), np.float32)
    Wocc[0, 0, 0, 0] = 0.0   # ignore background channel
    init("Wocc", Wocc)
    n("Conv", ["input", "Wocc"], "occ")  # [1,1,30,30]
    vi("occ", TensorProto.FLOAT, [1, 1, 30, 30])

    # --- row / col counts
    n("ReduceSum", ["occ"], "rc", axes=[3], keepdims=1)   # [1,1,30,1]
    vi("rc", TensorProto.FLOAT, [1, 1, 30, 1])
    n("ReduceSum", ["occ"], "cc", axes=[2], keepdims=1)   # [1,1,1,30]
    vi("cc", TensorProto.FLOAT, [1, 1, 1, 30])

    # counts are integers; compare directly in float (no Equal-on-float needed
    # because we threshold via two Greater comparisons combined).
    # rowfill: rc == 2  <=>  (rc > 1.5) & (rc < 2.5).  Counts are integers so
    # rc==2 <=> Greater(rc,1.5) & Less(rc,2.5).
    init("th15", np.array(1.5, np.float32))
    init("th25", np.array(2.5, np.float32))
    init("zero", np.array(0.0, np.float32))

    n("Greater", ["rc", "th15"], "rc_g1")        # rc>=2
    vi("rc_g1", TensorProto.BOOL, [1, 1, 30, 1])
    n("Less", ["rc", "th25"], "rc_l2")            # rc<=2
    vi("rc_l2", TensorProto.BOOL, [1, 1, 30, 1])
    n("And", ["rc_g1", "rc_l2"], "rc_eq2")        # rc==2
    vi("rc_eq2", TensorProto.BOOL, [1, 1, 30, 1])
    n("Greater", ["rc", "zero"], "rc_pos")        # row in grid
    vi("rc_pos", TensorProto.BOOL, [1, 1, 30, 1])

    n("Greater", ["cc", "th15"], "cc_g1")
    vi("cc_g1", TensorProto.BOOL, [1, 1, 1, 30])
    n("Less", ["cc", "th25"], "cc_l2")
    vi("cc_l2", TensorProto.BOOL, [1, 1, 1, 30])
    n("And", ["cc_g1", "cc_l2"], "cc_eq2")
    vi("cc_eq2", TensorProto.BOOL, [1, 1, 1, 30])
    n("Greater", ["cc", "zero"], "cc_pos")        # col in grid
    vi("cc_pos", TensorProto.BOOL, [1, 1, 1, 30])

    # in-grid mask (outer product of row/col positivity) -> [1,1,30,30]
    n("And", ["rc_pos", "cc_pos"], "ingrid")
    vi("ingrid", TensorProto.BOOL, [1, 1, 30, 30])
    # row-or-col empty -> [1,1,30,30]
    n("Or", ["rc_eq2", "cc_eq2"], "rceq")
    vi("rceq", TensorProto.BOOL, [1, 1, 30, 30])
    # candidate = ingrid & (rc==2 | cc==2)
    n("And", ["ingrid", "rceq"], "cand")
    vi("cand", TensorProto.BOOL, [1, 1, 30, 30])
    # background cells only: bg = (in0 == 1) i.e. occ == 0.  occ is 0/1 float.
    # bg bool from occ: occ<0.5
    init("th05", np.array(0.5, np.float32))
    n("Less", ["occ", "th05"], "bg")
    vi("bg", TensorProto.BOOL, [1, 1, 30, 30])
    # fill (bool) = cand & bg  -> [1,1,30,30]
    n("And", ["cand", "bg"], "fillb")
    vi("fillb", TensorProto.BOOL, [1, 1, 30, 30])

    # Final write into the free `output` tensor with a single Where. A fill cell
    # is always background, so we may overwrite the WHOLE channel stack there with
    # the green one-hot vector; the [1,1,30,30] bool condition broadcasts across
    # channels (900B), avoiding any 10-channel intermediate entirely.
    green = np.zeros((1, 10, 1, 1), np.float32)
    green[0, 3, 0, 0] = 1.0   # green = channel 3
    init("green", green)
    n("Where", ["fillb", "green", "input"], "output")

    return _model(nodes, inits, vinfos)
