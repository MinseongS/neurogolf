"""task252 (ARC-AGI a5f85a15) — recolor odd-column diagonal cells to yellow.

Rule (from the generator):
  A size x size grid (size in 3..15) sits at the top-left of the 30x30 canvas.
  A set of anti-diagonals {r-c in diags} is painted in a single `color`; the rest
  is black background (channel 0).  In the OUTPUT each colored cell KEEPS `color`
  on EVEN columns but becomes YELLOW (4) on ODD columns; background unchanged.

  So OUTPUT == INPUT everywhere EXCEPT colored cells sitting at an ODD column,
  which flip to yellow.  No cell appears or disappears.

Encoding (route the 10-channel expansion into the FREE output via Where):
  Work on the top-left S x S patch (S=15, the max grid size) with bool masks,
  then expand the change mask to 30x30 by zero-Concat (Pad rejects bool) so the
  ONLY full-canvas plane is the [1,1,30,30] bool cond.

    g       = rowocc (x) colocc      -> in-grid rectangle (separable, bool SxS)
    ch0     = input[:,0:1,:S,:S]      -> in-grid background indicator (fp32 SxS)
    colored = g AND NOT ch0           -> a painted cell (bool SxS)
    cond15  = colored AND oddcol      -> painted AND odd column (bool SxS)
    cond30  = zero-pad cond15 to 30x30 via two Concats (bool)
    output  = Where(cond30, yellow_onehot[1,10,1,1], input)

  Where broadcasts the [1,1,30,30] cond and the [1,10,1,1] yellow one-hot against
  the FREE [1,10,30,30] input, so the only 10-channel tensor is the free output.
  Dominant intermediate is a single [1,1,30,30] bool plane (900B).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model

S = 15  # max grid size (generator: size in 3..15)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    i64 = np.int64
    BOOL = onnx.TensorProto.BOOL

    init("st0", [0], i64)
    init("enS", [S], i64)
    init("axH", [2], i64)
    init("axW", [3], i64)
    init("st000", [0, 0, 0], i64)
    init("en1SS", [1, S, S], i64)
    init("axCHW", [1, 2, 3], i64)

    # --- in-grid odd-column rectangle on the SxS patch (separable) -----
    # Fold the odd-column predicate straight into the column occupancy
    # vector, so g_odd = rowocc (x) (colocc AND oddcol) is built with no
    # extra full-patch plane.
    n("ReduceMax", ["input"], "rocc", axes=[1, 3], keepdims=1)   # [1,1,30,1] f32
    n("ReduceMax", ["input"], "cocc", axes=[1, 2], keepdims=1)   # [1,1,1,30] f32
    n("Cast", ["rocc"], "rb", to=BOOL)
    n("Cast", ["cocc"], "cb", to=BOOL)
    n("Slice", ["rb", "st0", "enS", "axH"], "rbS")               # [1,1,S,1]
    n("Slice", ["cb", "st0", "enS", "axW"], "cbS")               # [1,1,1,S]
    oddc = (np.arange(S) % 2 == 1).reshape(1, 1, 1, S)
    init("oddC", oddc, np.bool_)
    n("And", ["cbS", "oddC"], "cbO")                             # [1,1,1,S] odd in-grid cols
    n("And", ["rbS", "cbO"], "gOdd")                             # [1,1,S,S] in-grid odd cells

    # --- colored AND odd = gOdd AND NOT background --------------------
    n("Slice", ["input", "st000", "en1SS", "axCHW"], "in0f")     # [1,1,S,S] f32 (ch0)
    n("Cast", ["in0f"], "bg", to=BOOL)                           # in-grid bg
    n("Not", ["bg"], "nbg")                                      # painted (within grid)
    n("And", ["gOdd", "nbg"], "cond15")                          # [1,1,S,S] bool

    # --- expand cond15 -> 30x30 by zero-Concat (bool, no Pad) ----------
    init("zR", np.zeros((1, 1, S, 30 - S), np.bool_), np.bool_)
    n("Concat", ["cond15", "zR"], "condW", axis=3)               # [1,1,S,30]
    init("zB", np.zeros((1, 1, 30 - S, 30), np.bool_), np.bool_)
    n("Concat", ["condW", "zB"], "cond30", axis=2)               # [1,1,30,30]

    # --- recolor: odd-painted cells -> yellow (4), else keep input -----
    yh = np.zeros((1, 10, 1, 1), np.float32)
    yh[0, 4, 0, 0] = 1.0
    init("yellow", yh, np.float32)
    n("Where", ["cond30", "yellow", "input"], "output")

    return _model(nodes, inits)
