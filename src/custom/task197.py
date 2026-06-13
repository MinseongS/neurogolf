"""Task 197 (82819916): pattern broadcast across partial rows.

Rule (from the ARC-GEN generator): a binary `pattern` of length `width` is
fully rendered in the first (topmost) marked row using that row's two colors
(light for pattern==0, dark for pattern==1).  Every lower marked row only shows
a leading prefix in the input (cells up to and including the first pattern
change); the output fills the whole row with the same pattern using that row's
own two colors.

Equivalently, for EVERY row and column:
    output[r, c] = input[r, src[c]]
where `src[c]` is a global column-gather index:
    src[c] = 0            if c < width and pattern[c] == pattern[0]
    src[c] = j            if c < width and pattern[c] != pattern[0]
    src[c] = c            if c >= width   (all-zero canvas columns)
with j = first column where the pattern differs from pattern[0].

`pattern` and the in-width mask are read off the topmost marked row (row0),
which is the only fully-rendered row.  topmost row = the row whose occupancy is
1 and which has no occupied row above it (strict-lower-triangular matmul).

Graph: derive the [30] int32 gather vector `src`, then a single
Gather(input, src, axis=3) writes `output` directly.  All intermediates are
1-D / 2-D and tiny.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ..builders import _model

I32 = onnx.TensorProto.INT32
F32 = np.float32
I64 = np.int64


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, list(inputs), [out], **attrs))
        return out

    # ---- constants -------------------------------------------------------
    init("cidx", np.arange(30, dtype=np.float32), F32)  # [30] index vector
    init("one", np.array(1.0, np.float32), F32)
    init("big", np.array(1000.0, np.float32), F32)

    # ---- topmost occupied row index r0 ----------------------------------
    # background is channel-0 one-hot; a cell is COLORED iff one of channels
    # 1..9 is set.  colored_count[r] = (sum over all channels) - (channel 0).
    # Reduce columns first into a small [1,10,30] tensor, reused for both sums.
    n("ReduceSum", ["input"], "colsum", axes=[3], keepdims=0)     # [1,10,30] ch x row
    n("ReduceSum", ["colsum"], "allsum", axes=[1], keepdims=0)    # [1,30] = W in-grid
    init("ch0", np.array(0, np.int32), np.int32)
    n("Gather", ["colsum", "ch0"], "bgsum", axis=1)               # [1,30] bg/row
    n("Sub", ["allsum", "bgsum"], "rowsum")                       # [1,30] colored/row
    n("Clip", ["rowsum"], "occ", min=0.0, max=1.0)                # [1,30] in {0,1}
    # r0 = min row index where occ==1  (argmin trick: r*occ + BIG*(1-occ))
    n("Mul", ["cidx", "occ"], "roc")
    n("Sub", ["one", "occ"], "nocc")
    n("Mul", ["nocc", "big"], "rpen")
    n("Add", ["roc", "rpen"], "rcand")                            # [1,30]
    n("Reshape", ["rcand", "sh_30"], "rcand1")                    # [30]
    n("ReduceMin", ["rcand1"], "r0f", axes=[0], keepdims=0)       # scalar
    n("Cast", ["r0f"], "r0", to=I32)                              # scalar int32

    # ---- row0 one-hot grid (the topmost, fully-rendered row) ------------
    n("Gather", ["input", "r0"], "row0", axis=2)                  # [1,10,30]

    # base color one-hot at column 0, as a [1,1,10] row vector.
    init("i0", np.array([0], np.int32), np.int32)
    n("Gather", ["row0", "i0"], "base", axis=2)                  # [1,10,1]
    n("Reshape", ["base", "sh_1_1_10"], "baser")                 # [1,1,10]
    init("sh_1_1_10", np.array([1, 1, 10], np.int64), I64)

    # same[c] = sum_ch base[ch]*row0[ch,c]  via matmul (no canvas product)
    n("MatMul", ["baser", "row0"], "samemat")                    # [1,1,30]
    n("Reshape", ["samemat", "sh_30"], "same")                   # [30]
    # diff[c] = 1 - same[c]
    n("Sub", ["one", "same"], "diff")                            # [30]

    # incol[c] = 1 if column c is colored in row0 (within width), else 0.
    # background cells are encoded all-zero, so incol = sum over channels.
    n("ReduceSum", ["row0"], "incol", axes=[1], keepdims=0)      # [1,30]

    # ---- first differing column j ---------------------------------------
    # dm[c] = diff[c]*incol[c]  (1 only for in-width differing columns)
    n("Mul", ["diff", "incol"], "dm")                            # [1,30]
    # cand[c] = c*dm + big*(1-dm)
    n("Mul", ["cidx", "dm"], "cd")
    n("Sub", ["one", "dm"], "ndm")
    n("Mul", ["ndm", "big"], "pen")
    n("Add", ["cd", "pen"], "candnd")                            # [.,30]
    n("Reshape", ["candnd", "sh_30"], "cand")                    # [30]
    n("ReduceMin", ["cand"], "j", axes=[0], keepdims=1)          # [1] scalar j

    # ---- src[c] ----------------------------------------------------------
    # within width:  j*diff[c]            (0 or j)
    # outside width: c
    n("Mul", ["diff", "j"], "jd")                                # [1,30] (j broadcast)
    n("Mul", ["jd", "incol"], "inpart")                          # zero outside width
    n("Sub", ["one", "incol"], "outmask")                        # 1 outside width
    n("Mul", ["outmask", "cidx"], "outpart")                     # c outside width
    n("Add", ["inpart", "outpart"], "srcf")                      # [1,30]
    n("Cast", ["srcf"], "srci", to=I32)
    n("Reshape", ["srci", "sh_30"], "src")                       # [30]
    init("sh_30", np.array([30], np.int64), I64)

    # ---- final gather into output ---------------------------------------
    n("Gather", ["input", "src"], "output", axis=3)

    return _model(nodes, inits)
