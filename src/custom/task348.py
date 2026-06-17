"""task348 (ARC-AGI db3e9e38) — orange/cyan pyramid spread from a vertical line.

Rule (from the generator, verified fresh):
  INPUT: a single vertical ORANGE(7) line at column `col`, rows 0..length-1,
  on a width x height grid (width,height in 5..10, anchored top-left).
  OUTPUT: a triangular "pyramid" of colored cells.  Cell (r,c) is colored iff
      r + |c - col| < length
  i.e. the colored region per column c is the row-prefix [0, length-|c-col|).
  Color = ORANGE(7) if (c - col) is even (same parity as col), else CYAN(8).
  All other in-grid cells stay background; off-grid cells stay all-zero.

Recovery (scalars / 1-D vectors only — exactly ONE 30x30 plane materialized):
  * orange-per-column  colsum7 = ReduceSum(input,axes=2)[:,7]  -> [1,1,1,30]
        length = ReduceMax(colsum7)        (orange column has `length` pixels)
        col    = Sum((colsum7>0) * colramp) (single nonzero column index)
  * width  = Sum(ReduceMax(input,axes=[1,2]))  (in-grid columns, top-left grid)
  * colabs[c] = |c - col| ;  thr0 = length - colabs
  * thresh[c] = thr0 * (c < width)   -> off-grid columns forced to <=0 so the
        pyramid never spills past the right edge.  (rows always r<length<height,
        and c>=0, so the row/left edges need no gate.)
  * colored = Less(rowramp[1,1,30,1], thresh[1,1,1,30])  -> [1,1,30,30] bool
        (the ONLY full 30x30 intermediate).
  * parity colour: colorval[c] = 7 + ((c+col) mod 2)   (even-parity -> 7 orange)
        colorOH = (arange10 == colorval) -> [1,10,1,30] one-hot (tiny, 300 elems)
  * output = Where(colored[1,1,30,30], colorOH[1,10,1,30], input[1,10,30,30])
        else-branch = input already carries correct in-grid bg + off-grid zeros,
        so the 10-ch expansion lands entirely in the FREE output.

Dominant intermediate: the single `colored` [1,1,30,30] plane.  Everything else
is 1-D column/row vectors or scalars.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- constants -------------------------------------------------------
    init("rowramp", np.arange(30, dtype=np.float16).reshape(1, 1, 30, 1), np.float16)
    init("colramp", np.arange(30, dtype=np.float16).reshape(1, 1, 1, 30), np.float16)
    init("arange10", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)
    init("zeroF", np.array(0.0, np.float16), np.float16)
    init("twoF", np.array(2.0, np.float16), np.float16)
    init("sevenF", np.array(7.0, np.float16), np.float16)

    # ---- recover length & col from orange (channel 7) column profile -----
    # ReduceSum over rows -> [1,10,1,30]; slice channel 7 -> [1,1,1,30].
    n("ReduceSum", ["input"], "colsum_all", axes=[2], keepdims=1)  # [1,10,1,30] f32
    init("o_s", np.array([0, 7, 0, 0], np.int64), np.int64)
    init("o_e", np.array([1, 8, 1, 30], np.int64), np.int64)
    init("o_ax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["colsum_all", "o_s", "o_e", "o_ax"], "colsum7")    # [1,1,1,30] f32
    n("Cast", ["colsum7"], "colsum7h", to=F16)                     # [1,1,1,30] f16

    n("ReduceMax", ["colsum7h"], "length", axes=[3], keepdims=1)   # [1,1,1,1] f16
    n("Greater", ["colsum7h", "zeroF"], "colmask")                 # [1,1,1,30] bool
    n("Cast", ["colmask"], "colmaskf", to=F16)                     # [1,1,1,30] f16
    n("Mul", ["colmaskf", "colramp"], "colidx")                    # [1,1,1,30] f16
    n("ReduceSum", ["colidx"], "col", axes=[3], keepdims=1)        # [1,1,1,1] f16

    # ---- recover width (in-grid column count) ----------------------------
    n("ReduceMax", ["input"], "colpres", axes=[1, 2], keepdims=1)  # [1,1,1,30] f32
    n("Cast", ["colpres"], "colpresh", to=F16)                     # [1,1,1,30] f16
    n("ReduceSum", ["colpresh"], "width", axes=[3], keepdims=1)    # [1,1,1,1] f16

    # ---- thresh[c] = (length - |c-col|) gated to c < width ---------------
    n("Sub", ["colramp", "col"], "cmc")                            # [1,1,1,30] f16
    n("Abs", ["cmc"], "colabs")                                    # [1,1,1,30] f16
    n("Sub", ["length", "colabs"], "thr0")                         # [1,1,1,30] f16
    n("Less", ["colramp", "width"], "cin")                         # [1,1,1,30] bool
    n("Cast", ["cin"], "cinf", to=F16)                             # [1,1,1,30] f16
    n("Mul", ["thr0", "cinf"], "thresh")                           # [1,1,1,30] f16

    # ---- colored mask (the only 30x30 plane) -----------------------------
    n("Less", ["rowramp", "thresh"], "colored")                    # [1,1,30,30] bool

    # ---- per-column colour one-hot ---------------------------------------
    n("Add", ["colramp", "col"], "cpc")                            # [1,1,1,30] f16
    n("Mod", ["cpc", "twoF"], "parity", fmod=1)                    # [1,1,1,30] f16
    n("Add", ["parity", "sevenF"], "colorval")                     # [1,1,1,30] f16
    n("Equal", ["arange10", "colorval"], "colorOH_b")              # [1,10,1,30] bool
    n("Cast", ["colorOH_b"], "colorOH", to=F32)                    # [1,10,1,30] f32

    # ---- route the 10-ch expansion into the FREE output ------------------
    n("Where", ["colored", "colorOH", "input"], "output")          # [1,10,30,30] f32

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task348", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
