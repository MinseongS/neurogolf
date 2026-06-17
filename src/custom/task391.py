"""task391 (ARC-AGI f8b3ba0a) — "sort the rare colours by descending count".

Rule (from the generator):
  The INPUT renders a width x height "bitmap" of cells onto a larger canvas of
  shape (2*height+1) x (3*width+1): logical cell (r,c) is drawn as a 2-wide
  horizontal pair at input[2r+1][3c+1..3c+2], everything else is background 0.
  The bitmap uses exactly 4 distinct colours (all in 1..9): one dominant
  "background" colour fills the grid and three other colours are sprinkled with
  DISTINCT counts sampled from {1,2,3,4}.  The dominant colour's count is always
  strictly larger (grid >=18 cells, sprinkled total <=9).

  OUTPUT is a 3x1 column: the THREE non-dominant colours sorted by DESCENDING
  count.  output[0]=most-frequent rare colour, output[2]=least-frequent.

Encoding (no per-cell plane — everything is tiny [1,10,..] tensors):
  cnt_k = ReduceSum(input_k) over space  -> [1,10,1,1] (relative order is
          preserved since each cell is a 2-px pair).  rank_k = #{j : cnt_j>cnt_k}
          via a [10,10] Greater + ReduceSum.  The dominant colour gets rank 0;
          the three rare colours get ranks 1,2,3 (unique).  Output row R wants
          the colour whose rank == R+1.  We build H[1,10,30,1] = Equal(rank,
          target[1,1,30,1]) with target=[1,2,3,99,99,...] (99 never matched by a
          rank in {0..4}, so rows R>=3 and the unused/dominant channels stay
          false), then AND with a col-0 mask [1,1,1,30] -> the FREE bool output
          [1,10,30,30].  No [1,10,30,30] intermediate is ever materialised.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
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

    # ---- per-colour pixel counts (input is FREE) --------------------------
    # Each cell is a 2-px pair so pixel counts preserve cell-count order.
    # Channel 0 (canvas background) is NOT a bitmap colour and must be
    # excluded from the ranking -> mask it to 0 (chmask: ch0=0, ch1..9=1).
    n("ReduceSum", ["input"], "cnt0", axes=[2, 3], keepdims=1)  # [1,10,1,1] f32
    chmask = np.ones((1, 10, 1, 1), np.float32)
    chmask[0, 0, 0, 0] = 0.0
    init("chmask", chmask, np.float32)
    n("Mul", ["cnt0", "chmask"], "cnt")  # [1,10,1,1] f32, ch0 zeroed

    # ---- pairwise rank: rank_k = #{ j : cnt_j > cnt_k } -------------------
    init("shapeA", np.array([10, 1], np.int64), np.int64)
    init("shapeB", np.array([1, 10], np.int64), np.int64)
    n("Reshape", ["cnt", "shapeA"], "cntA")   # [10,1]  -> k on rows
    n("Reshape", ["cnt", "shapeB"], "cntB")   # [1,10]  -> j on cols
    n("Greater", ["cntB", "cntA"], "G")       # [10,10] bool  G[k,j] = cnt_j > cnt_k
    n("Cast", ["G"], "Gf", to=F32)
    n("ReduceSum", ["Gf"], "rank2d", axes=[1], keepdims=1)  # [10,1] f32

    # reshape rank to [1,10,1,1] for broadcasting against the target ramp
    init("shapeR", np.array([1, 10, 1, 1], np.int64), np.int64)
    n("Reshape", ["rank2d", "shapeR"], "rank")  # [1,10,1,1] f32

    # ---- target rank per output ROW: [1,2,3,99,...,99] over 30 rows -------
    target = np.full((1, 1, 30, 1), 99.0, np.float32)
    target[0, 0, 0, 0] = 1.0
    target[0, 0, 1, 0] = 2.0
    target[0, 0, 2, 0] = 3.0
    init("target", target, np.float32)

    n("Equal", ["rank", "target"], "H")  # [1,10,30,1] bool

    # ---- column-0 mask [1,1,1,30] and route into FREE output --------------
    colmask = np.zeros((1, 1, 1, 30), np.bool_)
    colmask[0, 0, 0, 0] = True
    init("colmask", colmask, np.bool_)

    n("And", ["H", "colmask"], "output")  # [1,10,30,30] BOOL (free)

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task391", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
