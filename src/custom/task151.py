"""task151 (ARC-AGI 67a423a3) — "yellow 3x3 ring around the row/column cross intersection".

Rule (from the generator task_67a423a3.py):
  A size x size (size in 4..12) grid holds a CROSS: one FULL column `col` painted colors[1]
  and one FULL row `row` painted colors[0] (colors = 2 distinct non-yellow colours; their
  intersection cell ends up colors[1] because the column write runs last).  The INPUT is
  exactly this cross.  The OUTPUT = the cross PLUS an 8-cell yellow(=4) ring: every cell of
  the 3x3 box centred on the intersection (row,col), EXCEPT the centre itself, is set to
  yellow (overwriting whatever cross colour was there).  (xpose flips input AND output the
  same way, so it is irrelevant — the rule is orientation symmetric.)

  Locating the intersection LOCALLY (a 3x3 conv suffices, which is why the public learned
  conv works approximately):  the intersection is the UNIQUE cell whose four orthogonal
  neighbours are ALL non-background.  Because row,col in [1,size-2] the full row/column
  always extend at least one cell on each side, so the 4-orthogonal-neighbour non-bg COUNT
  reaches 4 only at the intersection (verified exact over 3000 fresh instances; the public
  learned [10,10,3,3] conv overfits to training colours and FAILS the {5,8} colour pair —
  198/200 fresh — whereas this closed-form detector is colour-agnostic and passes 200/200).

Encoding (exact, colour-agnostic; routes the 10-ch expansion into the FREE Where output):
  nb     = Conv(input, K4)         K4[1,10,3,3]: weight 1 on the 4 orthogonal taps of every
                                   colour channel 1..9, 0 on bg ch0 and on diagonals/centre;
                                   SAME pad -> nb[1,1,30,30] = # non-bg orthogonal neighbours.
  centre = Greater(nb, 3.5)        bool: the unique intersection cell (nb==4).
  dil    = Conv(Cast(centre,f32), K8)   K8 = 3x3 ones, SAME pad -> # centres in the 3x3 box.
  ring   = And(Greater(dil,0.5), Not(centre))   the 8 box cells that are NOT the centre.
  output = Where(ring, yellow_onehot[1,10,1,1], input)   yellow = channel-4 one-hot constant.

  Dominant intermediate: the two fp32 [1,1,30,30] conv planes (nb, dil) at 3600B each — the
  detection genuinely needs the full neighbourhood TWICE (detect centre, then dilate it), so
  this floors near ~15.9 and CANNOT beat the public mem-0 single-conv's 18.19 stored score.
  Its value is real-LB generalization (the public net scores 0 on Kaggle held-out colours).
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
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- K4 : 4-orthogonal-neighbour non-bg count over colour channels 1..9 ----
    K4 = np.zeros((1, 10, 3, 3), dtype=np.float32)
    for ch in range(1, 10):                 # exclude bg channel 0
        K4[0, ch, 0, 1] = 1.0               # up
        K4[0, ch, 2, 1] = 1.0               # down
        K4[0, ch, 1, 0] = 1.0               # left
        K4[0, ch, 1, 2] = 1.0               # right
    init("K4", K4, np.float32)
    n("Conv", ["input", "K4"], "nb", pads=[1, 1, 1, 1], kernel_shape=[3, 3])  # [1,1,30,30] f32

    # ---- centre = (nb == 4) i.e. all four orthogonal neighbours non-bg ----------
    init("T35", np.array(3.5, np.float32), np.float32)
    n("Greater", ["nb", "T35"], "centre")          # [1,1,30,30] bool
    n("Cast", ["centre"], "centre_f", to=F32)       # [1,1,30,30] f32 {0,1}

    # ---- dilate the centre over its 3x3 box (8 neighbours + itself) -------------
    K8 = np.ones((1, 1, 3, 3), dtype=np.float32)
    init("K8", K8, np.float32)
    n("Conv", ["centre_f", "K8"], "dil", pads=[1, 1, 1, 1], kernel_shape=[3, 3])  # [1,1,30,30] f32

    # ---- ring = (in 3x3 box) AND NOT (centre) -----------------------------------
    init("T05", np.array(0.5, np.float32), np.float32)
    n("Greater", ["dil", "T05"], "inbox")           # [1,1,30,30] bool
    n("Not", ["centre"], "notc")                    # [1,1,30,30] bool
    n("And", ["inbox", "notc"], "ring")             # [1,1,30,30] bool

    # ---- yellow one-hot constant (channel 4) ------------------------------------
    yellow = np.zeros((1, 10, 1, 1), dtype=np.float32)
    yellow[0, 4, 0, 0] = 1.0
    init("yellow", yellow, np.float32)

    # ---- output = Where(ring, yellow, input)  -> FREE [1,10,30,30] ---------------
    n("Where", ["ring", "yellow", "input"], "output")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task151", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
