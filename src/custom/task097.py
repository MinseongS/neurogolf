"""Task 097 (ARC-AGI 42a50994): erase isolated single-colour pixels.

Rule (from the ARC-GEN generator): the grid (width,height in [5,20]) holds
scattered pixels, all of ONE random colour. For every cell, count
`friends` = number of coloured cells in its 3x3 neighbourhood INCLUDING itself;
if friends <= 1 (i.e. a coloured cell with no coloured 8-neighbour, or a bg cell)
the cell becomes black(0). So the ONLY cells that change are coloured cells with
zero coloured 8-neighbours -> they are cleared to background. Background and
off-grid cells are unchanged.

  => output = input EVERYWHERE except in-grid coloured cells that have ZERO
     coloured 8-neighbours, which are cleared to ch0 (background).

Floor-break encoding (single conv plane, no 10-channel intermediate):
  BANDED 3x3 Conv over channels 1..9 with CENTRE weight 10 and the 8 neighbour
  weights 1:
      C = 10*(centre coloured) + (# coloured 8-neighbours)
  - bg / off-grid cell (centre not coloured):  C in 0..8.
  - coloured cell:                             C = 10 + (#nbrs) in 10..18.
  A coloured-and-isolated cell is EXACTLY C == 10 (the unique band, no separate
  occupancy plane needed; bg/off-grid never reach 10).  fp32 Equal on integer
  conv values is exact.
      kill   = (C == 10)
      output = Where(kill, bg_onehot, input)
  Off-grid cells (all-zero input) have C<=8 -> kill=0 -> output=input (stays
  all-zero); in-grid kept/bg cells pass input through; isolated coloured cells
  become ch0=1.  Off-grid SAME-pad neighbours read as 0, matching the generator's
  get_val() out-of-grid = 0.

  mem = ONE fp32 [1,1,30,30] conv plane (3600B, irreducible: Conv must consume the
  fp32 input) + one bool [1,1,30,30] kill mask; the 10-ch expansion is routed into
  the FREE Where output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **kw):
        nodes.append(helper.make_node(op, ins, [out], **kw))

    # Banded 3x3 Conv over channels 1..9: centre weight 10, the 8 neighbours weight 1.
    W = np.ones((1, 10, 3, 3), dtype=np.float32)
    W[0, 0, :, :] = 0.0          # channel 0 (background) never counts
    W[0, 1:, 1, 1] = 10.0        # centre cell of every colour channel
    init("W3x3", W)
    init("ten", np.array([10.0], dtype=np.float32))
    bg = np.zeros((1, 10, 1, 1), dtype=np.float32)
    bg[0, 0, 0, 0] = 1.0
    init("bg_onehot", bg)

    n("Conv", ["input", "W3x3"], "C", pads=[1, 1, 1, 1])   # [1,1,30,30] fp32 banded count
    n("Equal", ["C", "ten"], "kill")                        # bool (opset 11 float Equal): coloured & isolated
    n("Where", ["kill", "bg_onehot", "input"], "output")    # route 10-ch into FREE fp32 output

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task097", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
