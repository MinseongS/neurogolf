"""Task 097 (ARC-AGI): erase isolated single-colour pixels.

Rule (from the ARC-GEN generator): the grid holds scattered pixels, all of one
colour. For every cell, count `friends` = number of coloured cells in its 3x3
neighbourhood INCLUDING itself; if friends <= 1 (the pixel is isolated, only
itself) the cell becomes black(0). Everything else is unchanged. Background and
off-grid cells stay as they are.

Floor-break encoding (no 10-channel intermediate): the whole transform is one
final Where into the free `output`.
  cnt   = Conv3x3(coloured)              # friends count per cell
  keep  = cnt > 1.5                      # >=2 friends -> keep the cell
  ingrid= ReduceMax(input over channels) > 0.5   # 1 in-grid, 0 off-grid
  mask  = keep OR (NOT ingrid)
  output= Where(mask, input, bg_onehot)
Routing: off-grid cells (all-zero input) have ingrid=0 -> mask=1 -> output=input
(stays all-zero, matching the target). In-grid kept cells -> input. In-grid
erased/background cells -> bg_onehot (ch0=1 = black). The 3x3 Conv weight is 0
on channel 0 and 1 on channels 1..9, so it counts only coloured cells.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    # 3x3 Conv weight: count coloured cells (channels 1..9) in the neighbourhood.
    W = np.zeros((1, 10, 3, 3), dtype=np.float32)
    W[0, 1:, :, :] = 1.0
    init("W3x3", W)
    init("thr15", np.array([1.5], dtype=np.float32))
    init("thr05", np.array([0.5], dtype=np.float32))
    bg = np.zeros((1, 10, 1, 1), dtype=np.float32)
    bg[0, 0, 0, 0] = 1.0
    init("bg_onehot", bg)

    def n(op, ins, out, **kw):
        nodes.append(helper.make_node(op, ins, [out], **kw))

    n("Conv", ["input", "W3x3"], "cnt", pads=[1, 1, 1, 1])         # [1,1,30,30] friends
    n("Greater", ["cnt", "thr15"], "keep")                          # >=2 friends
    n("ReduceMax", ["input"], "s", axes=[1], keepdims=1)            # 1 in-grid, 0 off-grid
    n("Greater", ["s", "thr05"], "ingrid")
    n("Not", ["ingrid"], "notin")
    n("Or", ["keep", "notin"], "mask")
    n("Where", ["mask", "input", "bg_onehot"], "output")

    return _model(nodes, inits)
