"""Task 217 (8f2ea7aa): self-similar fractal sprite -> sprite (X) sprite.

Rule (from ARC-GEN): the 9x9 input (size=3, a 3x3 grid of 3x3 blocks) holds a
single copy of a 3x3 sprite S placed in one block. The output is the Kronecker
fractal S (X) S: output block (bR,bC) is filled with S iff S[bR,bC] is set, i.e.
  output[bR*3+r][bC*3+c] = S[bR][bC] * S[r][c].
Each example uses a single color, so exactly one of channels 1..9 carries S.

Graph (depthwise on the 9 color channels; channel 0 rebuilt as complement):
- Slice input to channels 1..9 and the 9x9 region: [1,9,9,9].
- Reshape to [1,9,3,3,3,3]=(br,r,bc,c); ReduceSum over br,bc -> sprite S [1,9,3,3]
  (only one block is nonzero, so this is exact). Reshape to kernel [9,1,3,3].
- Grouped ConvTranspose (stride 3, group 9): input = block-map S [1,9,3,3],
  weight = sprite S [9,1,3,3]; block (bR,bC) -> S[bR,bC]*S = the fractal (col).
- channel 0 = 1 - sum_c col over the 9x9 region; Concat then Pad to 30x30.
"""

import numpy as np
from onnx import helper, numpy_helper

from ..builders import _model


def build(task):
    inits = []
    nodes = []

    def init(name, arr, dtype=np.int64):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    # slice channels 1..9 and the top-left 9x9 region
    init("s_st", np.array([1, 0, 0], np.int64))
    init("s_en", np.array([10, 9, 9], np.int64))
    init("s_ax", np.array([1, 2, 3], np.int64))
    n("Slice", ["input", "s_st", "s_en", "s_ax"], "g99")          # [1,9,9,9]

    # reshape to (1,9,br,r,bc,c) and sum over block axes -> sprite S
    init("sh6", np.array([1, 9, 3, 3, 3, 3], np.int64))
    n("Reshape", ["g99", "sh6"], "g6")                            # [1,9,3,3,3,3]
    n("ReduceSum", ["g6"], "spr6", axes=[2, 4], keepdims=0)       # [1,9,3,3]

    # block-map input [1,9,3,3] and grouped ConvTranspose kernel [9,1,3,3]
    init("sh4", np.array([1, 9, 3, 3], np.int64))
    n("Reshape", ["spr6", "sh4"], "B")
    init("shk", np.array([9, 1, 3, 3], np.int64))
    n("Reshape", ["spr6", "shk"], "K")

    # fractal on the 9 colored channels
    n("ConvTranspose", ["B", "K"], "col", strides=[3, 3], group=9)  # [1,9,9,9]

    # channel 0 = 1 - any-color, within the 9x9 region
    n("ReduceSum", ["col"], "csum", axes=[1], keepdims=1)           # [1,1,9,9]
    init("one", np.array(1.0, np.float32), np.float32)
    n("Sub", ["one", "csum"], "ch0")                                # [1,1,9,9]

    # reassemble [ch0, col] and pad to the 30x30 canvas (opset10 Pad attrs)
    n("Concat", ["ch0", "col"], "full", axis=1)                     # [1,10,9,9]
    n("Pad", ["full"], "output", mode="constant",
      pads=[0, 0, 0, 0, 0, 0, 21, 21], value=0.0)

    return _model(nodes, inits)
