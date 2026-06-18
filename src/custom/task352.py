"""task352 (ARC-AGI dc1df850) — stamp a blue 8-neighbour halo around every RED dot.

Rule (from generator task_dc1df850.py, active grid H x W, W in 5..10, H in {W-1,W}):
  Sparse coloured dots placed at distinct (r,c) with a non-overlap guard that keeps
  every pair of dots >= 2 apart in Chebyshev distance.  Dot colours come from
  {2 (red)} U random_color(exclude=[blue,red]) = {2,3,4,5,6,7,8,9}; blue(1) NEVER
  appears in the input.  In the OUTPUT:
    - every dot is copied (centre keeps its colour);
    - for each RED dot, common.draw paints BLUE on the 8 surrounding cells (clipped
      to the active grid, then the red centre is re-painted).  Because dots are >=2
      apart, a red halo NEVER lands on another dot -> it only ever covers background
      cells inside the active grid.

  Closed form, per output channel (verified 0 mismatches on 500 fresh instances):
    ch2 (red)     = input[ch2]                      (centre copy)
    ch3..ch9      = input[ch_k]                     (centre copy)
    ch0 (bg)      = bg-cell AND not adjacent to red = B0c - (#red in 3x3) > 0
    ch1 (blue)    = bg-cell AND     adjacent to red = B0c + (#red in 3x3) - 1 > 0
  where B0c = input[ch0] at the centre (=1 iff in-grid background; 0 off-grid).
  The input-ch0 centre term is the load-bearing IN-GRID gate: an off-grid-on-canvas
  neighbour of an edge red has B0c=0 so ch1 = 0+1-1 = 0 (not painted), matching the
  draw-clips-to-grid target; a true blue cell is in-grid bg so B0c=1.

Encoding (single GROUPED Conv whose output IS the graph output -> mem 0):
  All cross-channel dependencies (red ch2 -> blue ch1 / bg ch0) live among channels
  {0,1,2,3,4}; channels {5..9} are pure copies.  So a Conv with group=2 (input/output
  split [0..4]|[5..9]) suffices: weight [10, 5, 3, 3] = 450 params (+10 bias) = 460,
  vs the public dense [10,10,3,3] = 910.  The harness scores (out>0), and every output
  channel's integer response is >0 exactly on its target -> exact.

  pts 18.87, mem 0, params 460, fresh 200/200.  Dominant cost = params (mem 0, the
  conv output is the free graph output).  Irreducible further: the 3x3 dilation forces
  k=3 on ch0/ch1, and group0 must hold {0,1,2} (red feeds blue+bg) so the minimal equal
  group size dividing 10 is 5 -> I/group=5 fixed.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT


def build(task):
    O, Ig, k = 10, 5, 3            # output ch, input-ch per group, kernel size
    W = np.zeros((O, Ig, k, k), np.float32)
    B = np.zeros((O,), np.float32)

    # ---- group 0: output channels 0..4 read input channels 0..4 ----
    # local index within group0: 0->ch0, 1->ch1, 2->ch2, 3->ch3, 4->ch4
    # ch0 (bg): +1 on input-ch0 centre, -1 on the full 3x3 of input-ch2 (red)
    W[0, 0, 1, 1] = 1.0
    W[0, 2, :, :] = -1.0
    # ch1 (blue): +1 on input-ch0 centre, +1 on the full 3x3 of input-ch2, bias -1
    W[1, 0, 1, 1] = 1.0
    W[1, 2, :, :] = 1.0
    B[1] = -1.0
    # ch2 (red): centre copy of input-ch2
    W[2, 2, 1, 1] = 1.0
    # ch3, ch4: centre copy of own channel
    W[3, 3, 1, 1] = 1.0
    W[4, 4, 1, 1] = 1.0

    # ---- group 1: output channels 5..9 read input channels 5..9 ----
    # local index 0->ch5, 1->ch6, ... 4->ch9 ; pure centre copies
    for j in range(5):
        W[5 + j, j, 1, 1] = 1.0

    # ---- ORT-1.26 grouped-Conv workaround (HARMLESS) ----
    # onnxruntime 1.26's grouped Conv miscomputes when group-0's weight block is
    # sparse (it silently corrupts group-1's output channels; the ONNX reference
    # evaluator computes the correct result, confirming an ORT bug, not a spec
    # issue).  Densifying group-0's weight block fixes it.  We add dummy weights on
    # group-0 LOCAL input index 1 == global input channel 1 (BLUE), which NEVER
    # appears in any input grid (dot colours come from {2,3,...,9}; blue is only
    # ever produced in the OUTPUT).  These weights multiply by an always-zero plane
    # so they change nothing, and params count ELEMENTS (already 450) so they are
    # free.  Verified ORT == reference on 500 fresh instances with this in place.
    W[0:5, 1, :, :] = 1.0

    inits = [
        numpy_helper.from_array(np.ascontiguousarray(W, np.float32), "W"),
        numpy_helper.from_array(np.ascontiguousarray(B, np.float32), "B"),
    ]
    node = helper.make_node(
        "Conv", ["input", "W", "B"], ["output"],
        group=2, kernel_shape=[k, k], pads=[1, 1, 1, 1], strides=[1, 1],
    )

    inp = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph([node], "task352", [inp], [out], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = IR_VERSION
    return model
