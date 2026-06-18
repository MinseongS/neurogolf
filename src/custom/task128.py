"""task128 (ARC-AGI 5521c0d9) — stack each bottom-anchored box on top of itself.

Rule (from the ARC-GEN generator, verified fresh):
  A 15x15 grid (placed top-left on the 30x30 canvas) holds up to `boxes`=3 solid
  axis-aligned rectangles, each BOTTOM-ANCHORED (occupying rows [size-tall, size)
  of its columns) with colour drawn from {1,2,4}. Boxes never overlap
  horizontally.  The output copies each box UP by exactly its own height:
      output[r-tall][c] = colour      for the box at rows [size-tall, size).
  i.e. a box of height h sitting at the bottom is re-stamped directly ON TOP of
  its original footprint (output rows [size-2h, size-h)); the original footprint
  is cleared to background.

Why this is a single DEPTHWISE conv (and how to make it small):
  The operation is purely PER-COLUMN (a width-1 kernel) and PER-CHANNEL.  Each
  column is one of only 8 canonical cases: empty, or a bottom-anchored run of
  height h in 1..7.  For the COLOUR channels {1,2,4} the map
  "run [15-h,14] -> stamp [15-2h,15-h-1]" is LINEARLY SEPARABLE in the column's
  occupancy vector (the harness scores out>0): a single 1-D filter + bias whose
  sign is +1 exactly on the output band.  For the BACKGROUND channel 0 (the
  harness target sets ch0=1 on every in-grid bg cell, 0 off-grid and 0 on box
  cells) the per-column "bg-after-transform" rule is ALSO linearly separable in
  ch0's own input occupancy.  Unused colour channels (3,5,6,7,8,9) never appear
  -> zero kernel + zero/negative bias -> output 0.

  A single ONNX Conv shares ONE (kernel_height, pads) across all channels, so we
  need ONE (L, pad_top, pad_bottom) that separates BOTH problems.  Joint
  margin-perceptron search => L=29, pads (top=15, bottom=13): the colour kernel
  needs to reach DOWN 13 rows to the run below an output cell; the background
  kernel needs to reach UP 15 rows to see whether a cell falls in the relocated
  box.  params = 10*29 + 10 = 300, mem = 0 (the 10-ch output is FREE).
  Score ~19.30 vs the public 59-tap net's 18.60 (params 600).  All weights are
  integers (fp32-exact).  Verified ok + isolated fresh 200/200.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

L, PT, PB = 29, 15, 13   # kernel height, pad top, pad bottom (width pad = 0)

# Per-column colour kernel (channels 1,2,4): stamp the bottom run up by its height.
KB = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               -3., -47., 23., 11., 12., 7., 6., 5., 8., 0., 6., -2., 6., 1., 5.],
              dtype=np.float32)
BB = -21.0

# Per-column background kernel (channel 0): bg-after-transform, in-grid only.
KG = np.array([-54., 10., 1., 9., 0., 11., -1., 12., 3., 10., 1., 13., 0., 10.,
               -5., -35., 10., 11., 12., 13., 11., 12., 11., 4., 3., 2., 2., 1., 1.],
              dtype=np.float32)
BG = -33.0

BOX_CHANNELS = (1, 2, 4)


def build(task):
    W = np.zeros((10, 1, L, 1), dtype=np.float32)
    B = np.zeros((10,), dtype=np.float32)          # unused colour channels: 0 -> out 0

    W[0, 0, :, 0] = KG
    B[0] = BG
    for ch in BOX_CHANNELS:
        W[ch, 0, :, 0] = KB
        B[ch] = BB

    inits = [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")]
    nodes = [helper.make_node(
        "Conv", ["input", "W", "B"], ["output"],
        group=10,
        kernel_shape=[L, 1],
        pads=[PT, 0, PB, 0],   # [top, left, bottom, right]
        strides=[1, 1],
    )]

    F = TensorProto.FLOAT
    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task128", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
