"""Task 108 (ARC-AGI 46f33fce) -- 4x block-upscale of an odd-cell sublattice.

Rule (from generator): input is a 10x10 grid (placed top-left of the 30x30
canvas).  Colored pixels live ONLY at odd cells (2r+1, 2c+1) for r,c in [0..4].
Each such pixel of colour `k` becomes a 4x4 block in the 20x20 output:
    output[4r:4r+4, 4c:4c+4] = k.
Equivalently, for EVERY channel (background ch0 included):
    output[i,j] = input[2*(i//4)+1, 2*(j//4)+1]   for i,j in [0,20)
    output[i,j] = 0                                for i>=20 or j>=20.

Single mem-0 op: a (group=1) ConvTranspose written directly into `output`.

Geometry (depthwise-style, stride 2, kernel 4x4 of ones, asymmetric crop):
  output index for input cell `a` (odd) = a*2 - pad_top + kt, kt in [0..3].
  pad_top=2  =>  odd cell a=2r+1 lands at block [4r .. 4r+3]   (exactly right).
  pads=[2,2,30,30] crops the (otherwise 58x58) ConvTranspose output to the
  top-left 30x30, keeping the content block at the top-left and discarding the
  all-zero bottom/right.  Because the *colour* channels (1..9) are zero at every
  even input cell, the even-cell contributions cancel and ch1..9 come out exact.

ch0 (background) FIX -- the contamination problem:
  In channel 0 the *even* input cells are 1 (background), so a plain depthwise
  upscale would leak ch0=1 into cells where a colour actually sits (the
  contaminated ch0 is a strict superset of the truth: extra 1s exactly at colour
  cells).  A single depthwise kernel cannot remove this (the odd-cell tap and the
  even-cell tap collide at the same kernel position).  So we use a group=1
  ConvTranspose [C_in=10, C_out=10, 4, 4] whose ch0 output also reads the colour
  channels and SUBTRACTS them:
      W[0,0]  = +1 (4x4 ones)          ch0 self upscale (superset)
      W[k,k]  = +1 (4x4 ones), k=1..9  colour self upscale (exact)
      W[k,0]  = -10 (4x4 ones), k=1..9 push ch0 negative wherever a colour sits
  After (out>0) thresholding, ch0 is 1 only on in-region background cells -> exact.

Cost: mem 0 (the op writes `output`); params = 10*10*4*4 = 1600 (dense weight;
the off-diagonal zeros are still counted -- sparse initializers are unusable here
because the harness sanitizer does not remap sparse-initializer names).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT


def build(task):
    # ConvTranspose weight layout: [C_in, C_out / group, kh, kw], group=1.
    W = np.zeros((10, 10, 4, 4), np.float32)
    W[0, 0] = 1.0
    for k in range(1, 10):
        W[k, k] = 1.0      # colour self-upscale (exact -- even cells are 0)
        W[k, 0] = -10.0    # subtract colours from ch0 (kills the bg leak)

    inits = [numpy_helper.from_array(W, "ctw")]
    nodes = [helper.make_node(
        "ConvTranspose", ["input", "ctw"], ["output"],
        group=1, strides=[2, 2], pads=[2, 2, 30, 30],
        output_padding=[0, 0], kernel_shape=[4, 4])]

    graph = helper.make_graph(
        nodes, "task108",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = IR_VERSION
    return model
