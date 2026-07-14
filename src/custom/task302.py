"""Task 302 — 4-state single-plane collapse of the fill/frame decode.

Rule: gray(5) hollow square outlines of side L in {3,4,5} on a fixed 12x12 grid;
each hole is filled with colour 3+L, gray frames are kept, rest stays black(0).

Mechanism (vs the prior public net's 2-channel Concat + [10,2,5,5] final conv):
the whole decode collapses onto ONE uint8 plane carrying four states, read by a
single-input-channel [10,1,5,5] QLinearConv (250 params instead of 500) whose
zero_point supplies the fourth state for free.

  u    = 14 * gray                                   {0, 14}
  ring = clamp(gray_count_in_5x5_border - 15)        {0, 1}   -> L5 ring centre
  dsp  = clamp(5 * ring_count_in_3x3 + 7)            {7, 12}  -> spread over the
                                                                whole 3x3 hole
  m    = Max(u, dsp)                                 {7 black, 12 L5-hole, 14 gray}

The final conv reads `m` with x_zero_point=10, so a padded (off-grid) cell
contributes exactly 0 -- a fourth state distinct from all three in-grid ones.
That is what makes one plane sufficient: a 3-state plane is provably impossible
here (out-ch 8 needs black == off-grid to isolate the ring signal, out-ch 0 needs
them distinguishable), and the 3x3 spread of the ring centre is what breaks the
tie. Kernels below are exact MILP fits (integer, verified zero-error over all
8555 distinct 5x5 windows,
grown by a cutting-plane loop until 20000 fresh generator instances were clean).
"""
import numpy as np
from onnx import TensorProto, helper

from ._exact import model, tensor

# Exact integer hyperplanes over the 5x5 window of (m - 10); colours 1,2,3,4,9
# are never emitted, so their kernels are zero and their bias parks them off.
FINAL_W = [
    [5, 2, -14, 2, 5, 2, -17, 1, -17, 2, -14, 1, -127, 1, -14, 2, -17, 1, -17, 2, 5, 2, -14, 2, 5],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -1, 1, 0, 0, 0, -4, -2, 0, 9, 0, -6, 94, -9, 0, -7, 2, 4, -10, -18, 8, -8, -13, 15, 8],
    [127, -127, -127, -127, 127, -127, 127, 127, 127, -127, -127, 127, -127, 127, -127, -127, 127, 127, 127, -127, 127, -127, -127, -127, 127],
    [-35, -25, 118, -26, -34, -23, 123, -73, 125, -30, 117, -69, -5, -72, 120, -24, 126, -72, 127, -23, -33, -27, 123, -26, -33],
    [10, -19, 45, -19, 10, -17, 61, -16, 61, -18, 38, -7, 31, -7, 39, -1, 49, -4, 49, -1, -14, 2, 28, 1, -15],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
FINAL_B = [-187, -1, -1, -1, -1, -174, -5143, -1256, -608, -1]


def build(task):
    ring_w = np.zeros((1, 1, 5, 5), np.uint8)
    ring_w[0, 0, 0, :] = ring_w[0, 0, 4, :] = 1
    ring_w[0, 0, :, 0] = ring_w[0, 0, :, 4] = 1  # the 16 border cells of a 5x5 window

    inits = [
        tensor('starts', np.array([5, 0, 0], np.int64)),
        tensor('ends', np.array([6, 12, 12], np.int64)),
        tensor('axes', np.array([1, 2, 3], np.int64)),
        tensor('s13inv', np.array([1.0 / 14.0], np.float32)),
        tensor('zero_u8', np.array([0], np.uint8)),
        tensor('one_f32', np.array([1.0], np.float32)),
        tensor('ring_w', ring_w),
        tensor('s14', np.array([14.0], np.float32)),
        tensor('ring_b', np.array([-210], np.int32)),
        tensor('d_w', np.ones((1, 1, 3, 3), np.uint8)),
        tensor('s5', np.array([5.0], np.float32)),
        tensor('zp7', np.array([7], np.uint8)),
        tensor('d_b', np.array([0], np.int32)),
        tensor('zp10', np.array([10], np.uint8)),
        tensor('final_w', np.array(FINAL_W, np.int8).reshape(10, 1, 5, 5)),
        tensor('zero_i8', np.array([0], np.int8)),
        tensor('final_b', np.array(FINAL_B, np.int32)),
    ]
    nodes = [
        helper.make_node('Slice', ['input', 'starts', 'ends', 'axes'], ['ch5']),
        helper.make_node('QuantizeLinear', ['ch5', 's13inv', 'zero_u8'], ['u']),
        helper.make_node(
            'QLinearConv',
            ['u', 'one_f32', 'zero_u8', 'ring_w', 'one_f32', 'zero_u8', 's14', 'zero_u8', 'ring_b'],
            ['ring']),
        helper.make_node(
            'QLinearConv',
            ['ring', 'one_f32', 'zero_u8', 'd_w', 's5', 'zero_u8', 'one_f32', 'zp7', 'd_b'],
            ['dsp'], pads=[3, 3, 3, 3]),
        helper.make_node('Max', ['u', 'dsp'], ['m']),
        helper.make_node(
            'QLinearConv',
            ['m', 'one_f32', 'zp10', 'final_w', 'one_f32', 'zero_i8', 'one_f32', 'zero_u8', 'final_b'],
            ['output'], pads=[2, 2, 20, 20]),
    ]
    return model('task302_4state', nodes, inits, output_dtype=TensorProto.UINT8, opset=13)
