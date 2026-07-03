"""Task162 candidate — free-input einsum box-count (kills the f32 occupancy slice),
u8 dilation + Pad + Where routing.

STATUS: NOT LANDABLE (documented negative / lever-1 demonstration).
  measured: mem 3320 + params 568 = 3888  (vs incumbent 4024+44 = 4068)  -> 16.734 pts
  fresh gate: 2000/2000 fresh, candidate fail = 0, candidate != incumbent = 0  (BIT-IDENTICAL
              to the incumbent on the entire real generator distribution).
  bundled gate: FAIL = 1 (train example #2).  Bundled == public LB, and any single fail
              zeros the task, so this is NOT landable despite the lower cost.

Why it is cheaper: it demonstrates dossier LEVER 1 -- the incumbent pays a [1,1,20,20]
f32 Slice (1600B) + u8 Cast (400B) just to materialize the black occupancy for a conv.
Here the black count per 3x3 corner is read straight off the FREE graph input by one
Einsum (all operands free; only the f32 count plane 1296B is counted), so the 2000B
occupancy pair vanishes.

Why it FAILS bundled: the incumbent's real value is a single FUSED, NON-SEPARABLE 4x4
QLinearConv that does detection AND row-major causal suppression together (u8 score,
324B), which is only possible because it has the occupancy plane to convolve.  A
free-input Einsum is a rank-1 separable bilinear form: it can compute the box-count
(detection) but CANNOT reproduce the rank-3 8/9-core + L-border suppression kernel, and
re-adding suppression as separate ops costs more planes than the occupancy it saved.
On the real generator the generator's own unambiguity filter removes every overlapping
hole, so no suppression is ever needed (H0 == oracle, 0/4000) and this net is fresh-exact;
but original-ARC bundled example #2 has a real vertical+horizontal hole overlap that
requires the suppression, so it over-paints the trailing row/col there.

Conclusion (see task report): the incumbent sits at the structural floor; neither dossier
lever is realizable without either breaking correctness (this file) or increasing cost.
"""
import numpy as np
from onnx import TensorProto, helper
from ._exact import model, tensor


def build(task):
    # band matrix B[r,H] = 1 iff H in {r, r+1, r+2}; corners r in 0..17, input axis H in 0..29
    # (cols 20..29 are off-grid, always 0 in the input, so they stay 0 here)
    B = np.zeros((18, 30), np.float32)
    for r in range(18):
        B[r, r:r + 3] = 1.0
    e0 = np.zeros(10, np.float32); e0[0] = 1.0            # select black channel
    blue = np.zeros((1, 10, 1, 1), np.float32); blue[0, 1, 0, 0] = 1.0

    inits = [
        tensor('e0', e0),
        tensor('B', B),
        tensor('one1', np.ones(1, np.float32)),
        tensor('thr8', np.uint8(8)),
        tensor('pads_hw', np.array([0, 0, 10, 10], np.int64)),
        tensor('pad_axes', np.array([2, 3], np.int64)),
        tensor('blue_pixel', blue),
    ]
    nodes = [
        # black count in each 3x3 window (corner r,c) directly from the free input;
        # size-1 'z' keeps the output 4D [1,1,18,18] for MaxPool.
        helper.make_node('Einsum', ['input', 'e0', 'B', 'B', 'one1'], ['cnt'],
                         equation='bchw,c,rh,sw,z->bzrs'),
        helper.make_node('Cast', ['cnt'], ['cnt_u8'], to=2),          # 0..9 -> u8
        # dilate counts so a fired corner paints its whole 3x3 block; expand 18->20
        helper.make_node('MaxPool', ['cnt_u8'], ['maxcnt'],
                         kernel_shape=[3, 3], pads=[2, 2, 2, 2], strides=[1, 1]),
        helper.make_node('Greater', ['maxcnt', 'thr8'], ['block20']),  # >=9  <=> all-black present
        helper.make_node('Pad', ['block20', 'pads_hw', '', 'pad_axes'], ['block30'], mode='constant'),
        helper.make_node('Where', ['block30', 'blue_pixel', 'input'], ['output']),
    ]
    value_infos = [
        helper.make_tensor_value_info('cnt', TensorProto.FLOAT, [1, 1, 18, 18]),
        helper.make_tensor_value_info('cnt_u8', TensorProto.UINT8, [1, 1, 18, 18]),
        helper.make_tensor_value_info('maxcnt', TensorProto.UINT8, [1, 1, 20, 20]),
        helper.make_tensor_value_info('block20', TensorProto.BOOL, [1, 1, 20, 20]),
        helper.make_tensor_value_info('block30', TensorProto.BOOL, [1, 1, 30, 30]),
    ]
    return model('task162_signed', nodes, inits, output_dtype=1, opset=18, value_infos=value_infos)
