"""Task 084 - overlay-routing via a single ScatterElements into the free output.

Rule (exact, from generator task_3bd67248):
    The input is a square A x A grid (A in 3..21) whose column 0 holds a single
    colour `k` (k in 1..9) and whose other cells are background 0.  The output
    keeps column 0 unchanged and overlays:
        * red (2) on the anti-diagonal cells (A-1-c, c) for c = 1..A-1
        * yellow (4) on the bottom row cells (A-1, c) for c = 1..A-1
    Cells outside the A x A grid stay all-zero (the one-hot input is zero there).

Compression mechanism:
    The colour of the overlay depends only on A, never on `k`, and column 0 is
    already correct in `input`.  So the whole transform is a single
    ScatterElements(axis=2) that writes the overlay straight into a copy of the
    free `input` tensor.  No full [1,10,30,30] grid is ever materialised as a
    non-free intermediate; the only sizeable tensors are the compact
    (1,5,2,21) index / update planes.

    A (the grid size) is recovered from total_sum = A**2 (sum of the one-hot
    input) via the square-comparison trick, giving last_row = A-1.  For each
    column c the scatter writes, at two "slots":
        slot0 (anti-diagonal): clear channel 0 and set channel 2 at row A-1-c
        slot1 (bottom row):    clear channel 0 and set channel 4 at row A-1
    Row offsets are stored once as (1,1,2,21) and broadcast across the 5-channel
    axis (channels 0..4, the highest overlay channel being yellow=4) via a small
    ones plane scaled by last_row.  Columns c >= A are masked to write 0 (via
    valid21) so out-of-grid cells stay zero and negative wrapped rows are inert.
"""
import numpy as np

from ._exact import model, tensor


def build(task):
    # squares [900, 1, 4, 9, ..., 400]: cols_sq[k] < A**2  <=>  k in 1..A-1
    cols_sq = np.empty((21,), dtype=np.float32)
    cols_sq[0] = 900.0
    cols_sq[1:] = (np.arange(1, 21, dtype=np.float32)) ** 2

    # row offsets added to last_row (= A-1).  slot0 = anti-diagonal (row A-1-c),
    # slot1 = bottom row (row A-1).  The c=0 entries (-22, +9) deliberately send
    # the masked column-0 writes into padding rows so the coloured column 0 is
    # never touched.  Stored once (channel-independent) and broadcast.
    row_off = np.zeros((1, 1, 2, 21), dtype=np.int32)
    row_off[0, 0, 0, 0] = -22
    row_off[0, 0, 0, 1:] = -np.arange(1, 21, dtype=np.int32)  # -1, -2, ..., -20
    row_off[0, 0, 1, 0] = 9
    # row_off[0,0,1,1:] stays 0 -> bottom row

    # broadcast last_row into the 5-channel axis (kept as [1,5,1,1] so the
    # scatter's channel span is created by the Add against row_off).
    ones5 = np.ones((1, 5, 1, 1), dtype=np.int32)

    # which channel gets set at each overlay cell:
    #   slot0 (anti-diagonal) -> channel 2 (red), slot1 (bottom) -> channel 4 (yellow)
    chan_mask = np.zeros((1, 5, 2, 1), dtype=np.float32)
    chan_mask[0, 2, 0, 0] = 1.0
    chan_mask[0, 4, 1, 0] = 1.0

    inits = [
        tensor('cols_sq21_f', cols_sq),
        tensor('row_offsets_small_i32', row_off),
        tensor('ones5_i32', ones5),
        tensor('update_channel_mask', chan_mask),
    ]

    from onnx import helper
    nodes = [
        helper.make_node('ReduceSum', ['input'], ['total_sum'], axes=[0, 1, 2, 3], keepdims=0),
        helper.make_node('Less', ['cols_sq21_f', 'total_sum'], ['valid_col21']),
        helper.make_node('Cast', ['valid_col21'], ['valid21'], to=1),
        helper.make_node('ReduceSum', ['valid21'], ['last_row_f'], axes=[0], keepdims=0),
        helper.make_node('Cast', ['last_row_f'], ['last_row_i32'], to=6),
        helper.make_node('Mul', ['ones5_i32', 'last_row_i32'], ['last_row_plane']),  # [1,5,1,1]
        helper.make_node('Add', ['last_row_plane', 'row_offsets_small_i32'], ['indices']),  # [1,5,2,21]
        helper.make_node('Mul', ['update_channel_mask', 'valid21'], ['updates']),  # [1,5,2,21]
        helper.make_node('ScatterElements', ['input', 'indices', 'updates'], ['output'], axis=2),
    ]
    return model('task084_overlay_scatter', nodes, inits, output_dtype=1, opset=11, value_infos=[])
