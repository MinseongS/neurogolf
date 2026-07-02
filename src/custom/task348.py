"""Task 348 — inverted-triangle paint via a compact uint8 label grid.

The output is a downward-pointing triangle of alternating orange(7)/cyan(8)
stripes (by column parity) over the black grid, anchored on the orange line.

Instead of materializing a [1,9,10,10] one-hot bool carrier (900 B) plus the
per-channel masks that feed its Concat, we compute a single [1,1,10,10] uint8
*label* grid whose cells hold the color value (0=black, 7=orange, 8=cyan, 250=
out-of-grid sentinel), pad it to [1,1,30,30] (padding + sentinel -> no channel),
and expand to the [1,10,30,30] one-hot ONLY as the last op via
`Equal(label, channel_ids[10,1,1])`, which writes straight to the FREE "output"
tensor. Two Where levels fold three disjoint masks into the label:
  stripe = orange(7)/cyan(8) by column parity; base = 0 inside grid else 250;
  label  = inside ? stripe : base.
Detection (orange line column via first non-black cell in row 0; triangle
length via the orange tail sum) is unchanged.
"""
import numpy as np
from onnx import TensorProto, helper, numpy_helper
from ..harness import IR_VERSION


def _t(name, arr):
    return numpy_helper.from_array(np.ascontiguousarray(arr), name)


def build(task):
    inits = [
        _t('axes_chw', np.array([1, 2, 3], np.int64)),
        _t('ch0_st', np.array([0, 0, 0], np.int64)),
        _t('ch0_row0_en', np.array([1, 1, 10], np.int64)),
        _t('ch0_col0_en', np.array([1, 10, 1], np.int64)),
        _t('ch7_st', np.array([7, 3, 2], np.int64)),
        _t('ch7_en', np.array([8, 9, 8], np.int64)),
        _t('pad_to_30', np.array([0, 0, 0, 0, 0, 0, 20, 20], np.int64)),
        _t('pos_r', np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7], np.float32).reshape(1, 1, 10, 1)),
        _t('pos_c', np.arange(10, dtype=np.float32).reshape(1, 1, 1, 10)),
        _t('two_i64', np.array([2], np.int16)),
        _t('col_parity_i64', np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], np.int16).reshape(1, 1, 1, 10)),
        # per-cell color labels (uint8): orange=7, cyan=8, out-of-grid sentinel=250.
        _t('u8_7', np.array(7, np.uint8)),
        _t('u8_8', np.array(8, np.uint8)),
        _t('u8_250', np.array(250, np.uint8)),
        _t('u8_0', np.array(0, np.uint8)),
        _t('pad_val', np.array(250, np.uint8)),
        _t('chan_ids', np.arange(10, dtype=np.uint8).reshape(10, 1, 1)),
    ]
    nodes = [
        helper.make_node('Slice', ['input', 'ch0_st', 'ch0_row0_en', 'axes_chw'], ['ch0_row0_f']),
        helper.make_node('Slice', ['input', 'ch0_st', 'ch0_col0_en', 'axes_chw'], ['ch0_col0_f']),
        helper.make_node('Slice', ['input', 'ch7_st', 'ch7_en', 'axes_chw'], ['ch7_tail_6x7']),
        helper.make_node('Cast', ['ch0_row0_f'], ['ch0_row0_b'], to=9),
        helper.make_node('Cast', ['ch0_col0_f'], ['valid_rows_b'], to=9),
        helper.make_node('ReduceSum', ['ch7_tail_6x7'], ['tail_f'], keepdims=1),
        helper.make_node('ArgMin', ['ch0_row0_f'], ['local_col_i64_i64'], axis=3, keepdims=1),
        helper.make_node('Cast', ['local_col_i64_i64'], ['local_col_i64'], to=5),
        helper.make_node('Cast', ['local_col_i64'], ['local_col_f'], to=1),
        helper.make_node('Equal', ['pos_c', 'local_col_f'], ['col_has_b']),
        helper.make_node('Or', ['ch0_row0_b', 'col_has_b'], ['valid_cols_b']),
        helper.make_node('And', ['valid_rows_b', 'valid_cols_b'], ['valid_grid_b']),
        helper.make_node('Sub', ['tail_f', 'pos_r'], ['half_width_f']),
        helper.make_node('Sub', ['pos_c', 'local_col_f'], ['dx_f']),
        helper.make_node('Abs', ['dx_f'], ['dist_f']),
        helper.make_node('GreaterOrEqual', ['half_width_f', 'dist_f'], ['inside_raw_b']),
        helper.make_node('And', ['inside_raw_b', 'valid_cols_b'], ['inside_b']),
        helper.make_node('Mod', ['local_col_i64', 'two_i64'], ['center_parity_i64'], fmod=0),
        helper.make_node('Equal', ['col_parity_i64', 'center_parity_i64'], ['even_b']),
        # Build the uint8 label grid with two Where levels:
        #   stripe: per-column triangle color (orange 7 on even, cyan 8 on odd),
        #   base:   valid black cell -> 0, out-of-grid -> 250 (sentinel),
        #   label:  inside cells take the stripe color, else the base.
        helper.make_node('Where', ['even_b', 'u8_7', 'u8_8'], ['stripe']),
        helper.make_node('Where', ['valid_grid_b', 'u8_0', 'u8_250'], ['base']),
        helper.make_node('Where', ['inside_b', 'stripe', 'base'], ['label_10']),
        helper.make_node('Pad', ['label_10', 'pad_to_30', 'pad_val'], ['label_30'], mode='constant'),
        # Expand to one-hot directly onto the free "output" tensor.
        helper.make_node('Equal', ['label_30', 'chan_ids'], ['output']),
    ]
    graph = helper.make_graph(
        nodes, 'task348_label',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info('output', 9, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid('', 13)])
