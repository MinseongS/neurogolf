"""Task 335 — HPWL rectilinear L-path between two endpoints (separable Einsum).

Rule (common.hpwl): a black grid holds a red endpoint (channel 2) at
(rows[0], cols[0]) and a cyan endpoint (channel 8) at (rows[1], cols[1]).
The output redraws the grid and adds a yellow (channel 4) L-path:
  * horizontal segment on row rows[0], columns strictly between cols[0]/cols[1];
  * vertical segment on column cols[1], rows from rows[0] (inclusive) toward
    rows[1] (exclusive) — so the corner (rows[0], cols[1]) is yellow.
Background / red / cyan / yellow sets are pairwise disjoint, so the output is
the sum of five rank-1 (separable) row-x-col outer products, contracted against
a [term, channel] weight matrix by a single Einsum that writes the FREE
[1,10,30,30] output directly.

Endpoint coordinates come from a tiny [2,10] channel selector: one Einsum
contracts the red/cyan channels against the width axis (row profile), another
against the height axis (col profile); an ArgMax reads off each coordinate.
Both the selector and everything downstream stay tiny — the detector no longer
needs the 600-element pooling convs of the prior graph, and nothing ever
materialises a 30x30 plane.
"""
import numpy as np
from onnx import TensorProto, helper, numpy_helper
from ..harness import IR_VERSION


def _t(name, arr):
    return numpy_helper.from_array(np.ascontiguousarray(arr), name)


def build(task):
    # channel selector: term0 -> red (ch2), term1 -> cyan (ch8)
    sel = np.zeros((2, 10), np.float32); sel[0, 2] = 1.0; sel[1, 8] = 1.0

    # channel weight matrix W[n=1,t,c]  (t = 5 rank-1 terms); batch axis lets the
    # final Einsum emit [1,10,30,30] with no per-factor batch dim / Unsqueeze.
    W = np.zeros((1, 5, 10), np.float16)
    W[0, 0, 0] = 1                    # grid rectangle -> black
    W[0, 1, 4] = 1; W[0, 1, 0] = -1   # horizontal segment -> yellow, clear black
    W[0, 2, 4] = 1; W[0, 2, 0] = -1   # vertical segment   -> yellow, clear black
    W[0, 3, 2] = 1; W[0, 3, 0] = -1   # red (start) endpoint
    W[0, 4, 8] = 1; W[0, 4, 0] = -1   # cyan (end) endpoint

    coords = np.arange(30, dtype=np.int64).reshape(1, 30)

    inits = [
        _t('sel', sel),
        _t('Wmat', W),
        _t('coords', coords),
        _t('s0', np.array([0], np.int64)),
        _t('s1', np.array([1], np.int64)),
        _t('s2', np.array([2], np.int64)),
        _t('ax1', np.array([1], np.int64)),
        _t('ax13', np.array([1, 3], np.int64)),
        _t('ax12', np.array([1, 2], np.int64)),
        _t('zero_f', np.array([[0.0]], np.float32)),
    ]

    nodes = [
        # detection: red/cyan row & col profiles via channel-selecting Einsum
        helper.make_node('Einsum', ['input', 'sel'], ['rowp'], equation='bchw,tc->bth'),
        helper.make_node('Einsum', ['input', 'sel'], ['colp'], equation='bchw,tc->btw'),
        helper.make_node('ArgMax', ['rowp'], ['rows'], axis=2, keepdims=0),   # [1,2]
        helper.make_node('ArgMax', ['colp'], ['cols'], axis=2, keepdims=0),   # [1,2]
        helper.make_node('Slice', ['rows', 's0', 's1', 'ax1'], ['R0']),
        helper.make_node('Slice', ['rows', 's1', 's2', 'ax1'], ['R1']),
        helper.make_node('Slice', ['cols', 's0', 's1', 'ax1'], ['C0']),
        helper.make_node('Slice', ['cols', 's1', 's2', 'ax1'], ['C1']),
        helper.make_node('ReduceMin', ['rows'], ['rmin'], axes=[1], keepdims=1),
        helper.make_node('ReduceMax', ['rows'], ['rmax'], axes=[1], keepdims=1),
        helper.make_node('ReduceMin', ['cols'], ['cmin'], axes=[1], keepdims=1),
        helper.make_node('ReduceMax', ['cols'], ['cmax'], axes=[1], keepdims=1),
        # grid extent (in-grid rows/cols have nonzero channel mass) -> [1,30]
        helper.make_node('ReduceSum', ['input', 'ax13'], ['rowtot'], keepdims=0),  # [1,30]
        helper.make_node('ReduceSum', ['input', 'ax12'], ['coltot'], keepdims=0),  # [1,30]
        helper.make_node('Greater', ['rowtot', 'zero_f'], ['gridR']),
        helper.make_node('Greater', ['coltot', 'zero_f'], ['gridC']),
        # row terms
        helper.make_node('Equal', ['coords', 'R0'], ['eR0']),
        helper.make_node('Equal', ['coords', 'R1'], ['eR1']),
        helper.make_node('Greater', ['coords', 'rmin'], ['r_gt']),
        helper.make_node('Less', ['coords', 'rmax'], ['r_lt']),
        helper.make_node('And', ['r_gt', 'r_lt'], ['rowOpen']),
        # col terms
        helper.make_node('Equal', ['coords', 'C0'], ['eC0']),
        helper.make_node('Equal', ['coords', 'C1'], ['eC1']),
        helper.make_node('Greater', ['coords', 'cmin'], ['c_gt']),
        helper.make_node('Less', ['coords', 'cmax'], ['c_lt']),
        helper.make_node('And', ['c_gt', 'c_lt'], ['colOpen']),
        helper.make_node('Or', ['colOpen', 'eC1'], ['horizCol']),
        # assemble separable factors [5,30]; batch lives in Wmat
        helper.make_node('Concat', ['gridR', 'eR0', 'rowOpen', 'eR0', 'eR1'], ['Rb'], axis=0),
        helper.make_node('Concat', ['gridC', 'horizCol', 'eC1', 'eC0', 'eC1'], ['Cb'], axis=0),
        helper.make_node('Cast', ['Rb'], ['Rf'], to=TensorProto.FLOAT16),
        helper.make_node('Cast', ['Cb'], ['Cf'], to=TensorProto.FLOAT16),
        helper.make_node('Einsum', ['Wmat', 'Rf', 'Cf'], ['output'],
                         equation='ntc,th,tw->nchw'),
    ]
    graph = helper.make_graph(
        nodes, 'task335_einsum',
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT16, [1, 10, 30, 30])],
        inits,
    )
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid('', 13)])
