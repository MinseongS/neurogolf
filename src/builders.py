"""ONNX graph builders for NeuroGolf networks (opset 10, IR 10).

All builders produce single-input ('input') single-output ('output') graphs
with statically-defined shapes. Tensors named 'input'/'output' are free in
the memory score, so we route data directly into 'output' wherever possible.
"""

import numpy as np
import onnx

from .harness import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS


def _model(nodes, initializers, value_infos=()):
    x = onnx.helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = onnx.helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = onnx.helper.make_graph(
        list(nodes), "graph", [x], [y], list(initializers),
        value_info=list(value_infos))
    return onnx.helper.make_model(
        graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)


def identity_network():
    """0 params, 0 memory -> 25.0 points."""
    node = onnx.helper.make_node("Identity", ["input"], ["output"])
    return _model([node], [])


def conv_network(weights, kh, kw):
    """Single no-bias Conv straight to 'output'. params = 100*kh*kw, 0 memory.

    weights: float array of shape [10, 10, kh, kw].
    """
    w = onnx.helper.make_tensor(
        "W", DATA_TYPE, [10, 10, kh, kw], np.asarray(weights, dtype=np.float32).flatten())
    pads = [kh // 2, kw // 2, kh // 2, kw // 2]
    node = onnx.helper.make_node(
        "Conv", ["input", "W"], ["output"], kernel_shape=[kh, kw], pads=pads)
    return _model([node], [w])
