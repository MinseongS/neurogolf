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


def memorizer_network(Z, R, O, k_proj):
    """Exact-match lookup network.

    Pipeline (all integer-valued float math, exact in float32):
      input one-hot -> base-11 cell codes packed 4/cell (Conv stride 4)
      -> random +-1 projection to k dims -> mismatch-count vs stored Z [N,k]
      -> one-hot row selector -> MatMul with stored outputs O [N,180]
      (base-11^5 packed) -> arithmetic unpack -> Equal-decode to one-hot.

    Z: float [N, k] projected stored inputs; R: float [240, k] +-1 projection;
    O: float [N, 180] packed stored outputs.
    """
    N = Z.shape[0]
    nodes, inits = [], []

    def init(name, arr, dtype=np.float32):
        t = onnx.numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name)
        inits.append(t)
        return name

    def scalar(name, value):
        return init(name, np.array(value, dtype=np.float32))

    def n(op, inputs, out, **attrs):
        nodes.append(onnx.helper.make_node(op, inputs, [out], **attrs))
        return out

    # --- encode input to packed base-11 codes ---
    # pack4: kernel [1,10,1,4], stride 4, pad right 2 -> [1,1,30,8] (240 codes)
    w4 = np.zeros((1, 10, 1, 4), np.float32)
    for c in range(10):
        for j in range(4):
            w4[0, c, 0, j] = (c + 1) * 11 ** (3 - j)
    init("w4", w4)
    n("Conv", ["input", "w4"], "p4", kernel_shape=[1, 4], strides=[1, 4],
      pads=[0, 0, 0, 2])
    init("shape_flat4", np.array([1, 240], np.int64), np.int64)
    n("Reshape", ["p4", "shape_flat4"], "x4")  # [1,240]

    # --- match against stored inputs via exact projection ---
    init("R", R)
    n("MatMul", ["x4", "R"], "z")              # [1,k]
    init("Z", Z)
    n("Sub", ["Z", "z"], "d")                  # [N,k]
    n("Abs", ["d"], "da")
    n("Clip", ["da"], "dc", min=0.0, max=1.0)
    n("ReduceSum", ["dc"], "mc", axes=[1], keepdims=1)   # [N,1] mismatch count
    n("Clip", ["mc"], "mcc", min=0.0, max=1.0)
    scalar("one", 1.0)
    n("Sub", ["one", "mcc"], "sel")            # [N,1], 1 iff exact match
    n("Transpose", ["sel"], "selr", perm=[1, 0])  # [1,N]

    # --- select stored output and unpack base-11^5 digits ---
    init("O", O)
    n("MatMul", ["selr", "O"], "yp")           # [1,180]
    init("shape_grid", np.array([1, 1, 30, 6], np.int64), np.int64)
    n("Reshape", ["yp", "shape_grid"], "y")    # [1,1,30,6]
    rem = "y"
    digits = []
    for j in range(4):
        p = 11 ** (4 - j)
        scalar(f"pw{j}", float(p))
        n("Div", [rem, f"pw{j}"], f"q{j}_raw")
        n("Floor", [f"q{j}_raw"], f"q{j}")
        digits.append(f"q{j}")
        n("Mul", [f"q{j}", f"pw{j}"], f"qm{j}")
        rem = n("Sub", [rem, f"qm{j}"], f"r{j}")
    digits.append(rem)                          # last digit, [1,1,30,6]
    for d in digits:
        n("Unsqueeze", [d], f"{d}_u", axes=[4])
    nodes.append(onnx.helper.make_node(
        "Concat", [f"{d}_u" for d in digits], ["dig"], axis=4))  # [1,1,30,6,5]
    init("shape_code", np.array([1, 1, 30, 30], np.int64), np.int64)
    n("Reshape", ["dig", "shape_code"], "code")

    # --- decode codes (color c <-> code c+1, empty 0) to one-hot ---
    # Equal doesn't take floats until opset 11, so compare as int32.
    n("Cast", ["code"], "code_i", to=onnx.TensorProto.INT32)
    init("chvec", np.arange(1, 11, dtype=np.int32).reshape(1, 10, 1, 1), np.int32)
    n("Equal", ["code_i", "chvec"], "eq")      # bool [1,10,30,30]
    n("Cast", ["eq"], "output", to=DATA_TYPE)
    return _model(nodes, inits)


def pack4_codes(grid_canvas):
    """numpy mirror of the in-graph pack4 encoder. grid_canvas: int [30,30] codes 0..10."""
    padded = np.zeros((30, 32), np.int64)
    padded[:, :30] = grid_canvas
    w = 11 ** np.arange(3, -1, -1)
    return (padded.reshape(30, 8, 4) * w).sum(axis=2).reshape(-1)  # [240]


def pack5_codes(grid_canvas):
    w = 11 ** np.arange(4, -1, -1)
    return (grid_canvas.reshape(30, 6, 5) * w).sum(axis=2).reshape(-1)  # [180]


def conv_network(weights, kh, kw, bias=None):
    """Single Conv straight to 'output'. params = 100*kh*kw (+10 bias), 0 memory.

    weights: float array of shape [10, 10, kh, kw]; bias: optional [10].
    """
    w = onnx.helper.make_tensor(
        "W", DATA_TYPE, [10, 10, kh, kw], np.asarray(weights, dtype=np.float32).flatten())
    inits, inputs = [w], ["input", "W"]
    if bias is not None:
        inits.append(onnx.helper.make_tensor(
            "B", DATA_TYPE, [10], np.asarray(bias, dtype=np.float32).flatten()))
        inputs.append("B")
    pads = [kh // 2, kw // 2, kh // 2, kw // 2]
    node = onnx.helper.make_node(
        "Conv", inputs, ["output"], kernel_shape=[kh, kw], pads=pads)
    return _model([node], inits)
