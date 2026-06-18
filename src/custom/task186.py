"""task186 (ARC-AGI 794b24be) — encode blue pixel COUNT as a fixed red-pixel pattern.

Rule (from the ARC-GEN generator, verified fresh):
  Input: a 3x3 grid with `count` (1..4) blue (colour 1) pixels at random positions.
  Output: a 3x3 grid whose red (colour 2) cells form a FIXED count-thermometer:
      out[0][0] red  if count >= 1
      out[0][1] red  if count >= 2
      out[0][2] red  if count >= 3
      out[1][1] red  if count >= 4
  every other cell is black (colour 0).

This is the COUNT->FIXED-PATTERN lever: the ENTIRE output is determined by ONE
scalar (the blue pixel count, range 1..4 => 4 possible outputs).

Encoding (no [1,10,H,W] plane ever materialised):
  count = ReduceSum over the channel-1 (blue) slice of the active 3x3 grid.
  idx   = count - 1  (0..3).
  A constant bank_red[4, 1, 3, 3] uint8 holds ONLY the red (channel-2) pattern for
  each count; Gather(axis=0) by the scalar idx yields red[1,1,3,3].
    channel 2 (red)   = red
    channel 0 (black) = Equal(red, 0)  (1 where grid cell is not red)
    channel 1 (blue)  = constant zeros
  Concat([ch0, ch1, red], axis=1) -> [1,3,3,3] uint8, then one uint8 Pad zero-fills
  the 7 trailing colour channels AND the 30x30 spatial border (its output IS the
  graph output, FREE). The harness scores (out>0), so uint8 {0,1} passes identically.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

S = 30  # full canvas
G = 3   # active grid size


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        npd = helper.tensor_dtype_to_np_dtype(dt)
        a = np.ascontiguousarray(arr, dtype=npd)
        inits.append(numpy_helper.from_array(a, name))
        return name

    def n(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # --- bank_red[count][1][r][c]: red (channel-2) pattern per count ---
    red_bank = np.zeros((4, 1, G, G), dtype=np.uint8)
    red_cells = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
    ]
    for ci, cells in enumerate(red_cells):
        for (r, c) in cells:
            red_bank[ci, 0, r, c] = 1
    init("red_bank", red_bank, TensorProto.UINT8)

    # --- count = sum of blue (channel 1) over the 3x3 grid ---
    init("blue_starts", np.array([1, 0, 0], dtype=np.int64), TensorProto.INT64)
    init("blue_ends", np.array([2, G, G], dtype=np.int64), TensorProto.INT64)
    init("blue_axes", np.array([1, 2, 3], dtype=np.int64), TensorProto.INT64)
    n("Slice", ["input", "blue_starts", "blue_ends", "blue_axes"], ["blue"])
    n("ReduceSum", ["blue"], ["count_f"], keepdims=0)  # scalar fp32

    init("ONE_F", np.array(1.0, dtype=np.float32), TensorProto.FLOAT)
    n("Sub", ["count_f", "ONE_F"], ["idx_f"])
    n("Cast", ["idx_f"], ["idx"], to=TensorProto.INT32)  # shape [1] idx 0..3

    # Gather along count axis -> red [1, 1, 3, 3] uint8
    n("Gather", ["red_bank", "idx"], ["red"], axis=0)

    # channel 0 (black) = 1 where not red
    init("zero_u8", np.array(0, dtype=np.uint8), TensorProto.UINT8)
    n("Equal", ["red", "zero_u8"], ["ch0_b"])
    n("Cast", ["ch0_b"], ["ch0"], to=TensorProto.UINT8)

    # channel 1 (blue) = zeros [1,1,3,3]
    init("ch1", np.zeros((1, 1, G, G), dtype=np.uint8), TensorProto.UINT8)

    # assemble leading 3 channels
    n("Concat", ["ch0", "ch1", "red"], ["small_out"], axis=1)

    # Pad: 7 trailing colour channels + spatial border to 30x30
    pads = np.array([0, 0, 0, 0, 0, 7, S - G, S - G], dtype=np.int64)
    init("pads", pads, TensorProto.INT64)
    n("Pad", ["small_out", "pads", "zero_u8"], ["output"], mode="constant")

    out_vi = helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, S, S])
    in_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, S, S])
    graph = helper.make_graph(nodes, "task186", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = IR_VERSION
    return model
