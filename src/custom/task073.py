"""task073 (ARC 3618c87e): blue tower-tops fall to the floor row.

Rule (5x5 grid at the canvas top-left, size fixed 5). The bottom row (row 4) is
all gray; each "tower" column c also has gray at row 3 and a single blue pixel
floating at row 2. The output keeps the gray towers and the gray floor, removes
the floating blue at row 2, and paints blue onto the floor cell (row 4) of every
tower column. Only colours 0 (bg), 1 (blue), 5 (gray) ever appear. Everything is
a purely VERTICAL, per-column, per-channel rearrangement -> one depthwise Conv
whose output IS the graph output (mem 0).

Per-channel vertical kernel (offset = source_row - target_row, out[r]=sum w*in[r+off]):
  ch1 (blue):  out[r] = in[r-2]                      -> single tap off -2
               (blue lives only at row 2, shifts to row 4; nothing feeds row 2 so
                the original auto-clears)
  ch5 (gray):  out[r] = in[r] - in[r-1]              -> taps off 0(+1), -1(-1)
               (gray at row 3 marks a tower; subtracting it removes the floor gray
                exactly under each tower, where blue now lands)
  ch0 (bg):    out[r] = -in[r-3] + in[r-2] + in[r]   -> taps off -3(-1), -2(+1), 0(+1)
               (re-creates bg at the vacated row-2 tower cells; far -3 tap keeps
                the SAME-pad edges correct)

The public net used a SAME-padded 9x1 depthwise conv (height 9 -> 10*9=90 params)
because its ch0/ch5 kernels reached offsets +/-4. The taps actually needed span
only offsets -3..0, so a height-4 kernel (SAME pad 3 top / 0 bottom) reproduces
the output exactly: params 10*4 = 40, mem 0. Verified exact on 2000 fresh
instances and brute-confirmed no shorter span (<=2) exists for ch0.
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def build(task):
    C = 10
    H = 4  # kernel rows; index ki maps to vertical offset (ki - 3): ki=0->-3 .. ki=3->0
    W = np.zeros((C, 1, H, 1), dtype=np.float32)

    def setw(ch, off, val):
        W[ch, 0, off + 3, 0] = val

    # ch1 (blue): shift down 2
    setw(1, -2, 1.0)
    # ch5 (gray): copy minus the tower-marker row above
    setw(5, 0, 1.0)
    setw(5, -1, -1.0)
    # ch0 (bg): restore vacated tower cells
    setw(0, -3, -1.0)
    setw(0, -2, 1.0)
    setw(0, 0, 1.0)

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    nodes = [
        helper.make_node(
            "Conv", ["input", "W"], ["output"],
            group=C, kernel_shape=[H, 1],
            # SAME-style vertical pad: 3 rows on top, 0 on bottom, so kernel row
            # ki contributes input row (r + ki - 3).
            pads=[3, 0, 0, 0], strides=[1, 1],
        )
    ]
    inits = [numpy_helper.from_array(W, "W")]
    graph = helper.make_graph(nodes, "task073", [inp], [out], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = 10
    return model
