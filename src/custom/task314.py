"""task314: per-channel midpoint-fill on a spacing-3 sublattice.

Rule (per colour channel, independent): the 8x8 grid carries marked cells on a
3x3 macro lattice (cell spacing 3). For every colour, within its sub-lattice a
middle cell is filled iff BOTH of its row-neighbours at +/-3 are present (vertical
fill) OR both col-neighbours at +/-3 are present (horizontal fill); original cells
are copied. This is a purely per-channel LOCAL op touching only offsets {0,+/-3}.

The public net used a dense 7x7 depthwise conv (490+10=500 params). Offsets only
ever reach +/-3, so a 3x3 kernel DILATED by 3 covers exactly {-3,0,+3} in each
axis -> footprint identical, kernel 10*1*3*3=90 + bias 10 = 100 params. mem=0
(output is the graph output).

Kernel (per fg channel, same cross trick as public net): center=+4, edge(+/-3
axis)=+2, corner(+/-3,+/-3)=-2, bias=-3. Center cell on alone: 4-3=1>0 (copy).
Middle with both endpoints: 2+2-3=1>0 (fill). One endpoint only: 2-3=-1<0. The
-2 corners also turn the blue background OFF at a newly-filled cell (its four
diagonal +/-3 neighbours are blue), and suppress diagonal false positives.
Channel 0 (black grid lines): copy-only (center=+2,bias=-1), since a line cell's
+/-3 neighbour is usually another line cell and would false-fill.
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def build(task):
    C = 10
    # 3x3 kernel, dilation 3 -> taps at row/col offsets {-3,0,+3}.
    W = np.zeros((C, 1, 3, 3), dtype=np.float32)
    B = np.full((C,), -3.0, dtype=np.float32)
    for ch in range(1, C):
        W[ch, 0, 1, 1] = 4.0      # center (offset 0,0)
        W[ch, 0, 0, 1] = 2.0      # (-3, 0)
        W[ch, 0, 2, 1] = 2.0      # (+3, 0)
        W[ch, 0, 1, 0] = 2.0      # (0, -3)
        W[ch, 0, 1, 2] = 2.0      # (0, +3)
        W[ch, 0, 0, 0] = -2.0     # (-3,-3)
        W[ch, 0, 0, 2] = -2.0     # (-3,+3)
        W[ch, 0, 2, 0] = -2.0     # (+3,-3)
        W[ch, 0, 2, 2] = -2.0     # (+3,+3)
    # channel 0 (black grid lines): copy-only, no fill (a line cell's +/-3
    # neighbour is often another line cell -> would false-fill).
    W[0, 0, 1, 1] = 2.0
    B[0] = -1.0

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    nodes = [
        helper.make_node(
            "Conv", ["input", "W", "Bb"], ["output"],
            group=C, kernel_shape=[3, 3], dilations=[3, 3],
            pads=[3, 3, 3, 3], strides=[1, 1],
        )
    ]
    inits = [
        numpy_helper.from_array(W, "W"),
        numpy_helper.from_array(B, "Bb"),
    ]
    graph = helper.make_graph(nodes, "task314", [inp], [out], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = 10
    return model
