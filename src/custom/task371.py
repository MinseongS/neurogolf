"""Task 371 (ARC-AGI e9614598): green plus at the midpoint of two blue dots.

Rule (orientation-equivariant, so xpose is handled for free): the input has
exactly two blue(1) cells, symmetric about a center. The output keeps both blue
cells and draws a green(3) plus (center + 4 orthogonal neighbors) at the center,
which is the centroid (mean position) of the two blue cells. Because the two
dots are symmetric about the center, the centroid IS the plus center in both the
horizontal (xpose=0) and vertical (xpose=1) orientations.

The center is exact in float32: there are always exactly 2 blue cells (space>=3
so they never coincide), and the row/col centroid sums divide by 2 to give an
integer coordinate (rows equal in one orientation; the two cols/rows are
symmetric in the other, summing to an even value).

Graph:
  blue = input channel 1 -> total = ReduceSum (==2)
  weighted row/col sums -> centroid_r, centroid_c (scalars)
  rowdist[i]=|i-cr|, coldist[j]=|j-cc|  ->  L1 dist[r,c]=rowdist+coldist
  plusmask = dist < 1.5  (center + 4 neighbors)
  output = Where(plusmask, green_onehot, input)   # blue is never covered
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits = []
    nodes = []
    vinfos = []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # blue = channel 1 of input -> [1,1,30,30]
    Wblue = np.zeros((1, 10, 1, 1), np.float32)
    Wblue[0, 1, 0, 0] = 1.0
    init("Wblue", Wblue)
    n("Conv", ["input", "Wblue"], "blue")
    vi("blue", TensorProto.FLOAT, [1, 1, 30, 30])

    # total count of blue cells (==2) -> [1,1,1,1]
    n("ReduceSum", ["blue"], "total", axes=[2, 3], keepdims=1)
    vi("total", TensorProto.FLOAT, [1, 1, 1, 1])

    # coordinate vectors
    rcoord = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    ccoord = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    init("rcoord", rcoord)
    init("ccoord", ccoord)

    # weighted sums: sum(r*blue), sum(c*blue)
    n("Mul", ["blue", "rcoord"], "rblue")
    vi("rblue", TensorProto.FLOAT, [1, 1, 30, 30])
    n("Mul", ["blue", "ccoord"], "cblue")
    vi("cblue", TensorProto.FLOAT, [1, 1, 30, 30])
    n("ReduceSum", ["rblue"], "rsum", axes=[2, 3], keepdims=1)
    vi("rsum", TensorProto.FLOAT, [1, 1, 1, 1])
    n("ReduceSum", ["cblue"], "csum", axes=[2, 3], keepdims=1)
    vi("csum", TensorProto.FLOAT, [1, 1, 1, 1])

    # centroid = sum / total  -> scalars [1,1,1,1]
    n("Div", ["rsum", "total"], "cr")
    vi("cr", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Div", ["csum", "total"], "cc")
    vi("cc", TensorProto.FLOAT, [1, 1, 1, 1])

    # rowdist[i] = |i - cr|  -> [1,1,30,1]; coldist[j] = |j - cc| -> [1,1,1,30]
    n("Sub", ["rcoord", "cr"], "rd0")
    vi("rd0", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Abs", ["rd0"], "rd")
    vi("rd", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Sub", ["ccoord", "cc"], "cd0")
    vi("cd0", TensorProto.FLOAT, [1, 1, 1, 30])
    n("Abs", ["cd0"], "cd")
    vi("cd", TensorProto.FLOAT, [1, 1, 1, 30])

    # L1 distance from center (broadcast outer sum) -> [1,1,30,30]
    n("Add", ["rd", "cd"], "dist")
    vi("dist", TensorProto.FLOAT, [1, 1, 30, 30])

    # plus mask = dist < 1.5  (covers center and its 4 orthogonal neighbors)
    init("th15", np.array(1.5, np.float32))
    n("Less", ["dist", "th15"], "plusmask")
    vi("plusmask", TensorProto.BOOL, [1, 1, 30, 30])

    # output = plusmask ? green_onehot : input.  Plus cells are always background
    # in the input (the blue dots sit at radius space>=3), so overwriting the
    # whole channel stack with the green one-hot is safe; blue is preserved.
    green = np.zeros((1, 10, 1, 1), np.float32)
    green[0, 3, 0, 0] = 1.0
    init("green", green)
    n("Where", ["plusmask", "green", "input"], "output")

    return _model(nodes, inits, vinfos)
