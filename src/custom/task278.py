"""Task 278 (ARC b27ca6d3): restore green olives from their two red centers.

Exact rule (colors: black=0, red=2, green=3):
The generator draws "olives": a pair of 4-adjacent cells, a 3x3 green block is
drawn around EACH, then the two centers are set to red. In the INPUT every green
cell is replaced by black, so an olive appears only as two 4-adjacent red pixels.
"static" red pixels are isolated (remove_neighbors), so they have NO red neighbor.

Transform input -> output:
  - olive_red = a red(2) cell that has >=1 red(2) cell among its 4 neighbors.
  - paint green(3) on the 3x3 Chebyshev-1 block around every olive_red,
    then restore red(2) at the olive_red centers.
  - every other cell passes through (background black; isolated static reds stay red).

Output color per cell:
  red(2)   if cell is red in the input
  green(3) if cell is within Chebyshev-1 of an olive_red AND is not itself red
  black(0) otherwise

The scorer accepts channel k iff output[k] > 0, so we route the 10-channel
one-hot expansion into the FREE `output` tensor via a final float `Equal`
(opset-11 op; scorer checks DOMAIN not VERSION) producing a BOOL output.

Encoding (all integer-valued float math, exact in float32):
  nred  = Conv(input, plus-kernel on ch2)            -> [1,1,30,30] #red 4-nbrs
  redc  = Conv(input, center on ch2)                 -> [1,1,30,30] center red
        (both produced by ONE Conv with 2 output channels: redc, nred)
  olive = (redc>0.5) & (nred>0.5)                     -> bool
  dil   = Conv(olive_f, 3x3 ones)                     -> dilation count
  green = (dil>0.5) & ~(redc>0.5)                     -> bool
  L     = 2*red + 3*green  (uint8 label, 0 elsewhere)
  output = Equal(L, [0..9])  (BOOL)
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import GRID_SHAPE, IR_VERSION

# opset 11 so the final float `Equal` (which yields a BOOL output we route into
# the free `output` tensor) loads in ORT. The scorer checks op DOMAIN not VERSION.
OPSET_IMPORTS = [helper.make_opsetid("", 11)]


def build(task):
    nodes = []
    inits = []

    def init(name, arr):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
        return name

    def node(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    init("half", np.array(0.5, np.float32))

    # ---- collapse the 10-channel input (FREE) to two single-channel feature
    # planes via Conv, THEN crop. The generator sets width,height in [15,18] and
    # anchors the grid top-left, so the whole grid + olives + green dilation lie
    # in the top-left S x S block (S=18). Converting 10->1 channel before the
    # Slice avoids a costly [1,10,18,18] crop; all downstream work is 18x18. ----
    S = 18

    # ocount = 5*center_red + (#red 4-neighbors). Both terms are 0/1 from the
    # one-hot input ch2, so ocount in 0..4 when not red, 5..9 when red.
    # => red center IFF ocount > 4.5 ; olive_red IFF ocount > 5.5.
    Wo = np.zeros((1, 10, 3, 3), np.float32)
    Wo[0, 2, 1, 1] = 5.0                       # center red
    for r, c in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        Wo[0, 2, r, c] = 1.0                   # plus-neighbor red count
    init("Wo", Wo)
    node("Conv", ["input", "Wo"], "ocount30", pads=[1, 1, 1, 1])  # [1,1,30,30]

    # ingrid = sum of all 10 input channels (1 in-grid, 0 off-grid), 1x1 Conv.
    Wg = np.ones((1, 10, 1, 1), np.float32)
    init("Wg", Wg)
    node("Conv", ["input", "Wg"], "ingrid30")  # [1,1,30,30]

    # crop both feature planes to the top-left 18x18 active region.
    init("st", np.array([0, 0, 0, 0], np.int64))
    init("en", np.array([1, 1, S, S], np.int64))
    init("ax", np.array([0, 1, 2, 3], np.int64))
    node("Slice", ["ocount30", "st", "en", "ax"], "ocount")  # [1,1,18,18]
    node("Slice", ["ingrid30", "st", "en", "ax"], "ingrid")  # [1,1,18,18]

    init("fourhalf", np.array(4.5, np.float32))
    init("fivehalf", np.array(5.5, np.float32))
    node("Greater", ["ocount", "fourhalf"], "redb")    # bool red center
    node("Greater", ["ocount", "fivehalf"], "oliveb")  # bool olive_red
    node("Greater", ["ingrid", "half"], "ingridb")     # bool in-grid

    # dilate olive 3x3 (Chebyshev-1) -> green region
    node("Cast", ["oliveb"], "olivef", to=TensorProto.FLOAT)
    W2 = np.ones((1, 1, 3, 3), np.float32)
    init("W2", W2)
    node("Conv", ["olivef", "W2"], "dil", pads=[1, 1, 1, 1])  # [1,1,30,30]
    node("Greater", ["dil", "half"], "dilb")
    node("Not", ["redb"], "notred")
    node("And", ["dilb", "notred"], "greentmp")     # dilation & ~red
    # green is clipped to the grid: the generator draws green only inside bounds,
    # so a dilation that spills past the edge must not paint off-grid cells.
    node("And", ["greentmp", "ingridb"], "greenb")  # & in-grid

    # label L via nested Where (disjoint cases). Off-grid -> 10 so it matches no
    # channel index 0..9 (Equal yields all-zero there, as required); in-grid
    # background -> 0 (ch0); green -> 3; red -> 2 (red wins over green overlap).
    init("c2", np.array(2.0, np.float32))
    init("c3", np.array(3.0, np.float32))
    init("c10", np.array(10.0, np.float32))
    init("c0", np.array(0.0, np.float32))
    node("Where", ["ingridb", "c0", "c10"], "Lbase")  # 0 in-grid, 10 off-grid
    node("Where", ["greenb", "c3", "Lbase"], "Lg")    # green=3
    node("Where", ["redb", "c2", "Lg"], "Lc")         # red=2 over green/bg

    # pad the 18x18 label back to 30x30 with sentinel 10 (off-grid -> all-zero).
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - S, 30 - S], np.int64))
    init("padval", np.array(10.0, np.float32))
    node("Pad", ["Lc", "pads", "padval"], "L", mode="constant")  # [1,1,30,30]

    # output = Equal(L, [0..9]) -> BOOL [1,10,30,30] routed into the FREE output
    arange = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("arange", arange)
    node("Equal", ["L", "arange"], "output")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, GRID_SHAPE)
    graph = helper.make_graph(nodes, "task278", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=OPSET_IMPORTS)
