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
one-hot expansion into the FREE `output` tensor via a final `Equal`
(opset-11 op; scorer checks DOMAIN not VERSION) producing a BOOL output.

Encoding (all integer-valued math, exact). The generator sets width,height in
[15,18] and anchors the grid top-left, so the whole grid + olives + the green
3x3 dilation lie in the top-left 18x18 block.  We SLICE the FREE input to its two
populated colour channels {0,2} over that 18x18 block FIRST (the only colours that
ever appear are black ch0 and red ch2 -- green was erased to black), then run a
single Conv on that tiny [1,2,18,18] block.  This avoids the full [1,1,30,30]
3600B Conv plane (the previous version's dominant cost).

  ocount = Conv(input{0,2}[18x18])  ONE plane carrying in-grid (band 100),
           red (band 500), olive (red + red-neighbor, band 501+) AND the
           red-neighbor count -- the three signals sit in disjoint magnitude
           bands so a single plane separates them by thresholds.
  ingrid/red/olive  = three thresholds on the single ocount plane.
  dil    = Conv(olive, 3x3 ones)  (fp16)   -> green = dil & in-grid.
  L      = uint8 label (red=2, green=3, bg=0, off-grid=10), built by Wheres.
  L      = Pad(L, 30x30, value=10)         off-grid sentinel -> all-zero output.
  output = Equal(L, [0..9])  (BOOL)        the 10-ch expansion lands in free output.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import GRID_SHAPE, IR_VERSION

# opset 11 so the final float `Equal` (which yields a BOOL output we route into
# the free `output` tensor) loads in ORT. The scorer checks op DOMAIN not VERSION.
OPSET_IMPORTS = [helper.make_opsetid("", 11)]

S = 18  # active region side (generator width,height in [15,18], anchored top-left)


def build(task):
    nodes = []
    inits = []

    def init(name, arr):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
        return name

    def node(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- Slice the FREE input to JUST the red channel (ch2) over the 18x18
    # active region in ONE Slice.  Only black (ch0) and red (ch2) ever appear,
    # and the grid is anchored top-left with size <=18, so red lives entirely in
    # this [1,1,18,18] block.  Running the Conv on this single-channel block (vs
    # the full 30x30 input, or a 2-channel slice) keeps the dominant working
    # plane minimal. ----
    init("st", np.array([0, 2, 0, 0], np.int64))
    init("en", np.array([1, 3, S, S], np.int64))
    init("ax", np.array([0, 1, 2, 3], np.int64))
    node("Slice", ["input", "st", "en", "ax"], "redblk")  # [1,1,18,18] f32

    # ONE Conv on the red block yields ocount, encoding red-center AND
    # red-neighbor count in disjoint magnitude bands.  The center weight (500) is
    # large; the 4-neighbour count (weight 1) is small so it can never lift a
    # non-red cell past the red threshold.
    #   ocount = 500*center_red + 1*(#red 4-neighbors)
    # Value bands:
    #   non-red cell:       0..4       (#red nbrs only)
    #   static red center:  500        (red, no red nbr)
    #   olive red center:   501..504   (red, >=1 red nbr)
    # => red IFF ocount>250 ; olive IFF ocount>500.5
    Wo = np.zeros((1, 1, 3, 3), np.float32)
    Wo[0, 0, 1, 1] = 500.0                     # center red
    for r, c in [(0, 1), (2, 1), (1, 0), (1, 2)]:
        Wo[0, 0, r, c] = 1.0                   # plus-neighbor red count
    init("Wo", Wo)
    node("Conv", ["redblk", "Wo"], "ocount", pads=[1, 1, 1, 1])  # [1,1,18,18] f32

    init("t250", np.array(250.0, np.float32))
    init("t500", np.array(500.5, np.float32))
    node("Greater", ["ocount", "t250"], "redb")        # bool red center
    node("Greater", ["ocount", "t500"], "oliveb")      # bool olive_red

    # ---- in-grid rectangle from cheap 1-D occupancy profiles of the FREE input.
    # The grid is a solid rectangle anchored top-left whose every interior cell is
    # either background (ch0) or red (ch2), so the populated rows/cols are exactly
    # the in-grid extent.  ReduceMax over (channel, one-spatial) axes recovers the
    # per-row / per-col occupancy as tiny [1,1,S,1] / [1,1,1,S] vectors; their
    # outer AND is the in-grid mask -- no 2-D occupancy plane, no ch0 in the Conv.
    init("rs", np.array([0, 0, 0, 0], np.int64))
    init("re", np.array([1, 1, S, 1], np.int64))
    init("rax", np.array([0, 1, 2, 3], np.int64))
    init("zero", np.array(0.0, np.float32))
    # Threshold to bool BEFORE slicing so the sliced occupancy vectors are bool
    # (1 byte) rather than fp32 -- the slice never materialises an fp32 [1,1,S,1].
    node("ReduceMax", ["input"], "rowprof30", axes=[1, 3], keepdims=1)  # [1,1,30,1]
    node("Greater", ["rowprof30", "zero"], "rowin30")  # bool [1,1,30,1]
    node("Slice", ["rowin30", "rs", "re", "rax"], "rowin")              # [1,1,18,1]
    init("ce", np.array([1, 1, 1, S], np.int64))
    node("ReduceMax", ["input"], "colprof30", axes=[1, 2], keepdims=1)  # [1,1,1,30]
    node("Greater", ["colprof30", "zero"], "colin30")  # bool [1,1,1,30]
    node("Slice", ["colin30", "rs", "ce", "rax"], "colin")              # [1,1,1,18]
    node("And", ["rowin", "colin"], "ingridb")         # bool [1,1,18,18] in-grid

    # dilate olive 3x3 (Chebyshev-1) -> green region. Done in fp16 (olive counts
    # are tiny ints, exact in fp16) to halve the cast + conv plane bytes.
    node("Cast", ["oliveb"], "olivef", to=TensorProto.FLOAT16)  # [1,1,18,18] f16
    W2 = np.ones((1, 1, 3, 3), np.float16)
    init("W2", W2)
    node("Conv", ["olivef", "W2"], "dil", pads=[1, 1, 1, 1])  # [1,1,18,18] f16
    init("half16", np.array(0.5, np.float16))
    node("Greater", ["dil", "half16"], "dilb")
    # green = dilation, clipped to the grid (the generator draws green only inside
    # bounds). No need to subtract red here: the red Where below runs LAST and
    # overrides green wherever the cell is an olive-red center.
    node("And", ["dilb", "ingridb"], "greenb")      # dilation & in-grid

    # label L via nested Where (disjoint cases), in UINT8 to keep the planes
    # cheap (ORT Where/Equal/Pad are implemented for uint8). Off-grid -> 10 so it
    # matches no channel index 0..9 (Equal yields all-zero there, as required);
    # in-grid background -> 0 (ch0); green -> 3; red -> 2 (red wins overlaps).
    init("c2", np.array(2, np.uint8))
    init("c3", np.array(3, np.uint8))
    init("c10", np.array(10, np.uint8))
    init("c0", np.array(0, np.uint8))
    node("Where", ["ingridb", "c0", "c10"], "Lbase")  # 0 in-grid, 10 off-grid
    node("Where", ["greenb", "c3", "Lbase"], "Lg")    # green=3
    node("Where", ["redb", "c2", "Lg"], "Lc")         # red=2 over green/bg

    # pad the 18x18 uint8 label back to 30x30 with sentinel 10 (Pad supports
    # uint8; the padded off-grid region -> all-zero output).
    init("pads", np.array([0, 0, 0, 0, 0, 0, 30 - S, 30 - S], np.int64))
    init("padval", np.array(10, np.uint8))
    node("Pad", ["Lc", "pads", "padval"], "L", mode="constant")  # [1,1,30,30]

    # output = Equal(L, [0..9]) -> BOOL [1,10,30,30] routed into the FREE output
    arange = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("arange", arange)
    node("Equal", ["L", "arange"], "output")

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID_SHAPE)
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, GRID_SHAPE)
    graph = helper.make_graph(nodes, "task278", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=OPSET_IMPORTS)
