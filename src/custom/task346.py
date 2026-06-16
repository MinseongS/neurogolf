"""task346 (ARC-AGI d9fac9be) — output the centre colour of the 3x3 mono block.

Rule (from the generator):
  The grid (<=12x12, top-left) is sprinkled with 0.2-density random pixels in two
  random colours c0,c1.  One special 3x3 block is stamped: the 8 ring cells are all
  colour c1, the centre cell is colour c0.  The output is a 1x1 grid holding `center`
  = c0, the CENTRE colour of that block.  So output tensor = 1.0 only at
  [0, c0, 0, 0], zero elsewhere.

Detector (exact, no flood-fill):
  Work on the <=12x12 active window.  Let V = colour-index plane (sum_k k*input_k),
  O = occupancy (V>0).  Using an 8-RING conv kernel (3x3 ones, centre 0):
    S1 = sum of the 8 neighbours' colour values,
    S2 = sum of the 8 neighbours' squared values,
    Cnt = count of filled neighbours.
  A cell is the block centre iff Cnt==8 (all 8 neighbours filled) AND the 8
  neighbour values are all equal.  By Cauchy-Schwarz, 8 values are all equal iff
  8*S2 == S1^2 (strict inequality otherwise).  Combined => exact unique centre.
  centre colour = the V value at that cell (recovered as a scalar via ReduceMax of
  CM*V).  Output = Equal(centre_colour, arange) placed at cell (0,0).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
I64 = TensorProto.INT64
B = TensorProto.BOOL

WORK = 11  # block centre col,row in [2,9]; ring reads rows/cols 1..10 -> indices 0..10


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # colour-index plane V = sum_k k*input_k via a 1x1 conv on the FREE full input,
    # then slice to the WORKxWORK active window (V is only [1,1,30,30]).
    wk = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("vidx", wk, np.float32)
    n("Conv", ["input", "vidx"], "V30", kernel_shape=[1, 1])  # [1,1,30,30] f32
    init("s0", np.array([0, 0], np.int64), np.int64)
    init("sW", np.array([WORK, WORK], np.int64), np.int64)
    init("sax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["V30", "s0", "sW", "sax"], "Vf")             # [1,1,W,W] f32 (small)
    n("Cast", ["Vf"], "V", to=F16)                            # [1,1,W,W] f16 (<=9 exact)

    # V^2
    n("Mul", ["V", "V"], "V2")                                # [1,1,W,W] f16 (<=81 exact)

    # ---- 8-ring conv kernel (3x3 ones, centre 0), SAME padding -----------------
    ring = np.ones((1, 1, 3, 3), np.float16)
    ring[0, 0, 1, 1] = 0.0
    init("ring", ring, np.float16)
    n("Conv", ["V", "ring"], "S1", kernel_shape=[3, 3], pads=[1, 1, 1, 1])   # <=72 exact
    n("Conv", ["V2", "ring"], "S2", kernel_shape=[3, 3], pads=[1, 1, 1, 1])  # <=648 exact

    # centre test: the 8 neighbours are all EQUAL and NON-ZERO.
    #   8*S2 == S1^2  <=>  the 8 neighbour values are all equal (Cauchy-Schwarz
    #   equality; strict gap otherwise).  S1 > 0 excludes the all-empty case.
    # At a TRUE centre S1=8*c1, S1^2=64*c1^2=8*S2 — both multiples of 64, f16-exact
    # even up to 5184.  For any non-equal 8-set the gap 8*S2 - S1^2 = a*b*(c0-c1)^2 >= 7
    # (a+b=8, a,b>=1) >> the f16 rounding step (4) at this magnitude, so the f16 Equal
    # is exact.  (Verified 2000/2000 fresh.)
    init("eight16", np.array(8.0, np.float16), np.float16)
    init("zero16", np.array(0.0, np.float16), np.float16)
    n("Mul", ["S2", "eight16"], "S2x8")                       # f16
    n("Mul", ["S1", "S1"], "S1sq")                            # f16
    n("Equal", ["S2x8", "S1sq"], "mono")                      # bool: 8 nbrs all equal
    n("Greater", ["S1", "zero16"], "nz")                      # bool: not all empty
    n("And", ["mono", "nz"], "CM")                            # centre mask bool

    # ---- recover centre colour scalar = max over grid of (V where CM) -----------
    n("Where", ["CM", "V", "zero16"], "CV")                  # [1,1,W,W] f16
    n("ReduceMax", ["CV"], "cc16", axes=[2, 3], keepdims=1)   # [1,1,1,1] f16 = c0
    n("Cast", ["cc16"], "cc", to=F32)

    # ---- output: 1.0 at [0, c0, 0, 0] ------------------------------------------
    # one-hot over channels at the single cell (0,0): Equal(c0, arange[1,10,1,1])
    chan = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("chan", chan, np.float32)
    n("Equal", ["cc", "chan"], "onehot")                      # bool [1,10,1,1]
    n("Cast", ["onehot"], "oh", to=F32)                       # f32 [1,10,1,1]
    # pad spatial dims 1->30 with zeros => [1,10,30,30]
    init("zero", np.array(0.0, np.float32), np.float32)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 29, 29], np.int64), np.int64)
    n("Pad", ["oh", "padpads", "zero"], "output", mode="constant")

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task346", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
