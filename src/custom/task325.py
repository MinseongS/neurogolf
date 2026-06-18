"""task325 (ARC-AGI d0f5fe59) — "count the cyan blobs, emit an N x N cyan diagonal".

Rule (from the generator):
  The input grid (<=16x16, placed top-left) contains N separate cyan (colour 8)
  "creatures": each a single 4-connected blob of 4..8 cells inside a <=4x4 bounding
  box, placed non-overlapping with a gap of >=1 cell.  N = number of 4-connected
  cyan components (verified N in 1..6).  OUTPUT is the N x N grid that is cyan on its
  main diagonal (cell (i,i)) and background everywhere else.

Counting N WITHOUT flood-fill (exact, verified over 60k samples):
  For a binary foreground b the cubical Euler characteristic
      euler = V - Eh - Ev + F          (V cells, Eh/Ev adjacent pairs, F full 2x2)
  equals (#components - #holes).  The ONLY hole shape this generator produces is the
  3x3 ring (8 cells around an empty centre), so
      holes = #{empty cells whose 8 neighbours are all filled}
  and  N = euler + holes.

Encoding (geometry-bounded):
  Active canvas is <=16x16, so slice cyan (channel 8) to a 16x16 fp32 plane (512B,
  not 3600B) and cast to f16 for all products.  Eh,Ev,F via shifted f16 Mul +
  ReduceSum (products of {0,1} exact).  holes via one 3x3 ring Conv -> 8-neighbour
  count == 8 AND centre empty.  N = V - Eh - Ev + F + holes (scalar).
  Output is at most 6x6, so build a tiny 6x6 uint8 label L (8 on the in-grid diagonal,
  0 on in-grid off-diagonal, 99 outside the N x N region), Pad to 30x30 with 99, then
  output = Equal(L, channel-arange) -> BOOL (cells == 99 match no channel -> all zero).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
BOOL = TensorProto.BOOL
U8 = TensorProto.UINT8
I64 = TensorProto.INT64

W = 16   # active-canvas bound
K = 6    # max output side (max N)
N30 = 30


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def nd(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- foreground plane (cyan = channel 8), cropped to WxW -----------------
    init("st8", np.array([0, 8, 0, 0], np.int64), np.int64)
    init("en8", np.array([1, 9, W, W], np.int64), np.int64)
    init("ax4", np.array([0, 1, 2, 3], np.int64), np.int64)
    nd("Slice", ["input", "st8", "en8", "ax4"], "b32")         # [1,1,W,W] f32
    nd("Cast", ["b32"], "b", to=F16)                           # f16 {0,1}

    # ---- V = #cells ---------------------------------------------------------
    nd("ReduceSum", ["b"], "V", axes=[2, 3], keepdims=1)       # [1,1,1,1] f16

    # ---- adjacency / 2x2 products -------------------------------------------
    init("z", np.array([0], np.int64), np.int64)
    init("wm1", np.array([W - 1], np.int64), np.int64)
    init("o1", np.array([1], np.int64), np.int64)
    init("wN", np.array([W], np.int64), np.int64)
    init("axc", np.array([3], np.int64), np.int64)
    init("axr", np.array([2], np.int64), np.int64)

    nd("Slice", ["b", "z", "wm1", "axc"], "Lh")                # cols 0..W-2
    nd("Slice", ["b", "o1", "wN", "axc"], "Rh")                # cols 1..W-1
    nd("Mul", ["Lh", "Rh"], "hp")                              # [1,1,W,W-1]
    nd("ReduceSum", ["hp"], "Eh", axes=[2, 3], keepdims=1)

    nd("Slice", ["b", "z", "wm1", "axr"], "Uv")                # rows 0..W-2
    nd("Slice", ["b", "o1", "wN", "axr"], "Dv")                # rows 1..W-1
    nd("Mul", ["Uv", "Dv"], "vp")                              # [1,1,W-1,W]
    nd("ReduceSum", ["vp"], "Ev", axes=[2, 3], keepdims=1)

    # full 2x2 = hp shifted one row down, multiplied
    nd("Slice", ["hp", "z", "wm1", "axr"], "hpU")              # [1,1,W-1,W-1]
    nd("Slice", ["hp", "o1", "wN", "axr"], "hpD")
    nd("Mul", ["hpU", "hpD"], "fp")
    nd("ReduceSum", ["fp"], "F", axes=[2, 3], keepdims=1)

    # ---- holes: empty cell with all 8 neighbours filled ---------------------
    ker = np.ones((1, 1, 3, 3), np.float16)
    ker[0, 0, 1, 1] = 0.0
    init("ringk", ker, np.float16)
    nd("Conv", ["b", "ringk"], "neigh", pads=[1, 1, 1, 1])     # [1,1,W,W] f16
    init("SEVEN5", np.array(7.5, np.float16), np.float16)
    nd("Greater", ["neigh", "SEVEN5"], "is8")                  # ==8
    init("PT5", np.array(0.5, np.float16), np.float16)
    nd("Less", ["b", "PT5"], "empty")                          # b==0
    nd("And", ["is8", "empty"], "hole_b")
    nd("Cast", ["hole_b"], "hole_f", to=F16)
    nd("ReduceSum", ["hole_f"], "H", axes=[2, 3], keepdims=1)

    # ---- N = V - Eh - Ev + F + H  (scalar) ----------------------------------
    nd("Sub", ["V", "Eh"], "t1")
    nd("Sub", ["t1", "Ev"], "t2")
    nd("Add", ["t2", "F"], "t3")
    nd("Add", ["t3", "H"], "Nf")                               # [1,1,1,1] f16

    # ---- tiny 6x6 label L ----------------------------------------------------
    rr = np.arange(K, dtype=np.float16).reshape(1, 1, K, 1)
    cc = np.arange(K, dtype=np.float16).reshape(1, 1, 1, K)
    init("rampr", rr, np.float16)                              # [1,1,K,1]
    init("rampc", cc, np.float16)                              # [1,1,1,K]
    nd("Equal", ["rampr", "rampc"], "ondiag")                  # [1,1,K,K] bool
    nd("Less", ["rampr", "Nf"], "rinN")                        # [1,1,K,1]
    nd("Less", ["rampc", "Nf"], "cinN")                        # [1,1,1,K]
    nd("And", ["rinN", "cinN"], "ingrid")                      # [1,1,K,K]
    nd("And", ["ondiag", "rinN"], "diag")                      # [1,1,K,K] (r==c & r<N)

    nd("Cast", ["ingrid"], "ingrid_f", to=F16)
    nd("Cast", ["diag"], "diag_f", to=F16)
    init("ONEF", np.array(1.0, np.float16), np.float16)
    init("EIGHT", np.array(8.0, np.float16), np.float16)
    init("N99", np.array(99.0, np.float16), np.float16)
    nd("Sub", ["ONEF", "ingrid_f"], "outg")                    # 1 outside grid
    nd("Mul", ["outg", "N99"], "Lout")                         # 99 outside else 0
    nd("Mul", ["diag_f", "EIGHT"], "Ldiag")                    # 8 on diagonal
    nd("Add", ["Lout", "Ldiag"], "Lf")                         # [1,1,K,K] f16
    nd("Cast", ["Lf"], "Lsmall", to=U8)                        # [1,1,K,K] uint8

    # ---- pad 6x6 -> 30x30 with sentinel 99 ----------------------------------
    init("SENTU", np.array(99, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, N30 - K, N30 - K], np.int64), np.int64)
    nd("Pad", ["Lsmall", "pads", "SENTU"], "L", mode="constant")  # [1,1,30,30] u8

    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    nd("Equal", ["L", "chan"], "output")                       # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task325", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
