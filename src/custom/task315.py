"""Task 315 (ARC-AGI cce03e0d) — self-referential fractal placement.

Rule (from the generator): the input is a fixed 3x3 grid with colours in
{0,1,2} (red==2).  The 9x9 output is a fractal: block (r,c) (a 3x3 tile at
output rows 3r..3r+2, cols 3c..3c+2) holds a full copy of the input grid IFF
input[r][c]==2 (red); every other block is all background.  Equivalently:

    output[u,v] = input[u%3, v%3]   if input[u//3, v//3] == 2
                = 0                  otherwise

The input always occupies the top-left 3x3 of the 30x30 canvas.

Encoding (tier B label-map, generalises offset-free):
  * colour-index of the input 3x3 region: colf = sum_k k*input_k, sliced to the
    top-left 3x3 (a 3x3 fp32 plane, 36 B).
  * Sflat = flatten to [9] colour values.
  * Kronecker index maps macro=(u//3)*3+(v//3), micro=(u%3)*3+(v%3) over the
    9x9 output (constant [9,9] int64 maps, same idiom as task195).
  * mask9 = (Sflat[macro] == 2)   -- which output cell is in a "red" block
    val9  =  Sflat[micro]          -- the input colour to render
    L9    = Where(mask9, val9, 0)  -- 9x9 colour-index label
  * Pad to 30x30 with sentinel 99 (matches no channel) and the FREE bool output
    is Equal(L, arange[1,10,1,1]) (opset 11).

Memory floor: no full 30x30 colour plane is ever built; the only 30x30 tensor
is the uint8 label-map L (900 B, the canonical label floor).  Everything else
is tiny (3x3 colf, [9] Sflat, [9,9] label).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8
B = TensorProto.BOOL


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- colour-index of the input 3x3 region (top-left) ----
    # Only colours 1 and 2 ever appear (red==2).  Slice channels 1..2 over the
    # top-left 3x3: [1,2,3,3] fp32 = 72 B.
    init("ss", np.array([0, 1, 0, 0], np.int64), np.int64)
    init("se", np.array([1, 3, 3, 3], np.int64), np.int64)
    init("sax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "ss", "se", "sax"], "in33")  # [1,2,3,3] fp32 (72B)

    # colf = 1*ch1 + 2*ch2 via a 1x1 Conv (kernel weight [1,2]) -> [1,1,3,3]
    # colour index (36B): bg->0, colour1->1, colour2->2.  No Mul plane.
    cw = np.array([1.0, 2.0], np.float32).reshape(1, 2, 1, 1)
    init("cw", cw, np.float32)
    n("Conv", ["in33", "cw"], "S")                  # [1,1,3,3] fp32 (36B)

    # ---- flatten to [9] colour values, cast fp16 for cheap gathers ----
    init("s9", np.array([9], np.int64), np.int64)
    n("Reshape", ["S", "s9"], "Sflat32")            # [9] fp32
    n("Cast", ["Sflat32"], "Sflat", to=F16)         # [9] fp16

    # ---- Kronecker index maps over 9x9 output ----
    u = np.arange(9).reshape(9, 1)
    v = np.arange(9).reshape(1, 9)
    macro = ((u // 3) * 3 + (v // 3)).astype(np.int64)   # [9,9]
    micro = ((u % 3) * 3 + (v % 3)).astype(np.int64)     # [9,9]
    init("macro", macro, np.int64)
    init("micro", micro, np.int64)
    n("Gather", ["Sflat", "macro"], "Smac")         # [9,9] fp16 (block selector)
    n("Gather", ["Sflat", "micro"], "Smic")         # [9,9] fp16 (cell colour)

    # ---- mask = block colour == 2 (red); value = cell colour ----
    init("two", np.array(2.0, np.float16), np.float16)
    n("Equal", ["Smac", "two"], "mask9")            # [9,9] bool

    # value as uint8 colour-index; outside-red -> 0.
    n("Cast", ["Smic"], "val9", to=U8)              # [9,9] uint8
    init("u0", np.array(0, np.uint8), np.uint8)
    n("Where", ["mask9", "val9", "u0"], "L9")       # [9,9] uint8 label

    init("Ls", np.array([1, 1, 9, 9], np.int64), np.int64)
    n("Reshape", ["L9", "Ls"], "L9b")               # [1,1,9,9] uint8

    # pad 9x9 -> 30x30 with off-grid sentinel 99 (matches no channel 0..9).
    init("u99", np.array(99, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 21, 21], np.int64), np.int64)
    n("Pad", ["L9b", "pads", "u99"], "L", mode="constant")  # [1,1,30,30] uint8

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")             # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task315", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
