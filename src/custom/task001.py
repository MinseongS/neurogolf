"""Task 001 (ARC-AGI 007bbfb7) — fractal self-tiling of a 3x3 sprite.

Rule (from the generator): a 3x3 grid S with 2..8 same-coloured on-cells (one
random colour 1..9) is rendered at 3x with copies of itself.  The 9x9 OUTPUT is
the Kronecker product  kron(S, S) * colour :
    output[3i+r, 3j+c] = colour  iff  S[i,j] AND S[r,c]   (else background 0).
Input sprite sits at top-left rows 0..2 cols 0..2; output 9x9 at top-left.

Recovery (no offset/upscale unlike task195 — sprite is fixed at 0..2):
  * occupancy S = (cell is non-background) = 1 - channel0 over the 3x3 corner.
  * colour = argmax over channels 1..9 of per-channel pixel counts (a scalar).
  * kron(S,S)[u,v] = Sflat[(u//3)*3+(v//3)] AND Sflat[(u%3)*3+(v%3)] via two
    Gathers of the recovered Sflat by constant [9,9] macro/micro index maps.
  * label L (9x9 uint8) = colour where kron else 0; Pad to 30x30 (sentinel 10
    off-footprint); free BOOL output = Equal(L, arange[1,10,1,1]) (opset 11).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
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

    # ---- 3x3 occupancy from channel-0 corner slice ----
    # exactly one channel set per cell; ch0==1 <=> background.  occ = 1 - ch0.
    init("ss", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("se", np.array([1, 1, 3, 3], np.int64), np.int64)
    init("sax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "ss", "se", "sax"], "bg")     # [1,1,3,3] fp32 (36B)
    init("one_f", np.array(1.0, np.float32), np.float32)
    n("Sub", ["one_f", "bg"], "occ")                   # [1,1,3,3] fp32 occupancy

    # ---- colour scalar = argmax over channels 1..9 of pixel counts ----
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=0)  # [1,10]
    # zero out channel 0 so background never wins.
    mask = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1]], np.float32)
    init("chmask", mask, np.float32)
    n("Mul", ["cnt", "chmask"], "cntm")                # [1,10]
    n("ArgMax", ["cntm"], "colidx", axis=1, keepdims=0)  # [1] int64 colour index

    # ---- kron(S,S) on a 9x9 bool canvas ----
    # kron(S,S)[u,v] = S[u//3,v//3] AND S[u%3,v%3].  Build each 9x9 factor by a
    # row-Gather then col-Gather of the 3x3 sprite (four [9] index vectors =
    # 36 params, vs two [9,9] flat maps = 162 params).
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["occ", "half"], "Sb")                # [1,1,3,3] bool

    div = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], np.int64)   # u//3
    mod = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], np.int64)   # u%3
    init("div", div, np.int64)
    init("mod", mod, np.int64)
    # macro factor: S[u//3, v//3]
    n("Gather", ["Sb", "div"], "macr", axis=2)         # [1,1,9,3] bool
    n("Gather", ["macr", "div"], "Smac", axis=3)       # [1,1,9,9] bool
    # micro factor: S[u%3, v%3]
    n("Gather", ["Sb", "mod"], "micr", axis=2)         # [1,1,9,3] bool
    n("Gather", ["micr", "mod"], "Smic", axis=3)       # [1,1,9,9] bool
    n("And", ["Smac", "Smic"], "Kb")                   # [1,1,9,9] bool (kron)

    # ---- label map L (9x9 uint8): colour where kron else 0 ----
    # colidx is int64 scalar -> cast to uint8 colour value.
    n("Cast", ["colidx"], "col_u8", to=TensorProto.UINT8)  # [1] uint8
    init("u0", np.array(0, np.uint8), np.uint8)
    # broadcast: Where(Kb[1,1,9,9], col_u8[1], u0) -> [1,1,9,9] uint8
    n("Where", ["Kb", "col_u8", "u0"], "Lsm4")         # [1,1,9,9] uint8

    init("u10", np.array(10, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 21, 21], np.int64), np.int64)
    n("Pad", ["Lsm4", "pads", "u10"], "L", mode="constant")  # [1,1,30,30] uint8

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")                # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task001", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
