"""Task 175 (73251a56): fill the black cutouts of a deterministic ratio grid.

Rule (from ARC-GEN generator), grid is a fixed SIZE=21 square:
    color = 2                if r == c
            (r+2)//(c+2)     if r > c
            (c+2)//(r+2)     if r < c
    value[r,c] = (color + modset) % mod + 1            (mod in 5..9, modset in 1..4)
The input is this value plane with up to 5 black (0) rectangles cut out; the
output restores the full value plane.  value >= 1 everywhere, so black (0) marks
ONLY cutouts -> the output is purely a function of (COLOR, mod, modset); the
input only supplies the two scalars (mod, modset).

COLOR is a COMPILE-TIME CONSTANT 21x21 matrix.  Recover the two scalars from the
per-channel pixel COUNT vector cnt = ReduceSum(input, axes=[2,3]) -> [1,10,1,1]
(40 bytes, NO 30x30 working plane at all):
  * mod = the largest colour index present  (residues span a full period, so the
          max value reached is exactly `mod`).
  * The cells with COLOR == 1 (220 of the 441) all carry the same value
    v1 = (1+modset)%mod + 1, and that residue band (COLOR == 1, 1+mod, ...) so
    dominates the colour histogram that ARGMAX(cnt) == v1 even after up to ~125
    cut cells (verified 0/5000, min top1-top2 margin 62).  Then
    modset = (v1 - 2) mod mod.
value = (COLOR + modset) % mod + 1, routed into the FREE bool output via one
Equal; the +1 is folded into the channel constant (chan[k]=k-1).  The only
full-canvas intermediates are the tiny fp16 value plane (882B) and the uint8
label (441/900B) -- no [1,10,*,*] or [1,1,30,30] fp32 plane is ever created.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

SIZE = 21


def _color_mat():
    C = np.zeros((SIZE, SIZE), np.float32)
    for r in range(SIZE):
        for c in range(SIZE):
            if r > c:
                col = (r + 2) // (c + 2)
            elif r < c:
                col = (c + 2) // (r + 2)
            else:
                col = 2
            C[r][c] = col
    return C


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    Cmat = _color_mat()                                       # [21,21] values 1..11

    # ---- constants ----
    init("COLOR", Cmat.reshape(1, 1, SIZE, SIZE), np.float16)
    # mask channel 0 (background) out of the count so it can never win ArgMax /
    # presence; karange = colour index per channel.
    mask0 = np.ones((1, 10, 1, 1), np.float32); mask0[0, 0, 0, 0] = 0.0
    init("mask0", mask0, np.float32)
    init("karange", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("zero", np.array(0.0, np.float32), np.float32)
    init("two", np.array(2.0, np.float16), np.float16)
    init("one", np.array(1.0, np.float16), np.float16)
    # channel constant with the +1 folded in: channel k matches value==k i.e.
    # vmod == k-1.  chan[k]=k-1 as uint8; chan[0]=255 (vmod<=8 never hits 255).
    chan_km1 = (np.arange(10, dtype=np.int64) - 1) % 256
    init("chan", chan_km1.reshape(1, 10, 1, 1).astype(np.uint8), np.uint8)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - SIZE, 30 - SIZE], np.int64),
         np.int64)
    init("padval", np.array(99, np.uint8), np.uint8)

    # ---- per-channel pixel count (the only reduction over the input) ----
    n("ReduceSum", ["input"], "cntraw", axes=[2, 3], keepdims=1)  # [1,10,1,1] fp32
    n("Mul", ["cntraw", "mask0"], "cnt")                      # zero out channel 0

    # ---- mod = largest present colour index ----
    n("Greater", ["cnt", "zero"], "presb")                    # [1,10,1,1] bool
    n("Cast", ["presb"], "presf", to=F)
    n("Mul", ["presf", "karange"], "presk")
    n("ReduceMax", ["presk"], "modf32", axes=[1], keepdims=1)  # [1,1,1,1] = mod
    n("Cast", ["modf32"], "modf", to=H)

    # ---- v1 = ArgMax(cnt) over channels = value at COLOR==1 cells ----
    n("ArgMax", ["cnt"], "v1i", axis=1, keepdims=1)           # [1,1,1,1] int64
    n("Cast", ["v1i"], "v1", to=H)
    # modset = (v1 - 2) mod mod   (add mod first to stay non-negative)
    n("Sub", ["v1", "two"], "v1m2")                           # >= -1
    n("Add", ["v1m2", "modf"], "v1m2p")                       # >= 0
    n("Mod", ["v1m2p", "modf"], "modsetf", fmod=1)            # scalar = modset

    # ---- value = (COLOR + modset) % mod + 1 (fold +1 into chan) ----
    n("Add", ["COLOR", "modsetf"], "cm")                      # [1,1,21,21] fp16
    n("Mod", ["cm", "modf"], "vmod", fmod=1)                  # value-1 in [0,mod-1]
    n("Cast", ["vmod"], "vmu", to=U8)                         # [1,1,21,21] uint8

    # ---- pad to 30x30 (sentinel 99) and route into the FREE bool output ----
    n("Pad", ["vmu", "padpads", "padval"], "L", mode="constant")
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task175", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
