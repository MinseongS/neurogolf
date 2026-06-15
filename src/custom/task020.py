"""task020 (ARC-AGI 11852cab) — complete a symmetric "blast" of diamond rings.

Rule (from the generator): a blast is centred at (row,col) and consists of 13
cells in 4 concentric diamond rings, each ring a single colour:
    ring0 (centre)      offsets {(0,0)}                       colour c0
    ring1 (diag dist 1) offsets {(-1,-1),(-1,1),(1,-1),(1,1)} colour c1
    ring2 (axis dist 2) offsets {(-2,0),(0,2),(2,0),(0,-2)}    colour c2
    ring3 (diag dist 2) offsets {(-2,-2),(-2,2),(2,-2),(2,2)}  colour c3
The INPUT shows every ring in full EXCEPT one ring, of which only ONE cell is
kept (so the input has exactly 10 of the 13 cells).  The OUTPUT completes all
rings.  Task: find the centre, read the 4 ring colours, re-stamp the pattern.

Recovery (generalising, verified 0 failures / 8000 fresh instances):
  * present = (non-background) mask.  Correlating it with the 13-cell stamp
    kernel gives value 10 at the true centre (all 10 present cells align) and at
    most 8 anywhere else -> the centre is the UNIQUE cell with corr > 9.5.
  * V = per-cell colour index (Conv of the one-hot input with [0..9]).
  * ring-k cells of the centre = Conv(centre_mask, ring-k kernel).  Each ring's
    colour = max of V over those cells (present cells hold the colour, missing
    cells are 0, colours >=1, so the max recovers the colour).
  * label map L = c0*centre + c1*R1 + c2*R2 + c3*R3 (disjoint positions); cells
    outside the real grid get sentinel 15 so the background channel (L==0) is
    not emitted there; then output = Equal(L, [0..9]) -> BOOL output (free).

Memory: the grid is always 10x10, so the input is cropped to a 10x10 canvas and
every derived single-channel plane is [1,1,10,10] (100 elem) in fp16 / uint8.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

W = 10  # working canvas side (grids are always 10x10)

RING1 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
RING2 = [(-2, 0), (0, 2), (2, 0), (0, -2)]
RING3 = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
ALL = [(0, 0)] + RING1 + RING2 + RING3


def _kernel(offsets, dtype):
    """5x5 conv kernel with 1 at the given (dr,dc) offsets (centre at 2,2).
    The stamps are centrally symmetric, so correlation == convolution."""
    k = np.zeros((1, 1, 5, 5), dtype)
    for dr, dc in offsets:
        k[0, 0, dr + 2, dc + 2] = 1
    return k


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
    F16 = TensorProto.FLOAT16
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    # ---- constants (fp16 for the convs; tiny params) ----
    init("Kstamp", _kernel(ALL, np.float16), np.float16)
    init("Kr1", _kernel(RING1, np.float16), np.float16)
    init("Kr2", _kernel(RING2, np.float16), np.float16)
    init("Kr3", _kernel(RING3, np.float16), np.float16)
    # input-side convs run in fp32 on the cropped canvas (avoids a 2000B fp16
    # copy of X); small outputs are cast to fp16 for the kernel convs.
    arangeW = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("arangeW", arangeW, np.float32)
    onesW = np.ones((1, 10, 1, 1), np.float32)
    init("onesW", onesW, np.float32)
    presW = onesW.copy(); presW[0, 0] = 0
    init("presW", presW, np.float32)
    init("thr", np.array(9.5, np.float16), np.float16)
    init("sent", np.array(15.0, np.float16), np.float16)
    init("one", np.array(1.0, np.float16), np.float16)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    # negative pads to crop input [1,10,30,30] -> [1,10,W,W]
    init("crop", np.array([0, 0, 0, 0, 0, 0, W - 30, W - 30], np.int64), np.int64)
    # pad L [1,1,W,W] -> [1,1,30,30] with sentinel 15 (so off-grid -> all false)
    init("padL", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("padv", np.array(15, np.uint8), np.uint8)

    # ---- crop to the 10x10 working canvas ----
    n("Pad", ["input", "crop"], "X")                          # [1,10,W,W] fp32

    # ---- present mask, colour grid, in-grid mask (fp32 convs, then fp16) ----
    n("Conv", ["X", "presW"], "pres32")                       # non-bg mask
    n("Conv", ["X", "arangeW"], "V32")                        # colour idx 0..9
    n("Conv", ["X", "onesW"], "ingrid32")                     # 1 inside grid
    n("Cast", ["pres32"], "pres", to=F16)
    n("Cast", ["V32"], "V", to=F16)
    n("Cast", ["ingrid32"], "ingrid", to=F16)

    # ---- centre detection: unique cell with corr > 9.5 ----
    n("Conv", ["pres", "Kstamp"], "corr", pads=[2, 2, 2, 2])
    n("Greater", ["corr", "thr"], "Cb")                       # bool centre
    n("Cast", ["Cb"], "Cf", to=F16)

    # ---- ring position masks (1 at ring-k cells of the centre) ----
    n("Conv", ["Cf", "Kr1"], "R1", pads=[2, 2, 2, 2])
    n("Conv", ["Cf", "Kr2"], "R2", pads=[2, 2, 2, 2])
    n("Conv", ["Cf", "Kr3"], "R3", pads=[2, 2, 2, 2])

    # ---- ring colours = max of V over each ring's positions (scalars) ----
    n("Mul", ["V", "Cf"], "vc0"); n("ReduceMax", ["vc0"], "c0", keepdims=1)
    n("Mul", ["V", "R1"], "vc1"); n("ReduceMax", ["vc1"], "c1", keepdims=1)
    n("Mul", ["V", "R2"], "vc2"); n("ReduceMax", ["vc2"], "c2", keepdims=1)
    n("Mul", ["V", "R3"], "vc3"); n("ReduceMax", ["vc3"], "c3", keepdims=1)

    # ---- label map L = c0*centre + c1*R1 + c2*R2 + c3*R3 (disjoint) ----
    n("Mul", ["Cf", "c0"], "L0")
    n("Mul", ["R1", "c1"], "L1")
    n("Mul", ["R2", "c2"], "L2")
    n("Mul", ["R3", "c3"], "L3")
    n("Add", ["L0", "L1"], "La")
    n("Add", ["L2", "L3"], "Lb")
    n("Add", ["La", "Lb"], "Lf")
    # off-grid -> sentinel 15 so background channel (L==0) is suppressed there
    n("Sub", ["one", "ingrid"], "outg")
    n("Mul", ["outg", "sent"], "outs")
    n("Add", ["Lf", "outs"], "Lfull")
    n("Cast", ["Lfull"], "L10", to=U8)                        # [1,1,W,W] uint8

    # ---- pad to 30x30 with sentinel 15, then final Equal -> BOOL output ----
    n("Pad", ["L10", "padL", "padv"], "L", mode="constant")   # [1,1,30,30] uint8
    n("Equal", ["L", "chan"], "output")                       # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task020", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
