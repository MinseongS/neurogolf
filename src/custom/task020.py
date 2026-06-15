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

Generator facts that bound the geometry:
  * grid is ALWAYS 10x10 (size=10);
  * centre row,col in randint(3, size-4) = {3,4,5,6};
  * therefore every blast cell (centre +/- up to 2) lies in rows/cols 1..8.
  => a 8x8 crop at offset (1,1) captures the WHOLE blast.  Working on 8x8
     instead of 10x10 shrinks the dominant fp32 input crop 10*100*4=4000B to
     10*64*4=2560B and every derived plane from 100->64 elements.

Recovery (generalising, verified 0 failures / fresh instances):
  * present = (non-background) mask.  Correlating it with the 13-cell stamp
    kernel gives value 10 at the true centre (all 10 present cells align) and at
    most 8 anywhere else -> the centre is the UNIQUE cell with corr > 9.5.
  * V = per-cell colour index (Conv of the one-hot input with [0..9]).
  * ring-k cells of the centre = Conv(centre_mask, ring-k kernel).  Each ring's
    colour = max of V over those cells (present cells hold the colour, missing
    cells are 0, colours >=1, so the max recovers the colour).
  * label map L (8x8) = c0*centre + c1*R1 + c2*R2 + c3*R3 (disjoint positions).
    The 8x8 crop is fully inside the 10x10 grid, so no off-grid sentinel is
    needed inside it.  Pad 8x8 -> 10x10 with 0 (the dropped grid border is all
    background => channel-0 true), then 10x10 -> 30x30 with sentinel 15 (off the
    real grid => all channels false).  Final output = Equal(L, [0..9]) -> BOOL.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

W = 8          # working canvas side (blast always fits rows/cols 1..8)
OFF = 1        # crop offset: keep grid indices 1..8

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

    # ---- conv kernels (fp16; tiny params) ----
    init("Kstamp", _kernel(ALL, np.float16), np.float16)
    init("Kr1", _kernel(RING1, np.float16), np.float16)
    init("Kr2", _kernel(RING2, np.float16), np.float16)
    init("Kr3", _kernel(RING3, np.float16), np.float16)
    # colour-index conv (1x1, run on the fp32 crop, in fp32).  Colours come from
    # {1,2,3,4,8}; channel 0 (bg) contributes 0 and channel 9 is unused, so we
    # crop only channels 1..8 (CH=8) and weight them by their colour index.
    CH = 8
    CH0 = 1  # first kept channel
    arangeW = np.arange(CH0, CH0 + CH, dtype=np.float32).reshape(1, CH, 1, 1)
    init("arangeW", arangeW, np.float32)
    init("thr", np.array(9.5, np.float16), np.float16)
    init("zero", np.array(0.0, np.float16), np.float16)
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    # negative pads to crop input [1,10,30,30] -> [1,CH,W,W]: keep spatial idx
    # OFF..OFF+W-1 and channels CH0..CH0+CH-1 (drop bg ch0 and unused ch9).
    init("crop", np.array([0, -CH0, -OFF, -OFF, 0, CH0 + CH - 10,
                           OFF + W - 30, OFF + W - 30], np.int64), np.int64)
    # pad L [1,1,W,W] -> [1,1,10,10] with 0 (dropped border is grid background)
    init("padG", np.array([0, 0, OFF, OFF, 0, 0, 10 - OFF - W, 10 - OFF - W],
                          np.int64), np.int64)
    # pad L [1,1,10,10] -> [1,1,30,30] with sentinel 15 (off-grid -> all false)
    init("padO", np.array([0, 0, 0, 0, 0, 0, 30 - 10, 30 - 10], np.int64),
         np.int64)
    init("sent", np.array(15, np.uint8), np.uint8)

    # ---- crop to the WxW working canvas (the one dominant fp32 plane) ----
    n("Pad", ["input", "crop"], "X")                          # [1,CH,W,W] fp32

    # ---- colour grid via one 1x1 channel-reduction conv; pres = (V>0) ----
    n("Conv", ["X", "arangeW"], "V32")                        # colour idx 0..9
    n("Cast", ["V32"], "V", to=F16)
    n("Greater", ["V", "zero"], "presB")                      # non-bg mask bool
    n("Cast", ["presB"], "pres", to=F16)

    # ---- centre detection: unique cell with corr > 9.5 ----
    n("Conv", ["pres", "Kstamp"], "corr", pads=[2, 2, 2, 2])
    n("Greater", ["corr", "thr"], "Cb")                       # bool centre
    n("Cast", ["Cb"], "Cf", to=F16)

    # ---- ring position masks (1 at ring-k cells of the centre) ----
    n("Conv", ["Cf", "Kr1"], "R1", pads=[2, 2, 2, 2])
    n("Conv", ["Cf", "Kr2"], "R2", pads=[2, 2, 2, 2])
    n("Conv", ["Cf", "Kr3"], "R3", pads=[2, 2, 2, 2])

    # ---- centre is ALWAYS present, so V*Cf already IS the coloured centre
    #      (L0); rings 1..3 may be incomplete, so broadcast their max colour. ----
    n("Mul", ["V", "Cf"], "L0")                               # coloured centre
    n("Mul", ["V", "R1"], "vc1"); n("ReduceMax", ["vc1"], "c1", keepdims=1)
    n("Mul", ["V", "R2"], "vc2"); n("ReduceMax", ["vc2"], "c2", keepdims=1)
    n("Mul", ["V", "R3"], "vc3"); n("ReduceMax", ["vc3"], "c3", keepdims=1)

    # ---- label map L = L0 + c1*R1 + c2*R2 + c3*R3 (disjoint positions) ----
    n("Mul", ["R1", "c1"], "L1")
    n("Mul", ["R2", "c2"], "L2")
    n("Mul", ["R3", "c3"], "L3")
    nodes.append(helper.make_node("Sum", ["L0", "L1", "L2", "L3"], ["Lf"]))
    n("Cast", ["Lf"], "Lw", to=U8)                            # [1,1,W,W] uint8

    # ---- pad W -> 10 (border bg, value 0) -> 30 (off-grid, sentinel 15) ----
    n("Pad", ["Lw", "padG"], "L10", mode="constant")          # [1,1,10,10] uint8
    n("Pad", ["L10", "padO", "sent"], "L", mode="constant")   # [1,1,30,30] uint8
    n("Equal", ["L", "chan"], "output")                       # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task020", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
