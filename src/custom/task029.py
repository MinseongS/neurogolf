"""task029 (ARC 1c786137): crop the interior of the unique rectangular ring.

Rule: the grid has random static plus one hollow rectangle drawn in a colour
never used by the static.  Output = the static content inside that rectangle,
translated to the top-left of the (zero-padded 30x30) output.

Leaner encoding vs the previous build (which used a region mask + sentinel
Where + a separate label plane):
  * The region mask + sentinel Where + separate label plane are FUSED into the
    crop gather.  colorid is padded with a constant sentinel-10 row/col; the
    per-output-row/col gather indices point at that pad row/col for any output
    cell outside the interior rectangle.  So the second gather output IS the
    label map L directly (10 = outside, 0..9 = colour id inside).  Removes the
    regionB plane, the Where, and the standalone L plane (~2.7KB).
  * All per-channel reductions run on the FREE full 30x30 input.
  * Final op Equal(L, arange[1,10,1,1]) writes the FREE BOOL output (opset 11).
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from src.harness import IR_VERSION

N = 30
BIG = 1000.0
I32 = TensorProto.INT32
F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8


def build(task):
    nodes, inits = [], []

    def init(name, arr, dtype=None):
        a = np.ascontiguousarray(arr, dtype=dtype) if dtype else np.ascontiguousarray(arr)
        inits.append(numpy_helper.from_array(a, name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    ih = np.arange(N, dtype=np.float16).reshape(1, 1, N, 1)
    iw = np.arange(N, dtype=np.float16).reshape(1, 1, 1, N)
    init("ih", ih)
    init("iw", iw)
    init("one", np.array(1.0, np.float16))
    init("big", np.array(BIG, np.float16))
    init("nbig", np.array(-BIG, np.float16))
    init("half", np.array(0.5, np.float32))
    init("halfh", np.array(0.5, np.float16))

    # --- per-channel row/col counts + presence (on FREE input) -----------
    n("ReduceSum", ["input"], "rc", axes=[3], keepdims=1)   # [1,10,30,1] f32
    n("ReduceSum", ["input"], "cc", axes=[2], keepdims=1)   # [1,10,1,30] f32
    n("Greater", ["cc", "half"], "cpb")

    # LEFT/RIGHT col edges (c0,c1) -> box width.
    n("Where", ["cpb", "iw", "big"], "clo")
    n("ReduceMin", ["clo"], "c0", axes=[3], keepdims=1)
    n("Where", ["cpb", "iw", "nbig"], "chi")
    n("ReduceMax", ["chi"], "c1", axes=[3], keepdims=1)

    n("Sub", ["c1", "c0"], "dw")        # bw-1
    n("Add", ["dw", "one"], "bw")       # box width

    # --- ring channel = the unique channel with exactly 2 FULL rows ------
    # A "full row" has rc == box-width; the rectangle ring is the only channel
    # with exactly two of them (verified unique on >8000 fresh instances).
    # Compare the f32 row counts directly against bw (cast the tiny scalar) so
    # no full-canvas count-cast plane is materialised.
    n("Cast", ["bw"], "bwf", to=F32)
    n("Equal", ["rc", "bwf"], "rowfull_b")        # [1,10,30,1] bool
    n("Cast", ["rowfull_b"], "rowfull", to=F16)
    n("ReduceSum", ["rowfull"], "nrf", axes=[2], keepdims=1)
    init("twof", np.array(2, np.float16))
    n("Equal", ["nrf", "twof"], "ring_b")
    n("Cast", ["ring_b"], "ring", to=F16)

    # top row edge r0 = the first FULL row (no separate presence plane needed)
    n("Where", ["rowfull_b", "ih", "big"], "rtop")
    n("ReduceMin", ["rtop"], "r0", axes=[2], keepdims=1)

    # box height from total ring-pixel count:  totc = 2*bw + 2*bh - 4
    #  -> bh = totc/2 - bw + 2.  totc is a tiny [1,10,1,1] f32 scalar (40B).
    n("ReduceSum", ["rc"], "totc_f", axes=[2], keepdims=1)  # [1,10,1,1] f32
    n("Cast", ["totc_f"], "totc", to=F16)                   # tiny scalar
    n("Mul", ["totc", "halfh"], "halfc")                    # totc/2
    init("twoh", np.array(2.0, np.float16))
    n("Sub", ["halfc", "bw"], "hcb")
    n("Add", ["hcb", "twoh"], "bh")                         # box height
    n("Mul", ["bh", "ring"], "score")                       # ring->bh, else 0

    n("ReduceMax", ["score"], "smax", axes=[1], keepdims=1)
    n("Cast", ["score"], "score_i", to=I32)
    n("Cast", ["smax"], "smax_i", to=I32)
    n("Equal", ["score_i", "smax_i"], "weq")
    n("Greater", ["score", "halfh"], "wpos")
    n("And", ["weq", "wpos"], "win_b")
    n("Cast", ["win_b"], "win", to=F16)

    # interior bounds of winner (scalars):
    #  interior top  = r0 + 1, interior left = c0 + 1
    #  interior h-1  = bh - 2, interior w-1  = bw - 2
    init("three", np.array(3.0, np.float16))
    n("Add", ["r0", "one"], "ir0c")
    n("Add", ["c0", "one"], "ic0c")
    n("Sub", ["bh", "three"], "idhc")   # interior height-1 = zoom_height-1
    n("Sub", ["bw", "three"], "idwc")   # interior width-1  = zoom_width-1
    for src, dst in [("ir0c", "ir0"), ("ic0c", "ic0"), ("idhc", "idh"), ("idwc", "idw")]:
        n("Mul", [src, "win"], dst + "_m")
        n("ReduceSum", [dst + "_m"], dst, axes=[1], keepdims=1)  # [1,1,1,1]

    # --- colour-id plane (0..9) + sentinel pad ---------------------------
    w_id = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("w_id", w_id, np.float32)
    n("Conv", ["input", "w_id"], "colorid_f")          # [1,1,30,30] f32
    n("Cast", ["colorid_f"], "colorid", to=U8)          # [1,1,30,30] u8
    init("pads", np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64))
    init("ten_u8", np.array(10, np.uint8), np.uint8)
    n("Pad", ["colorid", "pads", "ten_u8"], "colpad")   # [1,1,31,31] u8

    # --- gather indices: in-region -> ir0+r ; out-of-region -> sentinel --
    SENT = N                                        # index of the pad row/col
    init("scl1", np.array([1], np.int64), np.int64)
    n("Reshape", ["ir0", "scl1"], "ir0s")
    n("Reshape", ["ic0", "scl1"], "ic0s")
    n("Reshape", ["idh", "scl1"], "idhs")
    n("Reshape", ["idw", "scl1"], "idws")
    init("ar30f", np.arange(N, dtype=np.float16))      # output rows/cols 0..29
    init("sentf", np.array(SENT, np.float16))
    init("nh", np.array(-0.5, np.float16))
    n("Add", ["ar30f", "ir0s"], "gr0")
    n("Sub", ["idhs", "ar30f"], "rrem")                # >=0 where r <= idh
    n("Greater", ["rrem", "nh"], "rinB")
    n("Where", ["rinB", "gr0", "sentf"], "grf")
    n("Cast", ["grf"], "gr", to=I32)
    n("Add", ["ar30f", "ic0s"], "gc0")
    n("Sub", ["idws", "ar30f"], "crem")
    n("Greater", ["crem", "nh"], "cinB")
    n("Where", ["cinB", "gc0", "sentf"], "gcf")
    n("Cast", ["gcf"], "gc", to=I32)

    n("Gather", ["colpad", "gr"], "crop_r", axis=2)    # [1,1,30,31] u8
    n("Gather", ["crop_r", "gc"], "L", axis=3)         # [1,1,30,30] u8 label map

    init("chan10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan10"], "output")              # -> free BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, N, N])
    graph = helper.make_graph(nodes, "task029", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
