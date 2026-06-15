"""task029 (ARC 1c786137): crop the interior of the unique rectangular ring.

Rule (from ARC-GEN): the grid has random static plus one hollow rectangle drawn
in a color that is never used by the static.  The output is the static content
inside that rectangle, translated to the top-left.

Memory floor-break (single-plane crop + label map + final Equal):
  The previous encoding applied the crop/translate selection matrices to the
  full 10-channel input (Lr @ input @ Lc^T), materialising a [1,10,30,30] f32
  intermediate (36000B).  Instead we collapse the input to a single uint8/float
  colour-id plane colorid[1,1,30,30] (0..9), crop THAT with the same selection
  matrices (Lr @ colorid @ Lc^T -> [1,1,30,30], 3600B), and form a uint8 label
  map L: cropped colour id inside the output rectangle, sentinel 10 outside
  (10 never matches channels 0..9, so those cells are all-false, while
  background cells map to id 0 -> channel 0 true, exactly as required).  The
  final op Equal(L, arange[1,10,1,1]) writes the free BOOL `output` (opset 11).
  Bounds are recovered with the same tiny scalar arithmetic as before.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

N = 30
BIG = 1000.0
I32 = TensorProto.INT32
F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16


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
    init("ih", ih)            # [1,1,30,1] f16
    init("iw", iw)            # [1,1,1,30] f16
    init("one", np.array(1.0, np.float16))
    init("two", np.array(2.0, np.float16))
    init("big", np.array(BIG, np.float16))
    init("nbig", np.array(-BIG, np.float16))
    init("half", np.array(0.5, np.float32))    # for f32 count comparison
    init("halfh", np.array(0.5, np.float16))   # for f16 score comparison

    # --- per-channel counts & bbox (presence derived from counts via Where) ---
    n("ReduceSum", ["input"], "rc", axes=[3], keepdims=1)   # [1,10,30,1] row count
    n("ReduceSum", ["input"], "cc", axes=[2], keepdims=1)   # [1,10,1,30] col count
    n("Greater", ["rc", "half"], "rpb")     # bool [1,10,30,1] row present
    n("Greater", ["cc", "half"], "cpb")     # bool [1,10,1,30] col present

    # r0 = min present row, r1 = max present row
    n("Where", ["rpb", "ih", "big"], "rlo")     # [1,10,30,1]
    n("ReduceMin", ["rlo"], "r0", axes=[2], keepdims=1)     # [1,10,1,1]
    n("Where", ["rpb", "ih", "nbig"], "rhi")
    n("ReduceMax", ["rhi"], "r1", axes=[2], keepdims=1)
    n("Where", ["cpb", "iw", "big"], "clo")
    n("ReduceMin", ["clo"], "c0", axes=[3], keepdims=1)
    n("Where", ["cpb", "iw", "nbig"], "chi")
    n("ReduceMax", ["chi"], "c1", axes=[3], keepdims=1)

    # bbox dims
    n("Sub", ["r1", "r0"], "dh")        # bh-1
    n("Sub", ["c1", "c0"], "dw")        # bw-1
    n("Add", ["dh", "one"], "bh")       # [1,10,1,1]
    n("Add", ["dw", "one"], "bw")

    # --- ring test: exactly 2 full rows (rc==bw) and 2 full cols (cc==bh) ---
    # all counts/dims are <= 30 integers, exact in f16 -> use f16 Equal (no
    # int32 count planes).
    n("Cast", ["rc"], "rcf", to=F16)                 # [1,10,30,1] f16
    n("Cast", ["cc"], "ccf", to=F16)                 # [1,10,1,30] f16
    n("Equal", ["rcf", "bw"], "rowfull_b")           # [1,10,30,1] bool
    n("Equal", ["ccf", "bh"], "colfull_b")           # [1,10,1,30] bool
    n("Cast", ["rowfull_b"], "rowfull", to=F16)
    n("Cast", ["colfull_b"], "colfull", to=F16)
    n("ReduceSum", ["rowfull"], "nrf", axes=[2], keepdims=1)  # [1,10,1,1]
    n("ReduceSum", ["colfull"], "ncf", axes=[3], keepdims=1)  # [1,10,1,1]
    init("twof", np.array(2, np.float16))
    n("Equal", ["nrf", "twof"], "rr_b")
    n("Equal", ["ncf", "twof"], "cc_b")
    n("And", ["rr_b", "cc_b"], "ring_b")
    n("Cast", ["ring_b"], "ring", to=F16)            # [1,10,1,1]

    # score = perimeter * ring
    n("Add", ["dh", "dw"], "dhw")
    n("Mul", ["dhw", "two"], "peri")
    n("Mul", ["peri", "ring"], "score")       # [1,10,1,1]

    # --- pick winner channel ---
    n("ReduceMax", ["score"], "smax", axes=[1], keepdims=1)
    n("Cast", ["score"], "score_i", to=I32)
    n("Cast", ["smax"], "smax_i", to=I32)
    n("Equal", ["score_i", "smax_i"], "weq")
    n("Greater", ["score", "halfh"], "wpos")
    n("And", ["weq", "wpos"], "win_b")
    n("Cast", ["win_b"], "win", to=F16)       # [1,10,1,1] f16

    # interior bounds (winner)
    n("Add", ["r0", "one"], "ir0c")
    n("Sub", ["r1", "one"], "ir1c")
    n("Add", ["c0", "one"], "ic0c")
    n("Sub", ["c1", "one"], "ic1c")
    for src, dst in [("ir0c", "ir0"), ("ir1c", "ir1"), ("ic0c", "ic0"), ("ic1c", "ic1")]:
        n("Mul", [src, "win"], dst + "_m")
        n("ReduceSum", [dst + "_m"], dst, axes=[1], keepdims=1)  # [1,1,1,1]

    # --- collapse input to a single colour-id plane (cast to uint8: ids 0..9) ---
    w_id = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
    init("w_id", w_id, np.float32)
    n("Conv", ["input", "w_id"], "colorid_f")      # [1,1,30,30] f32 (id 0..9)
    n("Cast", ["colorid_f"], "colorid", to=TensorProto.UINT8)  # [1,1,30,30] u8

    # --- crop+translate by GATHER (output[r,c] = colorid[ir0+r, ic0+c]) ---
    # gather index vectors gr/gc = bound + arange(30), clamped to 0..29 (cells
    # past the region are masked to the sentinel below, so the clamp is safe).
    n("Reshape", ["ir0", init("scl1", np.array([1], np.int64), np.int64)], "ir0s")
    n("Reshape", ["ic0", "scl1"], "ic0s")          # [1] f16
    init("ar30f", np.arange(N, dtype=np.float16))  # [30] f16
    init("capf", np.array(N - 1, np.float16))      # clamp ceiling 29
    n("Add", ["ar30f", "ir0s"], "gr0")             # [30] f16
    n("Add", ["ar30f", "ic0s"], "gc0")
    n("Min", ["gr0", "capf"], "grf")               # clamp to <= 29 (f16)
    n("Min", ["gc0", "capf"], "gcf")
    n("Cast", ["grf"], "gr", to=I32)               # [30] int32 indices
    n("Cast", ["gcf"], "gc", to=I32)
    n("Gather", ["colorid", "gr"], "crop_r", axis=2)   # [1,1,30,30] rows shifted
    n("Gather", ["crop_r", "gc"], "cropped", axis=3)   # [1,1,30,30] cols shifted

    # --- output region mask (rows 0..dh-1 of interior, cols 0..dw-1) ---
    # interior height = ir1 - ir0 + 1, width = ic1 - ic0 + 1.
    n("Sub", ["ir1", "ir0"], "idh16")   # interior height - 1 (f16)
    n("Sub", ["ic1", "ic0"], "idw16")   # interior width - 1 (f16)
    # region: row <= idh  and  col <= idw  (output placed at top-left)
    init("ihf", np.arange(N, dtype=np.float16).reshape(1, 1, N, 1))  # row idx f16
    init("iwf", np.arange(N, dtype=np.float16).reshape(1, 1, 1, N))  # col idx f16
    n("Sub", ["idh16", "ihf"], "rd")    # >=0 where row <= idh
    n("Sub", ["idw16", "iwf"], "cd")
    init("nh", np.array(-0.5, np.float16))
    n("Greater", ["rd", "nh"], "rokB")  # row <= idh
    n("Greater", ["cd", "nh"], "cokB")  # col <= idw
    n("And", ["rokB", "cokB"], "regionB")          # [1,1,30,30] bool

    # --- label map: cropped colour id inside region, sentinel 10 outside ---
    init("v10", np.array(10, np.uint8), np.uint8)
    n("Where", ["regionB", "cropped", "v10"], "L")            # [1,1,30,30] uint8

    init("chan10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan10"], "output")                     # -> free BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, N, N])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, N, N])
    graph = helper.make_graph(nodes, "task029", [x], [y], inits)
    return helper.make_model(graph, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
