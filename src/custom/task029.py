"""task029 (ARC 1c786137): crop the interior of the unique rectangular ring.

Rule (from ARC-GEN): the grid has random static plus one hollow rectangle drawn
in a color that is never used by the static.  The output is the static content
inside that rectangle, translated to the top-left.

Approach (pure integer arithmetic, exact in float32):
  - per color channel, get bbox [r0,r1]x[c0,c1] from row/col projections.
  - the border color is the one whose bbox has all four edges fully filled
    (top/bottom rows count == bw, left/right cols count == bh); among those
    pick the largest perimeter.  Row/col counts come straight from ReduceSum
    on the free `input` tensor, so no canvas-sized intermediate is needed.
  - build crop+translate selection matrices Lr, Lc from the interior bounds
    and emit  output = Lr @ input @ Lc^T  (writes the free `output` tensor).
"""

import numpy as np
import onnx

from ..harness import DATA_TYPE, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS

N = 30
BIG = 1000.0
I32 = onnx.TensorProto.INT32
F32 = onnx.TensorProto.FLOAT


def build(task):
    nodes, inits = [], []

    def init(name, arr, dtype=None):
        a = np.ascontiguousarray(arr, dtype=dtype) if dtype else np.ascontiguousarray(arr)
        inits.append(onnx.numpy_helper.from_array(a, name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(onnx.helper.make_node(op, ins, [out], **attrs))
        return out

    F16 = onnx.TensorProto.FLOAT16
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
    n("ReduceSum", ["input"], "rc", axes=[3], keepdims=1)   # [1,10,30,1] f32 per-row count
    n("ReduceSum", ["input"], "cc", axes=[2], keepdims=1)   # [1,10,1,30] f32 per-col count
    n("Greater", ["rc", "half"], "rpb")     # bool [1,10,30,1] row present
    n("Greater", ["cc", "half"], "cpb")     # bool [1,10,1,30] col present

    # r0 = min present row, r1 = max present row  (ih broadcast against [1,10,30,1])
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

    # --- ring test: exactly 2 full rows (rc==bw) and exactly 2 full cols (cc==bh) ---
    n("Cast", ["rc"], "rci", to=I32)
    n("Cast", ["cc"], "cci", to=I32)
    n("Cast", ["bw"], "bwi", to=I32)
    n("Cast", ["bh"], "bhi", to=I32)
    n("Equal", ["rci", "bwi"], "rowfull_b")          # [1,10,30,1] bool
    n("Equal", ["cci", "bhi"], "colfull_b")          # [1,10,1,30] bool
    n("Cast", ["rowfull_b"], "rowfull", to=F16)
    n("Cast", ["colfull_b"], "colfull", to=F16)
    n("ReduceSum", ["rowfull"], "nrf", axes=[2], keepdims=1)  # [1,10,1,1]
    n("ReduceSum", ["colfull"], "ncf", axes=[3], keepdims=1)  # [1,10,1,1]
    n("Cast", ["nrf"], "nrfi", to=I32)
    n("Cast", ["ncf"], "ncfi", to=I32)
    init("twoi", np.array(2, np.int32))
    n("Equal", ["nrfi", "twoi"], "rr_b")
    n("Equal", ["ncfi", "twoi"], "cc_b")
    n("And", ["rr_b", "cc_b"], "ring_b")
    n("Cast", ["ring_b"], "ring", to=F16)            # [1,10,1,1]

    # score = perimeter * ring  ( perimeter = 2*(dh+dw) )
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

    # --- build Lr / Lc selection matrices ---
    I = np.arange(N, dtype=np.int32).reshape(N, 1)
    J = np.arange(N, dtype=np.int32).reshape(1, N)
    # Lr[i,j] = (j - i == ir0) and (j <= ir1)
    init("Dmat", (J - I).reshape(1, 1, N, N), np.int32)   # j - i   (param only)
    # LcT[i,j] = Lc[j,i] = (i - j == ic0) and (i <= ic1)
    init("Emat", (I - J).reshape(1, 1, N, N), np.int32)   # i - j
    init("Jvec", J.reshape(1, 1, 1, N), np.int32)         # column index, broadcasts
    init("Ivec", I.reshape(1, 1, N, 1), np.int32)         # row index, broadcasts

    n("Add", ["ir1", "one"], "ir1p")          # hi+1 for <=  comparison
    n("Add", ["ic1", "one"], "ic1p")

    def build_sel(prefix, dmat, kvec, lo, hip):
        # L[i,j] = (dmat==lo) & (hip > kvec)   (kvec is the bounded axis index)
        n("Cast", [lo], prefix + "_loi", to=I32)
        n("Cast", [hip], prefix + "_hpi", to=I32)
        n("Equal", [dmat, prefix + "_loi"], prefix + "_deq")     # bool [1,1,30,30]
        n("Greater", [prefix + "_hpi", kvec], prefix + "_le")    # hip > k  (tiny vector)
        n("And", [prefix + "_deq", prefix + "_le"], prefix + "_b")
        n("Cast", [prefix + "_b"], prefix, to=F32)
        return prefix

    # Lr bounded by j (cols of the row-selector); LcT bounded by i (rows)
    build_sel("Lr", "Dmat", "Jvec", "ir0", "ir1p")
    build_sel("LcT", "Emat", "Ivec", "ic0", "ic1p")

    # output = Lr @ input @ Lc^T
    n("MatMul", ["Lr", "input"], "cropped")        # [1,10,30,30] f32
    n("MatMul", ["cropped", "LcT"], "output")

    x = onnx.helper.make_tensor_value_info("input", DATA_TYPE, GRID_SHAPE)
    y = onnx.helper.make_tensor_value_info("output", DATA_TYPE, GRID_SHAPE)
    graph = onnx.helper.make_graph(nodes, "task029", [x], [y], inits)
    return onnx.helper.make_model(graph, ir_version=IR_VERSION,
                                  opset_imports=OPSET_IMPORTS)
