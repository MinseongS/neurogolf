"""Task 246 (ARC a2fd1cf0): red dot (2) at (r0,c0), green dot (3) at (r1,c1);
draw cyan (8) L-path along row r0 between the two columns, then down col c1
between the two rows; keep the red/green endpoints.

Single-channel colour-index plane via a SEPARABLE BILINEAR form
L[r,c] = R16[r,:] @ M @ C16[:,c], routed into the FREE bool output by
Equal(L, arange[0..9]).  This replaces the old 10-channel double-MatMul
(_hpwl: R10 [1,10,4,30] = 2400B) with a single [1,1,30,30] fp16 plane (1800B).

Per-axis vector v in {0,1,2} (1 at the red endpoint, 2 at the green endpoint)
comes from ONE Conv (weights red=1, green=2); occupancy w=clip(v,0,1); the
between-the-endpoints band b = prefixmax(w) * suffixmax(w) (inclusive both axes).
Features [w, v, b, in, ones]; the red-endpoint a=2w-v and green-endpoint c=v-w
fall out of M's coefficients, so no explicit Sub planes are needed.

Off-grid handling: a +99 sentinel (ones*ones - in*in) sends every off-grid cell
to L=99, which matches no arange value -> all channels False there (in-grid bg
cells net to L=0 -> channel 0).  in = ReduceMax(input,[1,3]/[1,2]) is the in-grid
row/col occupancy.

Floor-break: only the Conv / ReduceMax read the f32 input (4 x 120B); every
downstream vector op (Cast/Clip/MaxPool/Mul) runs in fp16 (60B each) and the
colour plane is fp16 (1800B).  mem 3900, params 457, score ~16.62 vs prior 16.29
(+0.33).  FRESH 500/500.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16
    BOOL = TensorProto.BOOL
    red, green = 2, 3
    KW = 18   # kernel width: covers all possible dot positions (cols/rows 1..17)

    # in-grid row/col occupancy (f32 on the f32 input), cast to fp16
    n("ReduceMax", ["input"], "rin_f", axes=[1, 3], keepdims=1)
    n("ReduceMax", ["input"], "cin_f", axes=[1, 2], keepdims=1)
    n("Cast", ["rin_f"], "rin", to=H)
    n("Cast", ["cin_f"], "cin", to=H)

    cmin = init("cmin", np.array(0.0, np.float16).reshape(1), dtype=np.float16)
    cmax = init("cmax", np.array(1.0, np.float16).reshape(1), dtype=np.float16)

    # ---- row side ----
    Wr = np.zeros((1, 10, 1, KW), np.float32)
    Wr[0, red, 0, :] = 1.0
    Wr[0, green, 0, :] = 2.0
    init("Wr", Wr)
    n("Conv", ["input", "Wr"], "vr_f", strides=[1, KW])   # f32 [1,1,30,1] v in {0,1,2}
    n("Cast", ["vr_f"], "vr", to=H)
    n("Clip", ["vr", "cmin", "cmax"], "wr")               # occupancy fp16
    n("MaxPool", ["wr"], "pr", kernel_shape=[30, 1], pads=[29, 0, 0, 0])
    n("MaxPool", ["wr"], "qr", kernel_shape=[30, 1], pads=[0, 0, 29, 0])
    n("Mul", ["pr", "qr"], "br")                           # between-inclusive [r0,r1]
    onesr = init("onesr", np.ones((1, 1, 30, 1), np.float16), dtype=np.float16)
    n("Concat", ["wr", "vr", "br", "rin", "onesr"], "R16", axis=3)  # [1,1,30,5]

    # ---- col side ----
    Wc = np.zeros((1, 10, KW, 1), np.float32)
    Wc[0, red, :, 0] = 1.0
    Wc[0, green, :, 0] = 2.0
    init("Wc", Wc)
    n("Conv", ["input", "Wc"], "vc_f", strides=[KW, 1])
    n("Cast", ["vc_f"], "vc", to=H)
    n("Clip", ["vc", "cmin", "cmax"], "wc")
    n("MaxPool", ["wc"], "pc", kernel_shape=[1, 30], pads=[0, 29, 0, 0])
    n("MaxPool", ["wc"], "qc", kernel_shape=[1, 30], pads=[0, 0, 0, 29])
    n("Mul", ["pc", "qc"], "bc")                           # between-inclusive [c0,c1]
    onesc = init("onesc", np.ones((1, 1, 1, 30), np.float16), dtype=np.float16)
    n("Concat", ["wc", "vc", "bc", "cin", "onesc"], "C16", axis=2)  # [1,1,5,30]

    # ---- bilinear M, features [w, v, b, in, ones] both axes ----
    # a = 2w - v (red endpoint), c = v - w (green endpoint).
    # L = 2*ar*ac + 3*cr*cc + 8*ar*(bc-wc) + 8*cc*(br-cr) + 99*ones*ones - 99*in*in
    # expanded to bilinear coefficients (verified residual 0):
    M = np.zeros((1, 1, 5, 5), np.float16)
    M[0, 0, 0, 0] = -13   # wr*wc
    M[0, 0, 0, 1] = 1     # wr*vc
    M[0, 0, 0, 2] = 16    # wr*bc
    M[0, 0, 1, 0] = 9     # vr*wc
    M[0, 0, 1, 1] = -3    # vr*vc
    M[0, 0, 1, 2] = -8    # vr*bc
    M[0, 0, 2, 0] = -8    # br*wc
    M[0, 0, 2, 1] = 8     # br*vc
    M[0, 0, 3, 3] = -99   # -99 * in_r*in_c
    M[0, 0, 4, 4] = 99    # +99 * ones*ones  -> off-grid sentinel 99
    init("M", M, dtype=np.float16)

    n("MatMul", ["M", "C16"], "MC")        # [1,1,5,30] fp16
    n("MatMul", ["R16", "MC"], "L")        # [1,1,30,30] fp16 colour-index plane
    arange = np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1)
    init("arange", arange, dtype=np.float16)
    n("Equal", ["L", "arange"], "output")  # [1,10,30,30] bool

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
