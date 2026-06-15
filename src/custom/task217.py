"""Task 217 (8f2ea7aa): self-similar fractal sprite -> sprite (X) sprite.

Rule (ARC-GEN, size=3 always): the 9x9 input (a 3x3 grid of 3x3 blocks) holds a
single copy of a 3x3 sprite S placed in one block, in a single colour cc.  The
output is the Kronecker fractal S (X) S over the same 9x9 region:
  output[bR*3+r][bC*3+c] = cc  iff  S[bR,bC] and S[r,c]   (else background 0).

Memory floor-break (label map + final Equal).  Collapse the 9 colour channels to
a single binary picture immediately, so no [1,9,9,9] / [1,10,9,9] float stack is
ever materialised:
  - colmask [1,1,9,9] (fp16) = the single-colour picture (channels 1..9 summed by
    a 1x1 Conv collapse weight, after slicing the 9x9 region);
  - sprite S [1,1,3,3] = block sum of colmask (only one block is nonzero -> exact);
  - fractal M [1,1,9,9] = ConvTranspose(S, S, stride 3) > 0;
  - cc = single colour index (uint8 scalar);
  - L [1,1,9,9] uint8 = M ? cc : 0; Pad to 30x30 with sentinel 10 (outside cells
    all-false, matching the generator's background-free pad), final Equal.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
U8 = TensorProto.UINT8


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **kw):
        nodes.append(helper.make_node(op, ins, [out], **kw))
        return out

    # ---- collapse colour channels 1..9 to a single binary picture, then slice ----
    # Conv on the full free input keeps a single 3600-byte plane (vs a 3240+ slice
    # of all 10 channels); slice to the 9x9 active region immediately.
    init("collapse", np.array([0.] + [1.] * 9, np.float32).reshape(1, 10, 1, 1), np.float32)
    n("Conv", ["input", "collapse"], "pic")                        # [1,1,30,30] f32 0/1
    init("s_st", np.array([0, 0], np.int64), np.int64)
    init("s_en", np.array([9, 9], np.int64), np.int64)
    init("s_ax", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["pic", "s_st", "s_en", "s_ax"], "colmask")         # [1,1,9,9] f32 0/1

    # ---- sprite S = block sum of colmask (one block nonzero -> exact) ----
    init("sh6", np.array([1, 1, 3, 3, 3, 3], np.int64), np.int64)
    n("Reshape", ["colmask", "sh6"], "c6")                         # [1,1,3,3,3,3]
    n("ReduceSum", ["c6"], "S", axes=[2, 4], keepdims=0)           # [1,1,3,3] f32

    # ---- fractal M = ConvTranspose(S, S, stride 3) > 0 ----
    n("ConvTranspose", ["S", "S"], "frac", strides=[3, 3])        # [1,1,9,9] f32
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["frac", "half"], "M")                            # [1,1,9,9] bool

    # ---- colour index cc (uint8 scalar) ----
    init("chvals", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    n("ReduceMax", ["input"], "pres", axes=[2, 3], keepdims=1)     # [1,10,1,1]
    n("Mul", ["pres", "chvals"], "ccparts")
    n("ReduceSum", ["ccparts"], "ccf", keepdims=1)                 # scalar cc float
    n("Cast", ["ccf"], "cc8", to=U8)

    # ---- label map L = M ? cc : 0  (uint8 9x9) ----
    init("v0", np.array(0, np.uint8), np.uint8)
    n("Where", ["M", "cc8", "v0"], "L9")                           # [1,1,9,9] uint8

    # ---- pad to 30x30 with sentinel 10 (outside all-false), final Equal ----
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 21, 21], np.int64), np.int64)
    init("padval", np.array(10, np.uint8), np.uint8)
    n("Pad", ["L9", "padpads", "padval"], "L", mode="constant")    # [1,1,30,30] uint8
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")                            # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "graph", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
