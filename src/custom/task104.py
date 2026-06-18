"""Task 104 (ARC-AGI 4522001f) — quadrant-selected 9x9 fixed pattern.

Rule (from the generator): the input is a fixed 3x3 grid encoding one of FOUR
"quadrant" cases q in {0,1,2,3}.  The centre is always red(2); exactly ONE of the
four corner cells [0,0],[0,2],[2,0],[2,2] is green(3) and that corner index IS the
quadrant (q0=[0,0], q1=[0,2], q2=[2,0], q3=[2,2]).  The OUTPUT is always a 9x9 grid
made of green(3)/black(0) only, and is one of FOUR fixed patterns selected solely
by q.  Cells outside the top-left 9x9 footprint are all-channels-off.

Recovery (closed-form, no detection):
  * qvec[4] = the four corner cells of the green channel (channel 3) — a single
    Conv with a [4,1,3,3] corner-picker kernel applied to the 3x3 channel-3 slice
    gives [1,4,1,1]; exactly one entry is 1.
  * The green 9x9 mask is a fixed table lookup: G = qvec @ Mflat, where Mflat is
    the [4,81] matrix of the four precomputed 9x9 green masks.  G is a tiny [1,81]
    intermediate (324B) reshaped to [1,1,9,9].
  * label L (9x9 uint8) = 3 where G>0 else 0; pad to 30x30 with off-grid sentinel
    99 (matches no channel) and the FREE BOOL output is Equal(L, arange[1,10,1,1]).

Memory floor: the lone 30x30 plane is the padded uint8 label map (900 B); every
other intermediate is <=324 B.  Pad rejects bool so uint8 900B is the label floor.
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

    # ---- slice green channel (3) top-left 3x3 ----
    init("ss", np.array([0, 3, 0, 0], np.int64), np.int64)
    init("se", np.array([1, 4, 3, 3], np.int64), np.int64)
    init("sax", np.array([0, 1, 2, 3], np.int64), np.int64)
    n("Slice", ["input", "ss", "se", "sax"], "g3")  # [1,1,3,3] fp32 (36B)

    # ---- corner-picker Conv -> qvec [1,4,1,1] ----
    # kernel k has a single 1 at corner k (q0=[0,0],q1=[0,2],q2=[2,0],q3=[2,2]).
    K = np.zeros((4, 1, 3, 3), np.float32)
    K[0, 0, 0, 0] = 1.0
    K[1, 0, 0, 2] = 1.0
    K[2, 0, 2, 0] = 1.0
    K[3, 0, 2, 2] = 1.0
    init("Kc", K, np.float32)
    n("Conv", ["g3", "Kc"], "qvec4")  # [1,4,1,1] fp32, exactly one entry == 1

    # ---- table lookup: G = qvec @ Mflat ----
    masks = np.zeros((4, 81), np.float32)
    # q0
    M0 = np.zeros((9, 9), np.float32); M0[0:4, 0:4] = 1; M0[4:8, 4:8] = 1
    # q1
    M1 = np.zeros((9, 9), np.float32); M1[0:4, 5:9] = 1; M1[4:8, 1:5] = 1
    # q2
    M2 = np.zeros((9, 9), np.float32); M2[1:5, 4:8] = 1; M2[5:9, 0:4] = 1
    # q3
    M3 = np.zeros((9, 9), np.float32); M3[1:5, 1:5] = 1; M3[5:9, 5:9] = 1
    masks[0] = M0.reshape(81); masks[1] = M1.reshape(81)
    masks[2] = M2.reshape(81); masks[3] = M3.reshape(81)
    init("Mflat", masks, np.float32)  # [4,81]

    init("q4", np.array([1, 4], np.int64), np.int64)
    n("Reshape", ["qvec4", "q4"], "qrow")  # [1,4]
    n("MatMul", ["qrow", "Mflat"], "Gflat")  # [1,81] fp32 (324B)

    init("g99", np.array([1, 1, 9, 9], np.int64), np.int64)
    n("Reshape", ["Gflat", "g99"], "G")  # [1,1,9,9] fp32

    # ---- label map L (9x9 uint8): 3 where green else 0 ----
    init("half", np.array(0.5, np.float32), np.float32)
    n("Greater", ["G", "half"], "Gb")  # [1,1,9,9] bool
    init("u3", np.array(3, np.uint8), np.uint8)
    init("u0", np.array(0, np.uint8), np.uint8)
    n("Where", ["Gb", "u3", "u0"], "L9")  # [1,1,9,9] uint8 (81B)

    # pad 9x9 -> 30x30 with off-grid sentinel 99 (matches no channel 0..9).
    init("u99", np.array(99, np.uint8), np.uint8)
    init("pads", np.array([0, 0, 0, 0, 0, 0, 21, 21], np.int64), np.int64)
    n("Pad", ["L9", "pads", "u99"], "L", mode="constant")  # [1,1,30,30] uint8 (900B)

    chan = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("chan", chan, np.uint8)
    n("Equal", ["L", "chan"], "output")  # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    gr = helper.make_graph(nodes, "task104", [x], [y], inits)
    return helper.make_model(gr, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
