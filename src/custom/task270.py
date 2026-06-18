"""Task 270 (ARC-AGI ae3edfdc) — two "flowers", pull petals back to the centre.

Rule (from the generator): a fixed 15x15 grid (top-left of the 30x30 canvas),
background 0, holds two flowers.
  * Flower 0: centre = a single colour-2 pixel, petals = colour 3.
  * Flower 1: centre = a single colour-1 pixel, petals = colour 7.
In each of the 4 orthogonal directions a flower MAY have one petal placed
somewhere along that ray at distance >= 2 ("it flew off"). The OUTPUT keeps both
centres fixed and moves every existing petal to the cell immediately ADJACENT to
its centre in that direction.

Because colour-3 pixels belong ONLY to flower 0 and colour-7 ONLY to flower 1,
and every petal sits on a ray through its centre, the WHOLE output is a function
of 12 scalars: the two centres (r2,c2),(r1,c1) and, per flower, 4 boolean
direction flags.  Each flag is a pure 1-D-profile test (NO 2-D plane):

  rowprof = ReduceSum(petal_channel, axis=col)   # [1,10,30,1]
  colprof = ReduceSum(petal_channel, axis=row)   # [1,10,1,30]
  up = any rowprof at rows < r ; dn = rows > r ; lf = colprof cols < c ; rt = >c

This is exact because vertical petals (col==c) only ever sit at row != r so they
land in rowprof at rows < r / > r and never collide with horizontal petals (all
at row == r, touching only rowprof[r]); symmetrically for colprof.

Reconstruction: there are 10 candidate output cells (2 centres + 8 petals).  Each
cell k has a target (row_k, col_k), a colour col_k, and an activity weight w_k
(centres always 1; petals = direction flag in {0,1}).  Because every cell is a
rank-1 placement, the whole label plane is a SINGLE small matrix product:

  RS[r,k] = Equal(rampR[r], row_k)          [15,10]
  CW[k,c] = (col_k * w_k) * Equal(col_k_pos, rampC[c])   [10,15]
  L[r,c]  = (RS @ CW)[r,c]                   [15,15]   (cells disjoint -> exact)

L (fp16, 15x15 = 450B) is then Cast to uint8, Pad'd to 30x30 with off-grid
sentinel 99, and Equal(L, arange[1,10,1,1]) routes the full 10-channel one-hot
into the FREE bool output.  No working tensor exceeds the 1200B row/col profile.
Verified fresh 500/500.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F32 = TensorProto.FLOAT
F16 = TensorProto.FLOAT16
I64 = TensorProto.INT64
U8 = TensorProto.UINT8
B = TensorProto.BOOL

N = 15  # active grid extent (generator size)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---- per-channel row/col profiles straight from the FREE input ----
    n("ReduceSum", ["input"], "rowprofA", axes=[3], keepdims=1)  # [1,10,30,1] fp32
    n("ReduceSum", ["input"], "colprofA", axes=[2], keepdims=1)  # [1,10,1,30] fp32

    # fp16 ramps (length 30 for profiles; length 15 for the in-grid placement)
    ramp = np.arange(30, dtype=np.float16)
    init("rampR", ramp.reshape(1, 1, 30, 1), np.float16)  # [1,1,30,1]
    init("rampC", ramp.reshape(1, 1, 1, 30), np.float16)  # [1,1,1,30]
    rampN = np.arange(N, dtype=np.float16)
    init("rampRN", rampN.reshape(1, 1, N, 1), np.float16)  # [1,1,15,1]
    init("rampCN", rampN.reshape(1, 1, 1, N), np.float16)  # [1,1,1,15]

    def chan_slice(prof, k, name):
        init(name + "_a", np.array([k], np.int64), np.int64)
        init(name + "_b", np.array([k + 1], np.int64), np.int64)
        init(name + "_x", np.array([1], np.int64), np.int64)
        return n("Slice", [prof, name + "_a", name + "_b", name + "_x"], name)

    # centre (r,c) for a centre colour kc (single pixel) -> fp16 scalars
    def centre_rc(kc, tag):
        rp = chan_slice("rowprofA", kc, tag + "rp")  # [1,1,30,1] fp32
        cp = chan_slice("colprofA", kc, tag + "cp")  # [1,1,1,30] fp32
        n("Cast", [rp], tag + "rph", to=F16)
        n("Cast", [cp], tag + "cph", to=F16)
        n("Mul", [tag + "rph", "rampR"], tag + "rpw")
        n("ReduceSum", [tag + "rpw"], tag + "r", axes=[2, 3], keepdims=1)  # [1,1,1,1] fp16
        n("Mul", [tag + "cph", "rampC"], tag + "cpw")
        n("ReduceSum", [tag + "cpw"], tag + "c", axes=[2, 3], keepdims=1)  # [1,1,1,1] fp16
        return tag + "r", tag + "c"

    r0, c0 = centre_rc(2, "f0")
    r1, c1 = centre_rc(1, "f1")

    # petal profiles (fp16)
    def petal_profiles(kp, tag):
        rp = chan_slice("rowprofA", kp, tag + "prp")
        cp = chan_slice("colprofA", kp, tag + "pcp")
        n("Cast", [rp], tag + "prph", to=F16)
        n("Cast", [cp], tag + "pcph", to=F16)
        return tag + "prph", tag + "pcph"

    pr0, pc0 = petal_profiles(3, "f0")
    pr1, pc1 = petal_profiles(7, "f1")

    # direction flags (fp16 {0,1} scalars)
    def flags(rp, cp, r, c, tag):
        n("Less", ["rampR", r], tag + "ltR")     # rows < r  [1,1,30,1] bool
        n("Greater", ["rampR", r], tag + "gtR")  # rows > r
        n("Less", ["rampC", c], tag + "ltC")
        n("Greater", ["rampC", c], tag + "gtC")

        def msum(mask, prof, nm):
            n("Cast", [mask], nm + "m", to=F16)
            n("Mul", [nm + "m", prof], nm + "x")
            n("ReduceSum", [nm + "x"], nm + "s", axes=[2, 3], keepdims=1)  # [1,1,1,1] fp16
            # -> {0,1}: Greater(.,0)
            n("Greater", [nm + "s", "zero"], nm + "b")
            n("Cast", [nm + "b"], nm, to=F16)
            return nm

        up = msum(tag + "ltR", rp, tag + "up")
        dn = msum(tag + "gtR", rp, tag + "dn")
        lf = msum(tag + "ltC", cp, tag + "lf")
        rt = msum(tag + "gtC", cp, tag + "rt")
        return up, dn, lf, rt

    init("zero", np.array([[[[0.0]]]], np.float16), np.float16)
    init("one", np.array([[[[1.0]]]], np.float16), np.float16)
    init("p1", np.array([[[[1.0]]]], np.float16), np.float16)
    init("m1", np.array([[[[-1.0]]]], np.float16), np.float16)
    init("col2", np.array([[[[2.0]]]], np.float16), np.float16)
    init("col1", np.array([[[[1.0]]]], np.float16), np.float16)
    init("col3", np.array([[[[3.0]]]], np.float16), np.float16)
    init("col7", np.array([[[[7.0]]]], np.float16), np.float16)

    up0, dn0, lf0, rt0 = flags(pr0, pc0, r0, c0, "f0")
    up1, dn1, lf1, rt1 = flags(pr1, pc1, r1, c1, "f1")

    # ---- assemble the 10 candidate cells as packed [1,1,1,10] vectors ----
    # Each cell: row_k, col_k along a length-10 axis (axis=3), colour*weight w_k.
    # We build RS[r,k] and CW[k,c] then L = RS @ CW.
    # cell order: [F0 C,U,D,L,R, F1 C,U,D,L,R]
    def add1(s, d, tag):  # s + d (fp16 scalars)
        return n("Add", [s, d], tag)

    rU0 = add1(r0, "m1", "rU0"); rD0 = add1(r0, "p1", "rD0")
    cL0 = add1(c0, "m1", "cL0"); cR0 = add1(c0, "p1", "cR0")
    rU1 = add1(r1, "m1", "rU1"); rD1 = add1(r1, "p1", "rD1")
    cL1 = add1(c1, "m1", "cL1"); cR1 = add1(c1, "p1", "cR1")

    # rows per cell -> concat along axis=3 -> [1,1,1,10]
    rows_cells = [r0, rU0, rD0, r0, r0, r1, rU1, rD1, r1, r1]
    cols_cells = [c0, c0, c0, cL0, cR0, c1, c1, c1, cL1, cR1]
    n("Concat", rows_cells, "rowK", axis=3)  # [1,1,1,10]
    n("Concat", cols_cells, "colK", axis=3)  # [1,1,1,10]

    # weights w_k = colour_k * activity_k
    init("oneA", np.array([[[[1.0]]]], np.float16), np.float16)
    # colour*weight for each cell
    def w(colour, gate, tag):
        if gate is None:
            return colour  # centre always-on
        return n("Mul", [colour, gate], tag)

    wcells = [
        "col2",                       # F0 centre
        w("col3", up0, "wU0"), w("col3", dn0, "wD0"),
        w("col3", lf0, "wL0"), w("col3", rt0, "wR0"),
        "col1",                       # F1 centre
        w("col7", up1, "wU1"), w("col7", dn1, "wD1"),
        w("col7", lf1, "wL1"), w("col7", rt1, "wR1"),
    ]
    n("Concat", wcells, "wK", axis=3)  # [1,1,1,10]

    # ---- RS[r,k] = Equal(rampRN[r], rowK[k]) -> [1,1,15,10] ----
    n("Equal", ["rampRN", "rowK"], "RSb")  # [1,1,15,10] bool
    n("Cast", ["RSb"], "RS", to=F16)       # [1,1,15,10] fp16
    # CW[k,c]: Equal(colK[k], rampCN[c]) -> [1,1,10,15]
    init("shp_k1", np.array([1, 1, 10, 1], np.int64), np.int64)
    n("Reshape", ["colK", "shp_k1"], "colK2")     # [1,1,10,1]
    n("Equal", ["colK2", "rampCN"], "CSb")         # [1,1,10,15] bool
    n("Cast", ["CSb"], "CS", to=F16)               # [1,1,10,15] fp16
    n("Reshape", ["wK", "shp_k1"], "wK2")          # [1,1,10,1]
    n("Mul", ["CS", "wK2"], "CW")                  # [1,1,10,15] fp16

    # ---- L = RS @ CW : [1,1,15,10] @ [1,1,10,15] -> [1,1,15,15] ----
    n("MatMul", ["RS", "CW"], "L")                 # [1,1,15,15] fp16
    # cells are disjoint so L holds exactly the colour index per cell.

    # ---- route into free output: Cast uint8, Pad sentinel 99, Equal vs arange ----
    n("Cast", ["L"], "Lu", to=U8)                  # [1,1,15,15] uint8
    pads = np.array([0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64)
    init("pads", pads, np.int64)
    init("sent", np.array(99, np.uint8), np.uint8)
    n("Pad", ["Lu", "pads", "sent"], "Lp", mode="constant")  # [1,1,30,30] uint8
    arange = np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)
    init("arange", arange, np.uint8)
    n("Equal", ["Lp", "arange"], "output")          # [1,10,30,30] bool

    out_vi = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    in_vi = helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task270", [in_vi], [out_vi], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
