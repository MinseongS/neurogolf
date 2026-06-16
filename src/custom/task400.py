"""Task 400 (ff805c23): D4-symmetric pattern with a 5x5 blue cutout.

Rule (from ARC-GEN generator): a 24x24 grid carries a pattern with full D4
dihedral symmetry (the 8-element orbit {(r,c),(c,r),(r,n-c),(n-c,r),(n-r,c),
(c,n-r),(n-r,n-c),(n-c,n-r)} with n=23 all share one colour).  A 5x5 block of
blue (colour 1) is stamped over part of the pattern, occluding it; the generator
guarantees every occluded cell still has at least one NON-blue orbit member.
The output is the 5x5 region under the cutout, reconstructed from symmetry.

Reconstruction:
  colf = sum_k k*input_k                       (colour-index plane, [1,1,30,30])
  slice colf and the blue channel to the 24x24 active grid (fp16 from here -- the
  colour indices 0..9 and the sentinels -1 / 10 are all integer-exact in fp16).
  val = where(blue, -1, colf)                  (blue cells -> sentinel that loses)
  The 8 D4 pullbacks of `val` are exactly {I, T} x {none, flipR, flipC, flipRC}
  (verified): for cell (i,j) the value is read from a symmetric image, so
  P[i,j] = max over the 8 transformed planes of val.  All non-blue orbit members
  equal the true colour c>=0; blue ones give -1; the max recovers c.  (Flips are
  step=-1 Slices, transpose is a Transpose -- all 0 params.)
  brow = first blue row, bcol = first blue col (ArgMax of blue row/col presence).
  crop P[brow:brow+5, bcol:bcol+5], pad to 30x30 with sentinel 10 (off-grid cells
  must be all-False), output = Equal(L30, arange[0..9])  ([1,10,30,30] BOOL).

Verified exact: numpy reference 0/200, ONNX stored 266/266, fresh 200/200.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F16 = TensorProto.FLOAT16


def build(task):
    inits, nodes = [], []

    def init(name, arr, dtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    # ---------- constants ----------
    init("kw", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("k10u", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    init("neg1", np.array(-1.0, np.float16), np.float16)
    init("off05", np.arange(5, dtype=np.int64), np.int64)        # [5]

    # colour-index plane (FREE input -> [1,1,30,30] fp32), then cast to fp16.
    n("Conv", ["input", "kw"], "colf32")                        # [1,1,30,30] fp32
    n("Cast", ["colf32"], "colf", to=F16)                       # [1,1,30,30] fp16

    # slice colf to the 24x24 active grid
    init("s0", np.array([0, 0], np.int64), np.int64)
    init("s24", np.array([24, 24], np.int64), np.int64)
    init("ax23", np.array([2, 3], np.int64), np.int64)
    n("Slice", ["colf", "s0", "s24", "ax23"], "g")             # [1,1,24,24] fp16

    # blue (colour 1) appears ONLY in the cutout (pattern excludes blue), so the
    # cutout mask is just g == 1 -- no separate blue-channel slice needed.
    init("one16", np.array(1.0, np.float16), np.float16)
    n("Equal", ["g", "one16"], "bluemask")                     # bool [1,1,24,24]

    # val = where(blue, -1, g)   (fp16)
    n("Where", ["bluemask", "neg1", "g"], "val")              # [1,1,24,24] fp16

    # ---------- locate cutout (brow, bcol) ----------
    # val == -1 exactly at blue cells; a row/col contains blue iff its min is < 0.
    # Reduce val (already materialized) -> tiny [1,1,24] vectors, no extra plane.
    init("zero16", np.array(0.0, np.float16), np.float16)
    n("ReduceMin", ["val"], "rmin", axes=[3], keepdims=0)      # [1,1,24] fp16
    n("ReduceMin", ["val"], "cmin", axes=[2], keepdims=0)      # [1,1,24]
    n("Less", ["rmin", "zero16"], "rp_b")                      # [1,1,24] bool
    n("Less", ["cmin", "zero16"], "cp_b")
    n("Cast", ["rp_b"], "rp", to=F16)                          # [1,1,24] fp16
    n("Cast", ["cp_b"], "cp", to=F16)
    n("ArgMax", ["rp"], "brow1", axis=2, keepdims=0)           # [1,1] int64
    n("ArgMax", ["cp"], "bcol1", axis=2, keepdims=0)           # [1,1]
    n("Squeeze", ["brow1"], "brow", axes=[0, 1])               # scalar int64
    n("Squeeze", ["bcol1"], "bcol", axes=[0, 1])               # scalar int64

    # Rather than symmetrize the whole 24x24 plane (8 full planes + 7 maxes), only
    # crop the 5x5 cutout region from each of the 8 orbit images directly.  Each
    # D4 transform reads val/valT at a 5x5 block whose row/col indices are either
    # ascending  [b .. b+4]   or descending  [23-b .. 23-b-4].
    init("c23", np.array(23, np.int64), np.int64)
    n("Add", ["off05", "brow"], "ridx_p")                      # [b..b+4]
    n("Add", ["off05", "bcol"], "cidx_p")
    n("Sub", ["c23", "brow"], "r23")                           # 23-b scalar
    n("Sub", ["c23", "bcol"], "c23c")
    n("Sub", ["r23", "off05"], "ridx_m")                       # [23-b .. 23-b-4]
    n("Sub", ["c23c", "off05"], "cidx_m")

    # gather a 5x5 block from `val` given row/col index vectors.
    def crop(ridx, cidx, tag):
        n("Gather", ["val", ridx], f"{tag}_r", axis=2)         # [1,1,5,24]
        n("Gather", [f"{tag}_r", cidx], f"{tag}_b", axis=3)    # [1,1,5,5]
        return f"{tag}_b"

    # transpose-group block: valT[ri,ci] = val[ci,ri], so gather val with rows=ci,
    # cols=ri then transpose the tiny 5x5 result (avoids materializing valT).
    def cropT(ridx, cidx, tag):
        n("Gather", ["val", cidx], f"{tag}_r", axis=2)         # [1,1,5,24]
        n("Gather", [f"{tag}_r", ridx], f"{tag}_g", axis=3)    # [1,1,5,5]
        n("Transpose", [f"{tag}_g"], f"{tag}_b", perm=[0, 1, 3, 2])
        return f"{tag}_b"

    blocks = [
        # val group: identity, flipR, flipC, flipRC
        crop("ridx_p", "cidx_p", "v0"),
        crop("ridx_m", "cidx_p", "v1"),
        crop("ridx_p", "cidx_m", "v2"),
        crop("ridx_m", "cidx_m", "v3"),
        # transpose group
        cropT("ridx_p", "cidx_p", "t0"),
        cropT("ridx_m", "cidx_p", "t1"),
        cropT("ridx_p", "cidx_m", "t2"),
        cropT("ridx_m", "cidx_m", "t3"),
    ]
    cur = blocks[0]
    for i, b in enumerate(blocks[1:]):
        cur = n("Max", [cur, b], f"mx{i}")
    # cur is the final [1,1,5,5] fp16 crop P5

    # Cast the 5x5 crop to uint8 BEFORE padding so the 30x30 label plane is uint8
    # (900B) not fp16 (1800B); pad value 10 (sentinel -> matches no colour).
    n("Cast", [cur], "P5u", to=TensorProto.UINT8)              # [1,1,5,5] uint8
    init("pads", np.array([0, 0, 0, 0, 0, 0, 25, 25], np.int64), np.int64)
    init("sent", np.array(10, np.uint8), np.uint8)
    n("Pad", ["P5u", "pads", "sent"], "L30", mode="constant")  # [1,1,30,30] uint8

    n("Equal", ["L30", "k10u"], "output")                      # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    graph = helper.make_graph(nodes, "task400", [x], [y], inits)
    return helper.make_model(
        graph, ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", 11)])
