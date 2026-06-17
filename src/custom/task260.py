"""task260 (ARC-AGI a78176bb) — "parallel diagonal echo".

Rule (from the ARC-GEN generator, verified fresh 20000/20000 + validate set):
  INPUT (always 10x10): a single solid main diagonal `r-c == diag` painted in
  one colour (never gray), plus gray triangle(s) hanging off the diagonal.  The
  gray cells of one corner occupy a contiguous band of diagonals from `diag+/-1`
  out to an extreme diagonal `e` (the corner's `row-col`).
  OUTPUT: the same colour painted on the original `r-c == diag` PLUS one echo
  per occupied side:
        gray ABOVE diag (r-c > diag): echo `r-c == (max gray-above) + 2`
        gray BELOW diag (r-c < diag): echo `r-c == (min gray-below) - 2`
  No gray in the output.  (Random instances have one corner; the hand-authored
  train cases straddle the diagonal with two corners -> two echoes.)

Closed-form recovery (no flood-fill, no 10-ch 30x30 plane):
  * cnt = ReduceSum(input, axes=[2,3])  -> [1,10,1,1] per-channel pixel counts.
  * cval = ReduceMax(idx where cnt>0 and idx!=0 and idx!=5)  -> colour scalar.
  * bgch / graych = channel 0 / 5 sliced to the 10x10 active crop.
  * colormask = (bgch==0) AND (graych==0)  (== the colour diagonal cells).
  * D[r,c] = r-c constant ; diag = ReduceMax(D where colormask).
  * maxg = ReduceMax(D where gray AND D>diag)  (floor -100 -> out of range).
    ming = ReduceMin(D where gray AND D<diag)  (floor +100 -> out of range).
  * mask = (D==diag) OR (D==maxg+2) OR (D==ming-2).
  * L = cval where mask else sentinel ; Pad to 30x30 with sentinel 99.
  * output = Equal(L_fp16, arange[0..9]) -> BOOL one-hot (free output tensor).

Memory: dominant intermediate is the single fp16 30x30 colour-index plane
(1800B) right before the final Equal; all mask logic is on the 10x10 crop.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

W = 10  # active canvas (grid is always 10x10 anchored top-left)


def build(task):
    inits, nodes = [], []

    def init(name, arr, dt):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=dt), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, list(ins), [out], **attrs))
        return out

    F = TensorProto.FLOAT
    H = TensorProto.FLOAT16
    B = TensorProto.BOOL

    # ---- constants ----
    # channel index ramp for the count-based colour readout
    init("idx", np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1), np.float32)
    init("zero", np.array(0.0, np.float32), np.float32)
    init("two", np.array(2.0, np.float16), np.float16)
    init("bigneg", np.array(-100.0, np.float16), np.float16)
    init("bigpos", np.array(100.0, np.float16), np.float16)
    # masks to exclude channels 0 and 5 from the colour search
    okidx = ((np.arange(10) != 0) & (np.arange(10) != 5)).astype(np.float32)
    init("okidx", okidx.reshape(1, 10, 1, 1), np.float32)   # 1 on real colour channels
    # slices of channels 0 / 5 down to the 10x10 active crop
    init("bg_s", np.array([0, 0, 0, 0], np.int64), np.int64)
    init("bg_e", np.array([1, 1, W, W], np.int64), np.int64)
    init("gr_s", np.array([0, 5, 0, 0], np.int64), np.int64)
    init("gr_e", np.array([1, 6, W, W], np.int64), np.int64)
    # D[r,c] = r - c  on 10x10
    D = (np.arange(W).reshape(W, 1) - np.arange(W).reshape(1, W)).astype(np.float16)
    init("D", D.reshape(1, 1, W, W), np.float16)
    # pad 10x10 -> 30x30 with sentinel 99 (>9 -> no channel matches in padding)
    init("padpads", np.array([0, 0, 0, 0, 0, 0, 30 - W, 30 - W], np.int64), np.int64)
    init("padval", np.array(99.0, np.float16), np.float16)
    init("zeroH", np.array(0.0, np.float16), np.float16)  # in-grid bg -> channel 0
    # final one-hot comparator (fp16)
    init("chan", np.arange(10, dtype=np.float16).reshape(1, 10, 1, 1), np.float16)

    # ---- colour scalar from per-channel counts (40B reduction, no plane) ----
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)  # [1,10,1,1]
    n("Greater", ["cnt", "zero"], "present")                   # cnt>0  [1,10,1,1]
    # restrict to real colour channels (exclude 0 and 5)
    n("Cast", ["present"], "presentf", to=F)
    n("Mul", ["presentf", "okidx"], "okpresent")               # 1 only on the colour ch
    n("Mul", ["okpresent", "idx"], "cidx")                     # idx on colour ch else 0
    n("ReduceMax", ["cidx"], "cvalF", keepdims=1)              # [1,1,1,1] colour value
    n("Cast", ["cvalF"], "cval", to=H)

    # ---- per-cell masks on the 10x10 crop (channels 0 and 5) ----
    n("Slice", ["input", "bg_s", "bg_e"], "bgch")              # [1,1,10,10]
    n("Slice", ["input", "gr_s", "gr_e"], "graych")            # [1,1,10,10]
    n("Equal", ["bgch", "zero"], "notbg")
    n("Equal", ["graych", "zero"], "notgray")
    n("And", ["notbg", "notgray"], "colormask")                # colour diagonal cells
    n("Greater", ["graych", "zero"], "graymask")               # gray cells

    # ---- diag scalar ----
    n("Where", ["colormask", "D", "bigneg"], "Dcol")
    n("ReduceMax", ["Dcol"], "diag", keepdims=1)               # [1,1,1,1]

    # ---- gray extent per side of diag (auto-gated by out-of-range floors) ----
    n("Greater", ["D", "diag"], "Dabove")
    n("Less", ["D", "diag"], "Dbelow")
    n("And", ["graymask", "Dabove"], "grayabove")
    n("And", ["graymask", "Dbelow"], "graybelow")
    n("Where", ["grayabove", "D", "bigneg"], "Dgmax")
    n("ReduceMax", ["Dgmax"], "maxg", keepdims=1)
    n("Where", ["graybelow", "D", "bigpos"], "Dgmin")
    n("ReduceMin", ["Dgmin"], "ming", keepdims=1)
    n("Add", ["maxg", "two"], "secA")                          # echo above
    n("Sub", ["ming", "two"], "secB")                          # echo below

    # ---- diagonal mask: original + both echoes ----
    n("Equal", ["D", "diag"], "isd")
    n("Equal", ["D", "secA"], "issa")
    n("Equal", ["D", "secB"], "issb")
    n("Or", ["isd", "issa"], "m1")
    n("Or", ["m1", "issb"], "mask")                            # [1,1,10,10] bool

    # ---- fp16 colour-index plane, padded to 30x30 with sentinel ----
    n("Where", ["mask", "cval", "zeroH"], "L10")               # [1,1,10,10] fp16; bg=0
    n("Pad", ["L10", "padpads", "padval"], "L", mode="constant")  # [1,1,30,30] fp16

    # ---- one-hot bool output ----
    n("Equal", ["L", "chan"], "output")                        # [1,10,30,30] bool

    graph = helper.make_graph(
        nodes, "task260",
        [helper.make_tensor_value_info("input", F, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", B, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = IR_VERSION
    return model
