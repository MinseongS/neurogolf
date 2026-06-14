"""Task 156 (ARC-AGI 694f12f3): colour the interiors of two yellow rectangles by
size — the SMALLER gets blue(1), the LARGER gets red(2).

Rule (verified exact on all 265 stored examples + fresh arc-gen). The input has
exactly two solid yellow(4) rectangles, vertically separated (disjoint row
bands with at least one empty row between them — the generator places one above
`border` and the other below). The output keeps the yellow cells and recolours
each rectangle's interior (the inset-by-1 sub-rectangle) by its area: the
rectangle with the smaller interior area becomes blue(1), the larger red(2).
(Exactly one of {width, height} differs between the two rectangles, so the area
ordering is unambiguous.)

Fully separable graph (no flood-fill):
  * interior mask  I = (3x3 all-yellow neighbourhood) — one 3x3 Conv on the
    yellow channel thresholded at 9 -> bool [1,1,30,30].
  * rowint[r] = #interior cells in row r  ([1,1,30,1]).
  * split the two row-bands with a lower-triangular cumulative trick: a row is
    in the BOTTOM band iff a gap-after-the-top-band lies at or above it.
  * compare band areas (ReduceSum of rowint over each side) -> scalar bigbottom.
  * build a per-row colour one-hot Cvec[1,10,30,1] (blue rows / red rows) and
    write   output = Where(I, Cvec, input)   into the free output.
Only one float canvas (the Conv count) + two bool canvases; everything else is
1-D row vectors, so memory stays small.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

from ..builders import _model


def build(task):
    inits = []
    nodes = []
    vinfos = []

    def init(name, arr, dtype=np.float32):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=dtype), name))
        return name

    def n(op, inputs, out, **attrs):
        nodes.append(helper.make_node(op, inputs, [out], **attrs))
        return out

    def vi(name, dtype, shape):
        vinfos.append(helper.make_tensor_value_info(name, dtype, shape))
        return name

    # ---- interior mask via a 3x3 Conv counting yellow(ch4) cells -------------
    # weight [1,10,3,3]: ones on channel 4's 3x3 window, zero elsewhere.
    W = np.zeros((1, 10, 3, 3), np.float32)
    W[0, 4, :, :] = 1.0
    init("Wint", W)
    # pad=1 keeps the canvas size; edge cells get <9 (zero padding) so they are
    # never interior, which is correct.
    n("Conv", ["input", "Wint"], "cnt", kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    vi("cnt", TensorProto.FLOAT, [1, 1, 30, 30])

    init("th85", np.array(8.5, np.float32))
    n("Greater", ["cnt", "th85"], "Imask")                 # [1,1,30,30] bool
    vi("Imask", TensorProto.BOOL, [1, 1, 30, 30])
    # fp16 canvas (counts <=30 -> exact) halves the interior float plane.
    n("Cast", ["Imask"], "If", to=TensorProto.FLOAT16)     # [1,1,30,30] fp16
    vi("If", TensorProto.FLOAT16, [1, 1, 30, 30])

    # ---- per-row interior count rowint[1,1,30,1] ----------------------------
    n("ReduceSum", ["If"], "rowint16", axes=[3], keepdims=1)  # [1,1,30,1] fp16
    vi("rowint16", TensorProto.FLOAT16, [1, 1, 30, 1])
    n("Cast", ["rowint16"], "rowint", to=TensorProto.FLOAT)   # [1,1,30,1] f32
    vi("rowint", TensorProto.FLOAT, [1, 1, 30, 1])

    init("zero", np.array(0.0, np.float32))
    init("half", np.array(0.5, np.float32))
    n("Greater", ["rowint", "zero"], "Rb")                 # row has interior
    vi("Rb", TensorProto.BOOL, [1, 1, 30, 1])
    n("Cast", ["Rb"], "R", to=TensorProto.FLOAT)           # [1,1,30,1] f32
    vi("R", TensorProto.FLOAT, [1, 1, 30, 1])

    # ---- separate the two row-bands -----------------------------------------
    # Single shared inclusive lower-tri matrix Linc (900 params); the strict
    # cumulative is Linc@R - R.
    Linc = np.tril(np.ones((30, 30), np.float32)).reshape(1, 1, 30, 30)
    init("Linc", Linc)
    n("MatMul", ["Linc", "R"], "incR")                     # [1,1,30,1] f32
    vi("incR", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Sub", ["incR", "R"], "seenf")                       # strict: rows above
    vi("seenf", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Greater", ["seenf", "half"], "seenb")
    vi("seenb", TensorProto.BOOL, [1, 1, 30, 1])
    n("Cast", ["seenb"], "seen", to=TensorProto.FLOAT)
    vi("seen", TensorProto.FLOAT, [1, 1, 30, 1])

    # isgap[r] = seen[r] AND NOT R[r]   (a gap row below the top band)
    init("one", np.array(1.0, np.float32))
    n("Sub", ["one", "R"], "notR")                         # 1-R
    vi("notR", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Mul", ["seen", "notR"], "isgap")                    # [1,1,30,1] f32
    vi("isgap", TensorProto.FLOAT, [1, 1, 30, 1])

    # gapcum[r] = #gap rows at or above r  (reuse the inclusive lower-tri)
    n("MatMul", ["Linc", "isgap"], "gapcum")               # [1,1,30,1] f32
    vi("gapcum", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Greater", ["gapcum", "half"], "belowb")             # in bottom band
    vi("belowb", TensorProto.BOOL, [1, 1, 30, 1])
    n("Cast", ["belowb"], "below", to=TensorProto.FLOAT)   # [1,1,30,1] f32
    vi("below", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Sub", ["one", "below"], "above")                    # top band side
    vi("above", TensorProto.FLOAT, [1, 1, 30, 1])

    # ---- compare band areas -> bigbottom scalar -----------------------------
    n("Mul", ["rowint", "below"], "botrows")
    vi("botrows", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Mul", ["rowint", "above"], "toprows")
    vi("toprows", TensorProto.FLOAT, [1, 1, 30, 1])
    n("ReduceSum", ["botrows"], "botcnt", axes=[2], keepdims=1)  # [1,1,1,1]
    vi("botcnt", TensorProto.FLOAT, [1, 1, 1, 1])
    n("ReduceSum", ["toprows"], "topcnt", axes=[2], keepdims=1)  # [1,1,1,1]
    vi("topcnt", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Greater", ["botcnt", "topcnt"], "bbb")              # bottom larger?
    vi("bbb", TensorProto.BOOL, [1, 1, 1, 1])
    n("Cast", ["bbb"], "bb", to=TensorProto.FLOAT)         # [1,1,1,1] f32
    vi("bb", TensorProto.FLOAT, [1, 1, 1, 1])

    # ---- per-row red/blue indicator -----------------------------------------
    # redrow[r] = below[r]*bb + (1-below[r])*(1-bb)   (red = larger band)
    n("Sub", ["one", "bb"], "nbb")                         # 1-bb  [1,1,1,1]
    vi("nbb", TensorProto.FLOAT, [1, 1, 1, 1])
    n("Mul", ["below", "bb"], "t1")                        # broadcast [1,1,30,1]
    vi("t1", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Mul", ["above", "nbb"], "t2")                       # [1,1,30,1]
    vi("t2", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Add", ["t1", "t2"], "redrow")                       # [1,1,30,1] f32
    vi("redrow", TensorProto.FLOAT, [1, 1, 30, 1])
    n("Sub", ["one", "redrow"], "bluerow")                 # [1,1,30,1]
    vi("bluerow", TensorProto.FLOAT, [1, 1, 30, 1])

    # ---- colour one-hot Cvec[1,10,30,1] = ch1*bluerow + ch2*redrow ----------
    oh1 = np.zeros((1, 10, 1, 1), np.float32); oh1[0, 1, 0, 0] = 1.0
    oh2 = np.zeros((1, 10, 1, 1), np.float32); oh2[0, 2, 0, 0] = 1.0
    init("oh1", oh1, np.float16)
    init("oh2", oh2, np.float16)
    # fp16 colour planes (values 0/1) halve these canvases; Cvec cast back to
    # f32 for the final Where (both Where branches must match input dtype).
    n("Cast", ["bluerow"], "bluerow16", to=TensorProto.FLOAT16)
    vi("bluerow16", TensorProto.FLOAT16, [1, 1, 30, 1])
    n("Cast", ["redrow"], "redrow16", to=TensorProto.FLOAT16)
    vi("redrow16", TensorProto.FLOAT16, [1, 1, 30, 1])
    n("Mul", ["oh1", "bluerow16"], "cb")                   # [1,10,30,1] fp16
    vi("cb", TensorProto.FLOAT16, [1, 10, 30, 1])
    n("Mul", ["oh2", "redrow16"], "cr")                    # [1,10,30,1] fp16
    vi("cr", TensorProto.FLOAT16, [1, 10, 30, 1])
    n("Add", ["cb", "cr"], "Cvec16")                       # [1,10,30,1] fp16
    vi("Cvec16", TensorProto.FLOAT16, [1, 10, 30, 1])
    n("Cast", ["Cvec16"], "Cvec", to=TensorProto.FLOAT)    # [1,10,30,1] f32
    vi("Cvec", TensorProto.FLOAT, [1, 10, 30, 1])

    # ---- output = Where(Imask, Cvec, input) ---------------------------------
    # interior cells are yellow in the input; overwrite them with the per-row
    # colour one-hot (blue/red); everything else stays as input.
    n("Where", ["Imask", "Cvec", "input"], "output")       # [1,10,30,30] f32

    return _model(nodes, inits, vinfos)
