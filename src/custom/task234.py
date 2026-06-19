"""Task 234 (ARC 98cf29f8): "frog eats fly".

Two solid rectangles — the frog (colors[0]) and the smaller fly (colors[1]) —
joined by a 1-wide tongue of the fly's colour.  Output keeps the frog fixed,
deletes the tongue, and slides the fly box along the tongue axis until it is
flush against the frog.  Orientation-equivariant (flip/xpose).

Lean build:
  * the ONLY per-channel [1,10,30,*] tensors are the two fp32 spatial sums of the
    input (row counts + col counts).  Frog/fly projections are pulled out as
    [1,1,30,*] slices with Gather(sum, channelIdx, axis=1) — NO [1,10,30,*]
    masked product planes (the prior +0.30 lever).
  * frog vs fly identified by the 1-wide TONGUE: the fly colour has a per-line
    (row OR col) min-positive count of 1; the frog's min dimension >= 3.  This
    needs only the two fp32 sums (no separate bbox-area span planes).
  * output = 3 separable rectangles (frog box, moved fly box, grid bg) -> one
    uint8 label map L -> Equal(L, arange) into the FREE bool output.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper
from ..harness import IR_VERSION

BIG = 1000.0


def build(task):
    inits, nodes = [], []
    FT = onnx.TensorProto.FLOAT16
    TU8 = onnx.TensorProto.UINT8
    TI32 = onnx.TensorProto.INT32

    def init(name, arr, npd=np.float16):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=npd), name))
        return name

    def n(op, inp, out, **a):
        nodes.append(helper.make_node(op, inp, [out], **a)); return out

    init("ri", np.arange(30).reshape(1, 1, 30, 1))
    init("ci", np.arange(30).reshape(1, 1, 1, 30))
    init("one", np.array(1.0))
    init("big", np.array(BIG))
    init("half", np.array(0.5))
    init("cidx", np.arange(10).reshape(1, 10, 1, 1))
    m0 = np.ones((1, 10, 1, 1)); m0[0, 0] = 0.0
    init("m0", m0)
    init("i1", np.array(1, np.int32), np.int32)
    init("half32", np.array(0.5, np.float32), np.float32)

    # ---- the only two per-channel [1,10,30,*] tensors (fp32) ----
    n("ReduceSum", ["input"], "rowSumf", axes=[3], keepdims=1)   # [1,10,30,1]
    n("ReduceSum", ["input"], "colSumf", axes=[2], keepdims=1)   # [1,10,1,30]
    n("Cast", ["rowSumf"], "rowSum", to=FT)
    n("Cast", ["colSumf"], "colSum", to=FT)

    # per-channel presence + min-positive per-line counts (tongue detector)
    n("ReduceSum", ["rowSumf"], "cntf", axes=[2], keepdims=1)
    n("Greater", ["cntf", "half32"], "posb"); n("Cast", ["posb"], "pos", to=FT)   # [1,10,1,1]
    n("Greater", ["rowSum", "half"], "rPos"); n("Where", ["rPos", "rowSum", "big"], "rwPos")
    n("ReduceMin", ["rwPos"], "minPosRow", axes=[2, 3], keepdims=1)
    n("Greater", ["colSum", "half"], "cPos"); n("Where", ["cPos", "colSum", "big"], "cwPos")
    n("ReduceMin", ["cwPos"], "minPosCol", axes=[2, 3], keepdims=1)
    n("Cast", ["minPosRow"], "mpr_i", to=TI32); n("Cast", ["minPosCol"], "mpc_i", to=TI32)
    n("Equal", ["mpr_i", "i1"], "thinR"); n("Equal", ["mpc_i", "i1"], "thinC")
    n("Or", ["thinR", "thinC"], "thinB"); n("Cast", ["thinB"], "thin", to=FT)
    n("Mul", ["thin", "pos"], "flraw"); n("Mul", ["flraw", "m0"], "isFly")
    n("Sub", ["pos", "isFly"], "frraw"); n("Mul", ["frraw", "m0"], "isFrog")

    # frog/fly colour = channel index (int32 [1] scalars for Gather + uint8 for L)
    n("Mul", ["isFrog", "cidx"], "fcp"); n("ReduceSum", ["fcp"], "frogIdxf", axes=[1, 2, 3], keepdims=0)
    n("Cast", ["frogIdxf"], "frogIdx", to=TI32)                 # [1]
    n("Cast", ["frogIdxf"], "frogColU", to=TU8)
    n("Mul", ["isFly", "cidx"], "flp"); n("ReduceSum", ["flp"], "flyIdxf", axes=[1, 2, 3], keepdims=0)
    n("Cast", ["flyIdxf"], "flyIdx", to=TI32)
    n("Cast", ["flyIdxf"], "flyColU", to=TU8)

    # ---- frog/fly projections via Gather (no [1,10,30,*] products) ----
    n("Gather", ["rowSum", "frogIdx"], "frogRsum", axis=1)      # [1,1,30,1]
    n("Gather", ["rowSum", "flyIdx"], "flyRsum", axis=1)
    n("Gather", ["colSum", "frogIdx"], "frogCsum", axis=1)      # [1,1,1,30]
    n("Gather", ["colSum", "flyIdx"], "flyCsum", axis=1)
    n("Greater", ["frogRsum", "half"], "frogRb"); n("Cast", ["frogRb"], "frogR", to=FT)
    n("Greater", ["flyRsum", "half"], "flyRb"); n("Cast", ["flyRb"], "flyR", to=FT)
    n("Greater", ["frogCsum", "half"], "frogCb"); n("Cast", ["frogCb"], "frogC", to=FT)
    n("Greater", ["flyCsum", "half"], "flyCb"); n("Cast", ["flyCb"], "flyC", to=FT)

    # thickness on each axis = min positive per-line fly count
    n("Where", ["flyCb", "flyCsum", "big"], "fcw"); n("ReduceMin", ["fcw"], "Tr", axes=[2, 3], keepdims=1)
    n("Where", ["flyRb", "flyRsum", "big"], "frw"); n("ReduceMin", ["frw"], "Tc", axes=[2, 3], keepdims=1)

    # moved axis = projections disjoint
    n("Mul", ["frogR", "flyR"], "ovR"); n("ReduceSum", ["ovR"], "ovRs", axes=[2, 3], keepdims=1)
    n("Mul", ["frogC", "flyC"], "ovC"); n("ReduceSum", ["ovC"], "ovCs", axes=[2, 3], keepdims=1)
    n("Greater", ["ovRs", "half"], "ovRb2"); n("Cast", ["ovRb2"], "ovRf", to=FT); n("Sub", ["one", "ovRf"], "moved_rows")
    n("Greater", ["ovCs", "half"], "ovCb2"); n("Cast", ["ovCb2"], "ovCf", to=FT); n("Sub", ["one", "ovCf"], "moved_cols")

    # frog/fly edges
    n("Mul", ["frogR", "ri"], "fRri"); n("ReduceMax", ["fRri"], "fr_rmax", axes=[2, 3], keepdims=1)
    n("Sub", ["one", "frogR"], "frRi"); n("Mul", ["frRi", "big"], "frRb"); n("Add", ["fRri", "frRb"], "fRrib")
    n("ReduceMin", ["fRrib"], "fr_rmin", axes=[2, 3], keepdims=1)
    n("Mul", ["frogC", "ci"], "fCci"); n("ReduceMax", ["fCci"], "fr_cmax", axes=[2, 3], keepdims=1)
    n("Sub", ["one", "frogC"], "frCi"); n("Mul", ["frCi", "big"], "frCb"); n("Add", ["fCci", "frCb"], "fCcib")
    n("ReduceMin", ["fCcib"], "fr_cmin", axes=[2, 3], keepdims=1)
    n("Mul", ["flyR", "ri"], "lRri"); n("Sub", ["one", "flyR"], "lRi"); n("Mul", ["lRi", "big"], "lRb")
    n("Add", ["lRri", "lRb"], "lRrib"); n("ReduceMin", ["lRrib"], "fl_rmin", axes=[2, 3], keepdims=1)
    n("Mul", ["flyC", "ci"], "lCci"); n("Sub", ["one", "flyC"], "lCi"); n("Mul", ["lCi", "big"], "lCb")
    n("Add", ["lCci", "lCb"], "lCcib"); n("ReduceMin", ["lCcib"], "fl_cmin", axes=[2, 3], keepdims=1)

    def run_mask(name, fl_min, fr_max, fr_min, T, ramp):
        n("Greater", [fl_min, fr_max], name + "_bb"); n("Cast", [name + "_bb"], name + "_be", to=FT)
        n("Sub", ["one", name + "_be"], name + "_ab")
        n("Add", [fr_max, "one"], name + "_lb"); n("Add", [fr_max, T], name + "_hb")
        n("Sub", [fr_min, T], name + "_la"); n("Sub", [fr_min, "one"], name + "_ha")
        n("Mul", [name + "_be", name + "_lb"], name + "_l1"); n("Mul", [name + "_ab", name + "_la"], name + "_l2")
        n("Add", [name + "_l1", name + "_l2"], name + "_lo")
        n("Mul", [name + "_be", name + "_hb"], name + "_h1"); n("Mul", [name + "_ab", name + "_ha"], name + "_h2")
        n("Add", [name + "_h1", name + "_h2"], name + "_hi")
        n("Sub", [name + "_lo", "one"], name + "_lom"); n("Add", [name + "_hi", "one"], name + "_hip")
        n("Greater", [ramp, name + "_lom"], name + "_ge"); n("Greater", [name + "_hip", ramp], name + "_le")
        n("And", [name + "_ge", name + "_le"], name + "_msk"); n("Cast", [name + "_msk"], name, to=FT)

    run_mask("runRow", "fl_rmin", "fr_rmax", "fr_rmin", "Tr", "ri")
    run_mask("runCol", "fl_cmin", "fr_cmax", "fr_cmin", "Tc", "ci")

    n("Mul", ["moved_rows", "runRow"], "mrR"); n("Sub", ["one", "moved_rows"], "kR")
    n("Mul", ["kR", "flyR"], "krR"); n("Add", ["mrR", "krR"], "flyOutRow")
    n("Mul", ["moved_cols", "runCol"], "mcC"); n("Sub", ["one", "moved_cols"], "kC")
    n("Mul", ["kC", "flyC"], "kcC"); n("Add", ["mcC", "kcC"], "flyOutCol")

    # grid extent
    n("ReduceMax", ["input"], "gRowf", axes=[1, 3], keepdims=1); n("Cast", ["gRowf"], "gRow", to=FT)
    n("ReduceMax", ["input"], "gColf", axes=[1, 2], keepdims=1); n("Cast", ["gColf"], "gCol", to=FT)

    # ---- label map ----
    n("Greater", ["frogR", "half"], "frB"); n("Greater", ["frogC", "half"], "fcB"); n("And", ["frB", "fcB"], "frogBox")
    n("Greater", ["flyOutRow", "half"], "flrB"); n("Greater", ["flyOutCol", "half"], "flcB"); n("And", ["flrB", "flcB"], "flyBox")
    n("Greater", ["gRow", "half"], "grB"); n("Greater", ["gCol", "half"], "gcB"); n("And", ["grB", "gcB"], "gridBox")

    init("u0", np.array(0, np.uint8), np.uint8)
    init("u10", np.array(10, np.uint8), np.uint8)
    n("Where", ["gridBox", "u0", "u10"], "Lg")
    n("Where", ["flyBox", "flyColU", "Lg"], "Lf")
    n("Where", ["frogBox", "frogColU", "Lf"], "L")
    init("chan", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8)
    n("Equal", ["L", "chan"], "output")

    x = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", onnx.TensorProto.BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task234", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 11)])
