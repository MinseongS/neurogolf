"""task051 (ARC-AGI 25d487eb) — "laser beam from an arrowhead".

Rule (from the generator, pre-gravity):
  * A solid downward-narrowing TRIANGLE (arrowhead) of colour c0 is drawn,
    widest at its base (width 2*depth-1) and narrowing to a single apex cell
    (depth in {3,4}).
  * A single TIP pixel of colour c1 sits at the centre of the base.
  * A BEAM of colour c1 fires from the apex out to the grid edge, filling the
    apex-side ray on the triangle's axis of symmetry.
  * apply_gravity then rotates/flips the whole figure into one of 4 cardinal
    orientations.  So in the input the arrowhead may point up/down/left/right.

  The INPUT shows the triangle + tip only; the OUTPUT additionally paints the
  beam.  Task = detect the figure and add the beam.

Recovery (fully reduction-based, verified 0 errors / 3000 fresh instances):
  * Two colours present.  TIP colour = the channel whose pixel COUNT == 1;
    TRIANGLE colour = the channel whose count > 1.  (counts via
    ReduceSum(input,[2,3]) -> [1,10,1,1].)
  * V[1,1,30,30] = per-cell colour index (1x1 Conv of one-hot with [0..9]).
  * tipmask / trimask [1,1,30,30] = the cells of each colour (sum over the
    selected channels).  tip colour value & tip (row,col) are scalars from
    weighted reductions of tipmask.  triangle centroid (cr,cc) likewise.
  * orientation: triangle is solid, so #rows-occupied = row-span, #cols =
    col-span.  Base is the WIDE edge => beam axis is the SHORTER span.
    vertical (beam up/down) iff cspan > rspan.
  * direction: tip sits at the base, centroid is shifted toward the apex, so
    the beam fires toward the side of the tip where the centroid lies:
    vertical -> up iff cr < ty ; horizontal -> left iff cc < tx.
  * beam mask = (on the axis line through the tip) AND (in the apex half-plane
    beyond the tip) AND (cell currently background).  Only background cells are
    painted, so the triangle is never overwritten.
  * L[1,1,30,30] uint8 label = V + tipcolour*beam ; sentinel handling not
    needed (off-grid border is background -> channel 0).  output = Equal(L,
    [0..9]) -> BOOL into the FREE output tensor.

Memory: the only sizeable intermediates are a handful of [1,1,30,30] planes
(900 B uint8 / 3600 B fp32 each).  No [1,10,30,30] is ever materialised.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION


def build(task):
    inits, nodes = [], []

    def init(name, arr, npdtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=npdtype), name))
        return name

    def n(op, ins, out, **attrs):
        if isinstance(out, str):
            nodes.append(helper.make_node(op, ins, [out], **attrs))
            return out
        nodes.append(helper.make_node(op, ins, out, **attrs))
        return out

    F = TensorProto.FLOAT
    F16 = TensorProto.FLOAT16
    U8 = TensorProto.UINT8
    B = TensorProto.BOOL

    # ---- initializers ----
    # colour-index 1x1 conv weight [out=1,in=10,1,1] = [0,1,..,9]
    init("colW", np.arange(10).reshape(1, 10, 1, 1), np.float32)
    # per-channel arange [1,10,1,1] for selecting tip colour value
    init("chvals", np.arange(10).reshape(1, 10, 1, 1), np.float32)
    init("one", np.array(1.0), np.float32)
    init("onehalf", np.array(1.5), np.float32)
    init("eps", np.array(0.5), np.float32)
    init("eps16", np.array(0.5), np.float16)
    init("one16", np.array(1.0), np.float16)
    init("rowidx", np.arange(30).reshape(1, 1, 30, 1), np.float16)
    init("colidx", np.arange(30).reshape(1, 1, 1, 30), np.float16)
    init("chan", np.arange(10).reshape(1, 10, 1, 1), np.uint8)
    # non-background channel gate [1,10,1,1] : 0 for ch0, 1 for ch1..9
    nonbg = np.ones((1, 10, 1, 1)); nonbg[0, 0, 0, 0] = 0
    init("nonbg", nonbg, np.float32)
    init("sent15", np.array(15.0), np.float16)

    # =================================================================
    # ALL per-cell work is in fp16 (values are small integers, exact).
    # The only [1,1,30,30] planes that survive are V16, ingrid, the masks,
    # one beam outer-product and L.  Everything else is 1-D (30 elems) or
    # channel-space (10 elems) and costs almost nothing.
    # =================================================================

    # ---- per-cell colour index V [1,1,30,30] (fp16) ----
    n("Conv", ["input", "colW"], "V32")                      # fp32 colour 0..9
    n("Cast", ["V32"], "V", to=F16)                          # fp16 plane

    # ---- in-grid mask (fp16): off-grid cells are all-channels-0. ----
    n("ReduceSum", ["input"], "ingridS", axes=[1], keepdims=1)  # fp32 [1,1,30,30]
    n("Greater", ["ingridS", "eps"], "ingridB")
    n("Cast", ["ingridB"], "ingrid", to=F16)

    # ---- channel counts -> tip / triangle colour VALUES (channel space) ----
    n("ReduceSum", ["input"], "cnt", axes=[2, 3], keepdims=1)  # [1,10,1,1] fp32
    n("Sub", ["cnt", "one"], "cm1")
    n("Mul", ["cm1", "cm1"], "cm1sq")
    n("Less", ["cm1sq", "eps"], "tipchanB")                  # count == 1
    n("Cast", ["tipchanB"], "tipchan0", to=F)
    n("Mul", ["tipchan0", "nonbg"], "tipchan")               # exclude ch0
    n("Greater", ["cnt", "onehalf"], "trichanB")             # count > 1
    n("Cast", ["trichanB"], "trichan0", to=F)
    n("Mul", ["trichan0", "nonbg"], "trichan")
    n("Mul", ["chvals", "tipchan"], "tipcv")
    n("ReduceSum", ["tipcv"], "tipcol32", axes=[1], keepdims=1)  # scalar fp32
    n("Cast", ["tipcol32"], "tipcol", to=F16)
    n("Mul", ["chvals", "trichan"], "tricv")
    n("ReduceSum", ["tricv"], "tricol32", axes=[1], keepdims=1)  # scalar fp32
    n("Cast", ["tricol32"], "tricol", to=F16)

    # ---- tip / triangle cell masks [1,1,30,30] fp16 = (V == colour) ----
    n("Sub", ["V", "tipcol"], "vmt"); n("Mul", ["vmt", "vmt"], "vmt2")
    n("Less", ["vmt2", "eps16"], "tipmaskB"); n("Cast", ["tipmaskB"], "tipm0", to=F16)
    n("Mul", ["tipm0", "ingrid"], "tipmask")
    n("Sub", ["V", "tricol"], "vmr"); n("Mul", ["vmr", "vmr"], "vmr2")
    n("Less", ["vmr2", "eps16"], "trimaskB"); n("Cast", ["trimaskB"], "trim0", to=F16)
    n("Mul", ["trim0", "ingrid"], "trimask")

    # ---- tip (ty,tx) scalars : reduce mask*idx, all fp16 (values < 2048) ----
    n("Mul", ["tipmask", "rowidx"], "tym")
    n("ReduceSum", ["tym"], "ty", axes=[2, 3], keepdims=1)
    n("Mul", ["tipmask", "colidx"], "txm")
    n("ReduceSum", ["txm"], "tx", axes=[2, 3], keepdims=1)

    # ---- triangle centroid (cr,cc) fp16 ----
    n("ReduceSum", ["trimask"], "trin", axes=[2, 3], keepdims=1)
    n("Mul", ["trimask", "rowidx"], "trr")
    n("ReduceSum", ["trr"], "trrs", axes=[2, 3], keepdims=1)
    n("Div", ["trrs", "trin"], "cr")
    n("Mul", ["trimask", "colidx"], "trc")
    n("ReduceSum", ["trc"], "trcs", axes=[2, 3], keepdims=1)
    n("Div", ["trcs", "trin"], "cc")

    # ---- orientation: row-span vs col-span of triangle (1-D reductions) ----
    n("ReduceMax", ["trimask"], "rowocc", axes=[3], keepdims=1)  # [1,1,30,1]
    n("ReduceSum", ["rowocc"], "rspan", axes=[2, 3], keepdims=1)
    n("ReduceMax", ["trimask"], "colocc", axes=[2], keepdims=1)  # [1,1,1,30]
    n("ReduceSum", ["colocc"], "cspan", axes=[2, 3], keepdims=1)
    n("Greater", ["cspan", "rspan"], "vertB")
    n("Cast", ["vertB"], "vert", to=F16)                     # 1=vertical scalar

    # ================= beam direction, built from 1-D vectors (fp16) =======
    # axis lines (1-D):
    n("Sub", ["colidx", "tx"], "dcol"); n("Mul", ["dcol", "dcol"], "dcol2")
    n("Less", ["dcol2", "eps16"], "oncolB"); n("Cast", ["oncolB"], "oncol", to=F16)  # tip col
    n("Sub", ["rowidx", "ty"], "drow"); n("Mul", ["drow", "drow"], "drow2")
    n("Less", ["drow2", "eps16"], "onrowB"); n("Cast", ["onrowB"], "onrow", to=F16)  # tip row

    # vertical half-plane (apex side, rows): up iff cr < ty
    n("Sub", ["ty", "cr"], "tymcr"); n("Greater", ["tymcr", "eps16"], "upB")
    n("Cast", ["upB"], "vup", to=F16)
    n("Less", ["rowidx", "ty"], "rowAboveB"); n("Cast", ["rowAboveB"], "rowAbove", to=F16)
    n("Greater", ["rowidx", "ty"], "rowBelowB"); n("Cast", ["rowBelowB"], "rowBelow", to=F16)
    n("Mul", ["vup", "rowAbove"], "vha")
    n("Sub", ["one16", "vup"], "vdn"); n("Mul", ["vdn", "rowBelow"], "vhb")
    n("Add", ["vha", "vhb"], "vhalf")                        # [1,1,30,1] rows

    # horizontal half-plane (apex side, cols): left iff cc < tx
    n("Sub", ["tx", "cc"], "txmcc"); n("Greater", ["txmcc", "eps16"], "leftB")
    n("Cast", ["leftB"], "hleft", to=F16)
    n("Less", ["colidx", "tx"], "colLeftB"); n("Cast", ["colLeftB"], "colLeft", to=F16)
    n("Greater", ["colidx", "tx"], "colRightB"); n("Cast", ["colRightB"], "colRight", to=F16)
    n("Mul", ["hleft", "colLeft"], "hla")
    n("Sub", ["one16", "hleft"], "hright"); n("Mul", ["hright", "colRight"], "hlb")
    n("Add", ["hla", "hlb"], "hhalf")                        # [1,1,1,30] cols

    # combine 1-D row/col vectors BEFORE the outer product -> single plane:
    #   vertical  -> linerow = vhalf (rows), linecol = oncol (tip column)
    #   horizontal-> linerow = onrow (tip row), linecol = hhalf (cols)
    n("Sub", ["one16", "vert"], "horiz")
    n("Mul", ["vert", "vhalf"], "lr_v"); n("Mul", ["horiz", "onrow"], "lr_h")
    n("Add", ["lr_v", "lr_h"], "linerow")                    # [1,1,30,1]
    n("Mul", ["vert", "oncol"], "lc_v"); n("Mul", ["horiz", "hhalf"], "lc_h")
    n("Add", ["lc_v", "lc_h"], "linecol")                    # [1,1,1,30]
    n("Mul", ["linerow", "linecol"], "beamLine")             # [1,1,30,30] fp16

    # ---- restrict to background & in-grid cells ----
    n("Less", ["V", "eps16"], "bgB"); n("Cast", ["bgB"], "bg", to=F16)
    n("Mul", ["beamLine", "bg"], "beam0")
    n("Mul", ["beam0", "ingrid"], "beam")

    # ---- label map L = V + tipcol*beam ; off-grid -> sentinel 15 ----
    n("Mul", ["beam", "tipcol"], "beamcol")
    n("Add", ["V", "beamcol"], "Lf0")
    n("Sub", ["one16", "ingrid"], "offgrid")
    n("Mul", ["offgrid", "sent15"], "sentadd")
    n("Add", ["Lf0", "sentadd"], "Lf")
    n("Cast", ["Lf"], "L", to=U8)                            # [1,1,30,30] uint8
    n("Equal", ["L", "chan"], "output")                      # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task051", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
