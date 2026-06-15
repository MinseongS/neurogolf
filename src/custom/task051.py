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
    # grids are always <=20x20 anchored top-left, so the active region fits a
    # 20x20 working canvas.  All per-cell planes live in 20x20 (400 elems),
    # then L is padded back to 30x30 (sentinel) just before the final Equal.
    CW = 20
    init("rowidx", np.arange(CW).reshape(1, 1, CW, 1), np.float16)
    init("colidx", np.arange(CW).reshape(1, 1, 1, CW), np.float16)
    init("chan", np.arange(10).reshape(1, 10, 1, 1), np.uint8)
    # non-background channel gate [1,10,1,1] : 0 for ch0, 1 for ch1..9
    nonbg = np.ones((1, 10, 1, 1)); nonbg[0, 0, 0, 0] = 0
    init("nonbg", nonbg, np.float32)
    init("sent15", np.array(15.0), np.float16)
    init("zero16", np.array(0.0), np.float16)
    # crop a [1,1,30,30] plane -> [1,1,CW,CW] (top-left) via negative Pad
    init("crop", np.array([0, 0, 0, 0, 0, 0, CW - 30, CW - 30], np.int64),
         np.int64)
    # pad L [1,1,CW,CW] -> [1,1,30,30] with sentinel (off-grid -> all false)
    init("padO", np.array([0, 0, 0, 0, 0, 0, 30 - CW, 30 - CW], np.int64),
         np.int64)
    init("sentU8", np.array(15), np.uint8)
    # crop row-profile [1,10,30,1] -> [1,10,CW,1]  (spatial axis 2)
    init("cropR", np.array([0, 0, 0, 0, 0, 0, CW - 30, 0], np.int64), np.int64)
    # crop col-profile [1,10,1,30] -> [1,10,1,CW]  (spatial axis 3)
    init("cropC", np.array([0, 0, 0, 0, 0, 0, 0, CW - 30], np.int64), np.int64)

    # =================================================================
    # ALL per-cell work is in fp16 (values are small integers, exact).
    # The only [1,1,30,30] planes that survive are V16, ingrid, the masks,
    # one beam outer-product and L.  Everything else is 1-D (30 elems) or
    # channel-space (10 elems) and costs almost nothing.
    # =================================================================

    # ---- per-cell colour index V (crop to CWxCW, then fp16) ----
    n("Conv", ["input", "colW"], "V32")                      # fp32 [1,1,30,30]
    n("Pad", ["V32", "crop"], "Vc")                          # [1,1,CW,CW] fp32
    n("Cast", ["Vc"], "V", to=F16)                           # fp16 plane CWxCW
    n("Less", ["V", "eps16"], "notpresB")                    # background cell
    n("Cast", ["notpresB"], "notpres", to=F16)

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

    # =================================================================
    # CHANNEL-SPACE ROW/COL PROFILES — collapse to 1-D BEFORE selecting the
    # tip/triangle colour, so NO [1,1,30,30] tip/triangle mask plane is ever
    # built.  rowprof[1,10,30,1] / colprof[1,10,1,30] are only 300 elems.
    # =================================================================
    n("ReduceSum", ["input"], "rowprof30", axes=[3], keepdims=1)  # [1,10,30,1]
    n("ReduceSum", ["input"], "colprof30", axes=[2], keepdims=1)  # [1,10,1,30]
    n("Pad", ["rowprof30", "cropR"], "rowprof32")            # [1,10,CW,1]
    n("Pad", ["colprof30", "cropC"], "colprof32")            # [1,10,1,CW]
    n("Cast", ["rowprof32"], "rowprof", to=F16)
    n("Cast", ["colprof32"], "colprof", to=F16)
    n("Cast", ["tipchan"], "tipchan16", to=F16)
    n("Cast", ["trichan"], "trichan16", to=F16)

    # in-grid extent (1-D): the grid is a solid HxW rectangle anchored top-left,
    # so in-grid = (row < H) AND (col < W).  rowany / colany detect occupied
    # rows / cols straight from the profiles -> NO in-grid conv plane needed.
    n("ReduceSum", ["rowprof"], "rowanyS", axes=[1], keepdims=1)  # [1,1,CW,1]
    n("Greater", ["rowanyS", "eps16"], "rowanyB")            # bool (row < H)
    n("ReduceSum", ["colprof"], "colanyS", axes=[1], keepdims=1)  # [1,1,1,CW]
    n("Greater", ["colanyS", "eps16"], "colanyB")            # bool (col < W)
    n("And", ["rowanyB", "colanyB"], "ingridB")              # [1,1,CW,CW] bool

    # tip 1-D occupancy vectors (single 1 at the tip's row / col)
    n("Mul", ["rowprof", "tipchan16"], "tr10"); n("ReduceSum", ["tr10"], "tr16", axes=[1], keepdims=1)
    n("Mul", ["colprof", "tipchan16"], "tc10"); n("ReduceSum", ["tc10"], "tc16", axes=[1], keepdims=1)
    n("Mul", ["tr16", "rowidx"], "tyv"); n("ReduceSum", ["tyv"], "ty", axes=[2, 3], keepdims=1)
    n("Mul", ["tc16", "colidx"], "txv"); n("ReduceSum", ["txv"], "tx", axes=[2, 3], keepdims=1)

    # triangle 1-D count-per-row / count-per-col vectors
    n("Mul", ["rowprof", "trichan16"], "Rr10"); n("ReduceSum", ["Rr10"], "Rr16", axes=[1], keepdims=1)
    n("Mul", ["colprof", "trichan16"], "Rc10"); n("ReduceSum", ["Rc10"], "Rc16", axes=[1], keepdims=1)

    # centroid (cr,cc) = sum(idx*count)/sum(count)
    n("ReduceSum", ["Rr16"], "trn", axes=[2, 3], keepdims=1)
    n("Mul", ["Rr16", "rowidx"], "trrv"); n("ReduceSum", ["trrv"], "trrs", axes=[2, 3], keepdims=1)
    n("Div", ["trrs", "trn"], "cr")
    n("Mul", ["Rc16", "colidx"], "trcv"); n("ReduceSum", ["trcv"], "trcs", axes=[2, 3], keepdims=1)
    n("Div", ["trcs", "trn"], "cc")

    # orientation: span = #occupied rows/cols = count of nonzero profile entries
    n("Greater", ["Rr16", "eps16"], "rocB"); n("Cast", ["rocB"], "roc", to=F16)
    n("ReduceSum", ["roc"], "rspan", axes=[2, 3], keepdims=1)
    n("Greater", ["Rc16", "eps16"], "cocB"); n("Cast", ["cocB"], "coc", to=F16)
    n("ReduceSum", ["coc"], "cspan", axes=[2, 3], keepdims=1)
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
    n("Add", ["lc_v", "lc_h"], "linecol0")                   # [1,1,1,CW]
    # fold the tip colour into the 1-D column vector (free) so the outer
    # product directly produces the beam's COLOUR contribution.
    n("Mul", ["linecol0", "tipcol"], "linecol")              # [1,1,1,CW]
    n("Mul", ["linerow", "linecol"], "beamLine")             # [1,1,CW,CW] fp16

    # ---- in-grid background = in-grid AND background (one Where) ----
    n("Where", ["ingridB", "notpres", "zero16"], "bgin")

    # ---- beam lands only on in-grid background; beamLine already carries the
    #      tip colour, so its product with bgin IS the colour contribution. ----
    n("Mul", ["beamLine", "bgin"], "beamcol")
    n("Add", ["V", "beamcol"], "Lf0")                        # colour value plane

    # ---- off-grid -> sentinel 15 (so the final Equal yields all-false) ----
    n("Where", ["ingridB", "Lf0", "sent15"], "Lf")
    n("Cast", ["Lf"], "Lc", to=U8)                           # [1,1,CW,CW] uint8
    # pad CWxCW -> 30x30 with sentinel 15 (the off-canvas border is off-grid)
    n("Pad", ["Lc", "padO", "sentU8"], "L", mode="constant")  # [1,1,30,30]
    n("Equal", ["L", "chan"], "output")                      # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task051", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
