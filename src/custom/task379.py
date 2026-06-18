"""task379 (ARC-AGI ecdecbb3) — "red dots shoot rays to cyan lines, stamp 3x3 boxes".

Rule (from the generator, canonical = horizontal-line orientation; xpose flips it):
  * There are 1-2 full-width horizontal CYAN(8) lines.
  * Each column has at most one RED(2) dot.
  * Each dot shoots a ray (painting red along its column) toward the NEAREST
    line above it AND the NEAREST line below it (a line strictly between the dot
    and a farther line blocks the farther one).
  * Where a ray reaches a line at (L, c): paint the inclusive column segment
    [dot..L] red, then stamp a 3x3 CYAN box centred at (L,c), then set the box
    centre (L,c) RED.  Paint order: ray-red < box-cyan < centre-red.

Closed-form (verified 0/266 over all stored instances):
  Per column scalars dotrow[c], coldot[c]; per-row lineflag.
  Ldown[c] = min line-row > dotrow ; Lup[c] = max line-row < dotrow.
  rayred  = column segments [dot..Ldown] and [Lup..dot]  (OR the dot itself)
  boxctr  = (row==Ldown & hasdown) | (row==Lup & hasup)
  box     = MaxPool 3x3 of boxctr
  colour  = cyan(lines) -> red(ray) -> cyan(box) -> red(boxctr)   (priority)

Orientation: build the canonical (horizontal-line) branch on (redin, cyanin)
and the transposed branch on their Transposes, then select with the scalar flag
hor = "does any full-width cyan ROW exist".  Final colour-index plane (values in
{0,2,8}) is uint8; Equal(L, arange[1,10,1,1]) -> BOOL into the FREE output.
"""

import numpy as np
from onnx import helper, numpy_helper, TensorProto

from ..harness import IR_VERSION

F = TensorProto.FLOAT
B = TensorProto.BOOL
U8 = TensorProto.UINT8


def build(task):
    inits, nodes = [], []

    def init(name, arr, npdtype):
        inits.append(numpy_helper.from_array(
            np.ascontiguousarray(arr, dtype=npdtype), name))
        return name

    def n(op, ins, out, **attrs):
        nodes.append(helper.make_node(op, ins, [out], **attrs))
        return out

    Ff = TensorProto.FLOAT16
    WK = 20  # active working canvas (generator bounds width,height in [12,20])
    # constants (fp16 for the cheap working planes; all integer-valued so exact)
    init("rowidx", np.arange(WK).reshape(1, 1, WK, 1), np.float16)   # [1,1,WK,1]
    init("BIG", np.array(1e4, np.float16), np.float16)
    init("nBIG", np.array(-1e4, np.float16), np.float16)
    init("half", np.array(0.5, np.float16), np.float16)
    init("half32", np.array(0.5, np.float32), np.float32)
    init("BIG2", np.array(5e3, np.float16), np.float16)
    # slice extents: crop channel k AND the WKxWK active region in ONE Slice
    init("axes123", np.array([1, 2, 3], np.int64), np.int64)
    init("st_r", np.array([2, 0, 0], np.int64), np.int64)
    init("en_r", np.array([3, WK, WK], np.int64), np.int64)
    init("st_c", np.array([8, 0, 0], np.int64), np.int64)
    init("en_c", np.array([9, WK, WK], np.int64), np.int64)
    init("st_0", np.array([0, 0, 0], np.int64), np.int64)
    init("en_0", np.array([1, WK, WK], np.int64), np.int64)
    init("chan", np.arange(10).reshape(1, 10, 1, 1), np.uint8)
    # pad uint8 label [1,1,WK,WK] -> [1,1,30,30] with sentinel 99
    init("padO", np.array([0, 0, 0, 0, 0, 0, 30 - WK, 30 - WK], np.int64), np.int64)
    init("sentU8", np.array(99, np.uint8), np.uint8)
    init("u0", np.array(0, np.uint8), np.uint8)
    init("u2", np.array(2, np.uint8), np.uint8)
    init("u8", np.array(8, np.uint8), np.uint8)

    # extract+crop the red/cyan/bg planes in ONE Slice each -> [1,1,WK,WK] fp32
    n("Slice", ["input", "st_r", "en_r", "axes123"], "redinC")   # red
    n("Slice", ["input", "st_c", "en_c", "axes123"], "cyaninC")  # cyan
    n("Slice", ["input", "st_0", "en_0", "axes123"], "bgC")      # ch0 (in-grid bg)
    # uint8 masks (400B planes) for cheap transpose + orientation select
    # (ORT Where supports uint8 but not bool)
    n("Greater", ["redinC", "half32"], "redmB")
    n("Greater", ["cyaninC", "half32"], "cymB")
    n("Cast", ["redmB"], "redmU", to=U8)
    n("Cast", ["cymB"], "cymU", to=U8)
    n("Transpose", ["redmU"], "redmT", perm=[0, 1, 3, 2])
    n("Transpose", ["cymU"], "cymT", perm=[0, 1, 3, 2])

    def branch(redB, cyB, tag):
        """Emit the canonical (horizontal-line) pipeline from bool red/cyan
        masks; return uint8 colour-index plane [1,1,WK,WK] (values {0,2,8})."""
        def t(s):
            return f"{tag}_{s}"

        # redB/cyB are uint8 masks; derive fp16 (for reductions) and bool (for
        # Where conditions / Or)
        n("Cast", [redB], t("redf"), to=Ff)                            # fp16 0/1
        n("Cast", [cyB], t("cyf"), to=Ff)
        n("Cast", [redB], t("redBb"), to=B)                            # bool
        n("Cast", [cyB], t("cyBb"), to=B)
        # per-row cyan count -> line flag (full-width rows)
        n("ReduceSum", [t("cyf")], t("rowcyan"), axes=[3], keepdims=1)   # [1,1,WK,1]
        n("ReduceMax", [t("rowcyan")], t("W"), axes=[2, 3], keepdims=1)
        n("Equal", [t("rowcyan"), t("W")], t("lineB"))                 # [1,1,WK,1] bool
        # per-col dot presence + dot row (ArgMax avoids a full rred plane)
        n("ReduceMax", [t("redf")], t("coldot"), axes=[2], keepdims=1)  # [1,1,1,WK]
        n("ArgMax", [t("redf")], t("dotrowI"), axis=2, keepdims=1)     # int64 [1,1,1,WK]
        n("Cast", [t("dotrowI")], t("dotrow"), to=Ff)                  # [1,1,1,WK] fp16

        # line rows as scalars (<=2 lines).  These reductions are over the tiny
        # [1,1,WK,1] lineB vector, so NO WKxWK plane is built.
        n("Where", [t("lineB"), "rowidx", "BIG"], t("lminv"))          # [1,1,WK,1]
        n("ReduceMin", [t("lminv")], t("Lmin"), axes=[2, 3], keepdims=1)  # scalar
        n("Where", [t("lineB"), "rowidx", "nBIG"], t("lmaxv"))
        n("ReduceMax", [t("lmaxv")], t("Lmax"), axes=[2, 3], keepdims=1)  # scalar

        # Ldown[c] = nearest line BELOW dot ; Lup[c] = nearest line ABOVE dot.
        # All ops on [1,1,1,WK] dotrow vector (160B), no full plane.
        n("Less", [t("dotrow"), t("Lmin")], t("d_ltmin"))              # dot above both
        n("Less", [t("dotrow"), t("Lmax")], t("d_ltmax"))             # dot above Lmax
        n("Where", [t("d_ltmax"), t("Lmax"), "BIG"], t("Ldn0"))       # below Lmax -> Lmax
        n("Where", [t("d_ltmin"), t("Lmin"), t("Ldn0")], t("Ldown"))  # below Lmin -> Lmin
        n("Greater", [t("dotrow"), t("Lmax")], t("d_gtmax"))          # dot below both
        n("Greater", [t("dotrow"), t("Lmin")], t("d_gtmin"))         # dot below Lmin
        n("Where", [t("d_gtmin"), t("Lmin"), "nBIG"], t("Lup0"))     # above Lmin -> Lmin
        n("Where", [t("d_gtmax"), t("Lmax"), t("Lup0")], t("Lup"))   # above Lmax -> Lmax

        # has reach (and column has a dot)
        n("Less", [t("Ldown"), "BIG2"], t("dexB"))
        n("Greater", [t("Lup"), "nBIG"], t("uexB"))                    # found
        n("Greater", [t("coldot"), "half"], t("hasdotB"))
        n("And", [t("dexB"), t("hasdotB")], t("hasdownB"))             # [1,1,1,WK] bool
        n("And", [t("uexB"), t("hasdotB")], t("hasupB"))
        # row-vs-dot comparison planes still needed for the ray extent
        n("Greater", ["rowidx", t("dotrow")], t("Lgt"))                # [1,1,WK,WK]
        n("Less", ["rowidx", t("dotrow")], t("Llt"))

        # rayred: [dotrow..Ldown] for down, [Lup..dotrow] for up
        # down: rowidx>=dotrow AND rowidx<=Ldown
        n("Not", [t("Llt")], t("rge_dot"))                             # rowidx>=dotrow
        n("Not", [t("Lgt")], t("rle_dot"))                             # rowidx<=dotrow
        # rowidx<=Ldown
        n("Greater", ["rowidx", t("Ldown")], t("gt_ld"))
        n("Not", [t("gt_ld")], t("le_ld"))
        n("And", [t("rge_dot"), t("le_ld")], t("draw0"))
        n("And", [t("draw0"), t("hasdownB")], t("drawn"))              # down ray bool
        # rowidx>=Lup
        n("Less", ["rowidx", t("Lup")], t("lt_lu"))
        n("Not", [t("lt_lu")], t("ge_lu"))
        n("And", [t("rle_dot"), t("ge_lu")], t("uraw0"))
        n("And", [t("uraw0"), t("hasupB")], t("urawn"))                # up ray bool
        n("Or", [t("drawn"), t("urawn")], t("rayB0"))
        n("Or", [t("rayB0"), t("redBb")], t("rayB"))                   # incl dot cell

        # box centres: (rowidx==Ldown & hasdown) | (rowidx==Lup & hasup)
        n("Equal", ["rowidx", t("Ldown")], t("eqd"))
        n("And", [t("eqd"), t("hasdownB")], t("bcd"))
        n("Equal", ["rowidx", t("Lup")], t("equ"))
        n("And", [t("equ"), t("hasupB")], t("bcu"))
        n("Or", [t("bcd"), t("bcu")], t("bcB"))                        # [1,1,30,30] bool
        n("Cast", [t("bcB")], t("bcF"), to=TensorProto.FLOAT16)
        # 3x3 dilation via MaxPool (SAME pad)
        n("MaxPool", [t("bcF")], t("boxF"), kernel_shape=[3, 3],
          pads=[1, 1, 1, 1], strides=[1, 1])
        n("Greater", [t("boxF"), "half"], t("boxB"))

        # compose colour-index plane in UINT8 (priority via nested Where; all
        # operands uint8 -> 400 B planes, no fp32 PrecisionFreeCast upcast).
        # value: bcB?2 : boxB?8 : rayB?2 : cyB?8 : 0
        n("Where", [t("cyBb"), "u8", "u0"], t("L0"))                   # uint8
        n("Where", [t("rayB"), "u2", t("L0")], t("L1"))
        n("Where", [t("boxB"), "u8", t("L1")], t("L2"))
        n("Where", [t("bcB"), "u2", t("L2")], t("L3"))
        return t("L3")

    # orientation flag (scalar): a horizontal line has cyan-count == gridwidth in
    # ONE row; vertical line -> a column has cyan == gridheight. Compare the max
    # cyan-per-row with the max cyan-per-col.
    n("Cast", ["cymB"], "cymf", to=Ff)
    n("ReduceSum", ["cymf"], "rcnt", axes=[3], keepdims=1)
    n("ReduceMax", ["rcnt"], "maxrow", axes=[2, 3], keepdims=1)        # scalar
    n("ReduceSum", ["cymf"], "ccnt", axes=[2], keepdims=1)
    n("ReduceMax", ["ccnt"], "maxcol", axes=[2, 3], keepdims=1)        # scalar
    n("Greater", ["maxrow", "maxcol"], "horB")                        # scalar bool

    # select canonical (horizontal-line) uint8 masks, run ONE branch
    n("Where", ["horB", "redmU", "redmT"], "redC")     # uint8
    n("Where", ["horB", "cymU", "cymT"], "cyanC")
    Lc = branch("redC", "cyanC", "h")
    n("Transpose", [Lc], "LcT", perm=[0, 1, 3, 2])
    n("Where", ["horB", Lc, "LcT"], "L")                             # uint8 [1,1,WK,WK]

    # in-grid mask (cropped): bg(ch0) OR red OR cyan set -> in grid; else 99
    # (reuse redmB/cymB bool masks; bool Ors at 400B beat fp32 Adds at 1600B)
    n("Greater", ["bgC", "half32"], "bgB")
    n("Or", ["bgB", "redmB"], "ig0B")
    n("Or", ["ig0B", "cymB"], "ingB")
    n("Where", ["ingB", "L", "sentU8"], "Lm")                       # off-grid->99 (u8)
    # pad back to 30x30 with sentinel 99 (off-active-region -> no channel)
    n("Pad", ["Lm", "padO", "sentU8"], "Lpad")                      # [1,1,30,30] u8
    n("Equal", ["Lpad", "chan"], "output")                          # [1,10,30,30] BOOL

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", B, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task379", [x], [y], inits)
    return helper.make_model(g, ir_version=IR_VERSION,
                             opset_imports=[helper.make_opsetid("", 11)])
